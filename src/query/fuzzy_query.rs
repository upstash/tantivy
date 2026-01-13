use common::BitSet;
use levenshtein_automata::{Distance, LevenshteinAutomatonBuilder, DFA};
use once_cell::sync::OnceCell;
use tantivy_fst::Automaton;

use crate::index::SegmentReader;
use crate::query::phrase_prefix_query::prefix_end;
use crate::query::{
    AutomatonWeight, BitSetDocSet, EnableScoring, Explanation, Query, Scorer, Weight,
};
use crate::schema::{Field, IndexRecordOption, Term, Type};
use crate::TantivyError::InvalidArgument;
use crate::{DocId, DocSet, Score, TantivyError};

/// The Levenshtein automaton a [`FuzzyTermQuery`] matches terms with.
///
/// Obtained from [`FuzzyTermQuery::automaton`], e.g. to stream the term dictionary
/// and collect the terms the query expands to.
pub struct DfaWrapper(pub(crate) DFA);

impl Automaton for DfaWrapper {
    type State = u32;

    fn start(&self) -> Self::State {
        self.0.initial_state()
    }

    fn is_match(&self, state: &Self::State) -> bool {
        match self.0.distance(*state) {
            Distance::Exact(_) => true,
            Distance::AtLeast(_) => false,
        }
    }

    fn can_match(&self, state: &u32) -> bool {
        *state != levenshtein_automata::SINK_STATE
    }

    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        self.0.transition(*state, byte)
    }
}

/// A Fuzzy Query matches all of the documents
/// containing a specific term that is within
/// Levenshtein distance
/// ```rust
/// use tantivy::collector::{Count, TopDocs};
/// use tantivy::query::FuzzyTermQuery;
/// use tantivy::schema::{Schema, TEXT};
/// use tantivy::{doc, Index, IndexWriter, Term};
///
/// fn example() -> tantivy::Result<()> {
///     let mut schema_builder = Schema::builder();
///     let title = schema_builder.add_text_field("title", TEXT);
///     let schema = schema_builder.build();
///     let index = Index::create_in_ram(schema);
///     {
///         let mut index_writer: IndexWriter = index.writer(15_000_000)?;
///         index_writer.add_document(doc!(
///             title => "The Name of the Wind",
///         ))?;
///         index_writer.add_document(doc!(
///             title => "The Diary of Muadib",
///         ))?;
///         index_writer.add_document(doc!(
///             title => "A Dairy Cow",
///         ))?;
///         index_writer.add_document(doc!(
///             title => "The Diary of a Young Girl",
///         ))?;
///         index_writer.commit()?;
///     }
///     let reader = index.reader()?;
///     let searcher = reader.searcher();
///
///     {
///         let term = Term::from_field_text(title, "Diary");
///         let query = FuzzyTermQuery::new(term, 1, true);
///         let (top_docs, count) = searcher.search(&query, &(TopDocs::with_limit(2).order_by_score(), Count)).unwrap();
///         assert_eq!(count, 2);
///         assert_eq!(top_docs.len(), 2);
///     }
///
///     Ok(())
/// }
/// # assert!(example().is_ok());
/// ```
#[derive(Debug, Clone)]
pub struct FuzzyTermQuery {
    /// What term are we searching
    term: Term,
    /// How many changes are we going to allow
    distance: u8,
    /// Should a transposition cost 1 or 2?
    transposition_cost_one: bool,
    /// is a starts with query
    prefix: bool,
}

impl FuzzyTermQuery {
    /// Creates a new Fuzzy Query
    pub fn new(term: Term, distance: u8, transposition_cost_one: bool) -> FuzzyTermQuery {
        FuzzyTermQuery {
            term,
            distance,
            transposition_cost_one,
            prefix: false,
        }
    }

    /// Creates a new Fuzzy Query of the Term prefix
    pub fn new_prefix(term: Term, distance: u8, transposition_cost_one: bool) -> FuzzyTermQuery {
        FuzzyTermQuery {
            term,
            distance,
            transposition_cost_one,
            prefix: true,
        }
    }

    /// Returns the automaton this query matches terms with.
    ///
    /// For a JSON term, it matches the term's text only, not its JSON path.
    ///
    /// ```rust
    /// use tantivy::query::{DfaWrapper, FuzzyTermQuery};
    /// use tantivy::schema::{Schema, TEXT};
    /// use tantivy::Term;
    ///
    /// let mut schema_builder = Schema::builder();
    /// let title = schema_builder.add_text_field("title", TEXT);
    /// let query = FuzzyTermQuery::new(Term::from_field_text(title, "diary"), 1, true);
    /// let _automaton: DfaWrapper = query.automaton().unwrap();
    /// ```
    pub fn automaton(&self) -> crate::Result<DfaWrapper> {
        static AUTOMATON_BUILDER: [[OnceCell<LevenshteinAutomatonBuilder>; 2]; 3] = [
            [OnceCell::new(), OnceCell::new()],
            [OnceCell::new(), OnceCell::new()],
            [OnceCell::new(), OnceCell::new()],
        ];

        let automaton_builder = AUTOMATON_BUILDER
            .get(self.distance as usize)
            .ok_or_else(|| {
                InvalidArgument(format!(
                    "Levenshtein distance of {} is not allowed. Choose a value less than {}",
                    self.distance,
                    AUTOMATON_BUILDER.len()
                ))
            })?
            .get(self.transposition_cost_one as usize)
            .unwrap()
            .get_or_init(|| {
                LevenshteinAutomatonBuilder::new(self.distance, self.transposition_cost_one)
            });

        let term_value = self.term.value();

        let term_text = if term_value.typ() == Type::Json {
            if let Some(json_path_type) = term_value.json_path_type() {
                if json_path_type != Type::Str {
                    return Err(InvalidArgument(format!(
                        "The fuzzy term query requires a string path type for a json term. Found \
                         {json_path_type:?}"
                    )));
                }
            }

            std::str::from_utf8(self.term.serialized_value_bytes()).map_err(|_| {
                InvalidArgument(
                    "Failed to convert json term value bytes to utf8 string.".to_string(),
                )
            })?
        } else {
            term_value.as_str().ok_or_else(|| {
                InvalidArgument("The fuzzy term query requires a string term.".to_string())
            })?
        };
        let automaton = if self.prefix {
            automaton_builder.build_prefix_dfa(term_text)
        } else {
            automaton_builder.build_dfa(term_text)
        };
        Ok(DfaWrapper(automaton))
    }

    fn specialized_weight(&self) -> crate::Result<AutomatonWeight<DfaWrapper>> {
        let automaton = self.automaton()?;
        if let Some((json_path_bytes, _)) = self.term.value().as_json() {
            Ok(AutomatonWeight::new_for_json_path(
                self.term.field(),
                automaton,
                json_path_bytes,
            ))
        } else {
            Ok(AutomatonWeight::new(self.term.field(), automaton))
        }
    }
}

impl Query for FuzzyTermQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        if !enable_scoring.is_scoring_enabled() {
            return Ok(Box::new(self.specialized_weight()?));
        }
        let json_path_bytes = self
            .term
            .value()
            .as_json()
            .map(|(json_path_bytes, _)| json_path_bytes.to_vec().into_boxed_slice());
        Ok(Box::new(FuzzyTermWeight {
            field: self.term.field(),
            automaton: self.automaton()?,
            json_path_bytes,
        }))
    }
}

/// The weight of a [`FuzzyTermQuery`] with scoring enabled. A document scores
/// `boost / (1 + distance)`, where `distance` is the smallest Levenshtein distance among its
/// matching terms.
struct FuzzyTermWeight {
    field: Field,
    automaton: DfaWrapper,
    json_path_bytes: Option<Box<[u8]>>,
}

impl Weight for FuzzyTermWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let max_doc = reader.max_doc();
        let mut matching_docs = BitSet::with_max_value(max_doc);
        let mut distance_plus_one = vec![0u8; max_doc as usize];
        let inverted_index = reader.inverted_index(self.field)?;
        let mut term_stream_builder = inverted_index.terms().search_with_state(&self.automaton);
        if let Some(json_path_bytes) = &self.json_path_bytes {
            term_stream_builder = term_stream_builder.ge(json_path_bytes);
            if let Some(end) = prefix_end(json_path_bytes) {
                term_stream_builder = term_stream_builder.lt(&end);
            }
        }
        let mut term_stream = term_stream_builder.into_stream()?;
        while let Some((_, term_info, state)) = term_stream.next() {
            let term_distance_plus_one = self.automaton.0.distance(*state).to_u8() + 1;
            let mut block_segment_postings = inverted_index
                .read_block_postings_from_terminfo(term_info, IndexRecordOption::Basic)?;
            loop {
                let docs = block_segment_postings.docs();
                if docs.is_empty() {
                    break;
                }
                for &doc in docs {
                    matching_docs.insert(doc);
                    let doc_distance_plus_one = &mut distance_plus_one[doc as usize];
                    if *doc_distance_plus_one == 0
                        || term_distance_plus_one < *doc_distance_plus_one
                    {
                        *doc_distance_plus_one = term_distance_plus_one;
                    }
                }
                block_segment_postings.advance();
            }
        }
        Ok(Box::new(FuzzyTermScorer {
            matching_docs: BitSetDocSet::from(matching_docs),
            distance_plus_one,
            boost,
        }))
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let mut scorer = self.scorer(reader, 1.0)?;
        if scorer.seek(doc) == doc {
            Ok(Explanation::new("FuzzyTermScorer", scorer.score()))
        } else {
            Err(TantivyError::InvalidArgument(
                "Document does not exist".to_string(),
            ))
        }
    }
}

struct FuzzyTermScorer {
    matching_docs: BitSetDocSet,
    /// Per document, one more than the smallest distance among its matching terms, and 0 for a
    /// document without a matching term. A zeroed vector allocates lazily, so only the pages of
    /// matching documents are touched.
    distance_plus_one: Vec<u8>,
    boost: Score,
}

impl DocSet for FuzzyTermScorer {
    fn advance(&mut self) -> DocId {
        self.matching_docs.advance()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        self.matching_docs.seek(target)
    }

    fn doc(&self) -> DocId {
        self.matching_docs.doc()
    }

    fn size_hint(&self) -> u32 {
        self.matching_docs.size_hint()
    }
}

impl Scorer for FuzzyTermScorer {
    fn score(&mut self) -> Score {
        let doc = self.matching_docs.doc();
        self.boost / self.distance_plus_one[doc as usize] as Score
    }
}

#[cfg(test)]
mod test {
    use super::FuzzyTermQuery;
    use crate::collector::{Count, TopDocs};
    use crate::indexer::NoMergePolicy;
    use crate::query::QueryParser;
    use crate::schema::{Schema, STORED, TEXT};
    use crate::{assert_nearly_equals, Index, IndexWriter, TantivyDocument, Term};

    #[test]
    pub fn test_fuzzy_automaton_streams_expanded_terms() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let title = schema_builder.add_text_field("title", TEXT);
        let index = Index::create_in_ram(schema_builder.build());
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(doc!(title => "The Diary of a Dairy Cow"))?;
        index_writer.add_document(doc!(title => "A Daily Log"))?;
        index_writer.commit()?;
        let searcher = index.reader()?.searcher();

        let query = FuzzyTermQuery::new(Term::from_field_text(title, "diary"), 1, true);
        let automaton = query.automaton()?;
        let inverted_index = searcher.segment_reader(0).inverted_index(title)?;
        let mut stream = inverted_index.terms().search(automaton).into_stream()?;
        let mut terms = Vec::new();
        while stream.advance() {
            terms.push(String::from_utf8(stream.key().to_vec()).unwrap());
        }
        assert_eq!(terms, vec!["dairy", "diary"]);
        assert_eq!(searcher.search(&query, &Count)?, 1);
        Ok(())
    }

    #[test]
    pub fn test_fuzzy_json_path() -> crate::Result<()> {
        // # Defining the schema
        let mut schema_builder = Schema::builder();
        let attributes = schema_builder.add_json_field("attributes", TEXT | STORED);
        let schema = schema_builder.build();

        // # Indexing documents
        let index = Index::create_in_ram(schema.clone());

        let mut index_writer = index.writer_for_tests()?;
        index_writer.set_merge_policy(Box::new(NoMergePolicy));
        let doc = TantivyDocument::parse_json(
            &schema,
            r#"{
            "attributes": {
                "a": "japan"
            }
        }"#,
        )?;
        index_writer.add_document(doc)?;
        let doc = TantivyDocument::parse_json(
            &schema,
            r#"{
            "attributes": {
                "aa": "japan"
            }
        }"#,
        )?;
        index_writer.add_document(doc)?;
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        // # Fuzzy search
        let query_parser = QueryParser::for_index(&index, vec![attributes]);

        let get_json_path_term = |query: &str| -> crate::Result<Term> {
            let query = query_parser.parse_query(query)?;
            let mut terms = Vec::new();
            query.query_terms(&mut |term, _| {
                terms.push(term.clone());
            });

            Ok(terms[0].clone())
        };

        // shall not match the first document due to json path mismatch
        {
            let term = get_json_path_term("attributes.aa:japan")?;
            let fuzzy_query = FuzzyTermQuery::new(term, 2, true);
            let top_docs =
                searcher.search(&fuzzy_query, &TopDocs::with_limit(2).order_by_score())?;
            assert_eq!(top_docs.len(), 1, "Expected only 1 document");
            assert_eq!(top_docs[0].1.doc_id, 1, "Expected the second document");
        }

        // shall match the first document because Levenshtein distance is 1 (substitute 'o' with
        // 'a')
        {
            let term = get_json_path_term("attributes.a:japon")?;

            let fuzzy_query = FuzzyTermQuery::new(term, 1, true);
            let top_docs =
                searcher.search(&fuzzy_query, &TopDocs::with_limit(2).order_by_score())?;
            assert_eq!(top_docs.len(), 1, "Expected only 1 document");
            assert_eq!(top_docs[0].1.doc_id, 0, "Expected the first document");
        }

        // shall not match because non-prefix Levenshtein distance is more than 1 (add 'a' and 'n')
        {
            let term = get_json_path_term("attributes.a:jap")?;

            let fuzzy_query = FuzzyTermQuery::new(term, 1, true);
            let top_docs =
                searcher.search(&fuzzy_query, &TopDocs::with_limit(2).order_by_score())?;
            assert_eq!(top_docs.len(), 0, "Expected no document");
        }

        Ok(())
    }

    #[test]
    pub fn test_fuzzy_term() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let country_field = schema_builder.add_text_field("country", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            index_writer.add_document(doc!(
                country_field => "japan",
            ))?;
            index_writer.add_document(doc!(
                country_field => "korea",
            ))?;
            index_writer.commit()?;
        }
        let reader = index.reader()?;
        let searcher = reader.searcher();

        // passes because Levenshtein distance is 1 (substitute 'o' with 'a')
        // Score should be 1.0 / (1.0 + 1) = 0.5
        {
            let term = Term::from_field_text(country_field, "japon");
            let fuzzy_query = FuzzyTermQuery::new(term, 1, true);
            let top_docs =
                searcher.search(&fuzzy_query, &TopDocs::with_limit(2).order_by_score())?;
            assert_eq!(top_docs.len(), 1, "Expected only 1 document");
            let (score, _) = top_docs[0];
            assert_nearly_equals!(0.5, score);
        }

        // fails because non-prefix Levenshtein distance is more than 1 (add 'a' and 'n')
        {
            let term = Term::from_field_text(country_field, "jap");

            let fuzzy_query = FuzzyTermQuery::new(term, 1, true);
            let top_docs =
                searcher.search(&fuzzy_query, &TopDocs::with_limit(2).order_by_score())?;
            assert_eq!(top_docs.len(), 0, "Expected no document");
        }

        // passes because prefix Levenshtein distance is 0
        {
            let term = Term::from_field_text(country_field, "jap");
            let fuzzy_query = FuzzyTermQuery::new_prefix(term, 1, true);
            let top_docs =
                searcher.search(&fuzzy_query, &TopDocs::with_limit(2).order_by_score())?;
            assert_eq!(top_docs.len(), 1, "Expected only 1 document");
            let (score, _) = top_docs[0];
            assert_nearly_equals!(1.0, score);
        }
        Ok(())
    }

    #[test]
    pub fn test_fuzzy_term_transposition_cost_one() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let country_field = schema_builder.add_text_field("country", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(doc!(country_field => "japan"))?;
        index_writer.commit()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let term_jaapn = Term::from_field_text(country_field, "jaapn");
        {
            let fuzzy_query_transposition = FuzzyTermQuery::new(term_jaapn.clone(), 1, true);
            let count = searcher.search(&fuzzy_query_transposition, &Count)?;
            assert_eq!(count, 1);
        }
        {
            let fuzzy_query_transposition = FuzzyTermQuery::new(term_jaapn, 1, false);
            let count = searcher.search(&fuzzy_query_transposition, &Count)?;
            assert_eq!(count, 0);
        }
        Ok(())
    }

    #[test]
    pub fn test_fuzzy_scoring_ranks_exact_matches_higher() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let word_field = schema_builder.add_text_field("word", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            // Add exact match (doc_id=0)
            index_writer.add_document(doc!(word_field => "hello"))?;
            // Add distance-1 match (doc_id=1): e→a substitution
            index_writer.add_document(doc!(word_field => "hallo"))?;
            // Add distance-2 match (doc_id=2): e→i and l→o (two substitutions)
            index_writer.add_document(doc!(word_field => "hiloo"))?;
            index_writer.commit()?;
        }
        let reader = index.reader()?;
        let searcher = reader.searcher();

        // Search for "hello" with distance 2
        let term = Term::from_field_text(word_field, "hello");
        let fuzzy_query = FuzzyTermQuery::new(term, 2, true);
        let top_docs = searcher.search(&fuzzy_query, &TopDocs::with_limit(10).order_by_score())?;

        assert_eq!(top_docs.len(), 3);

        // Verify scores:
        // - "hello" (exact, distance 0): 1.0 / (1 + 0) = 1.0
        // - "hallo" (distance 1): 1.0 / (1 + 1) = 0.5
        // - "hiloo" (distance 2): 1.0 / (1 + 2) = 0.333...

        // First result should be exact match with score 1.0
        let (score_0, doc_addr_0) = top_docs[0];
        assert_nearly_equals!(1.0, score_0);
        assert_eq!(doc_addr_0.doc_id, 0);

        // Second should be distance-1 match with score 0.5
        let (score_1, doc_addr_1) = top_docs[1];
        assert_nearly_equals!(0.5, score_1);
        assert_eq!(doc_addr_1.doc_id, 1);

        // Third should be distance-2 match with score ~0.333
        let (score_2, doc_addr_2) = top_docs[2];
        assert_nearly_equals!(1.0 / 3.0, score_2);
        assert_eq!(doc_addr_2.doc_id, 2);

        Ok(())
    }

    #[test]
    pub fn test_fuzzy_prefix_scoring_ranks_exact_matches_higher() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let word_field = schema_builder.add_text_field("word", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            // Add exact match (doc_id=0)
            index_writer.add_document(doc!(word_field => "hello"))?;
            // Add distance-1 match (doc_id=1): e→a substitution
            index_writer.add_document(doc!(word_field => "hallo"))?;
            // Add distance-2 match (doc_id=2): e→i and l→o (two substitutions)
            index_writer.add_document(doc!(word_field => "hiooo"))?;
            index_writer.commit()?;
        }
        let reader = index.reader()?;
        let searcher = reader.searcher();

        // Search for "hel" with distance 1, with prefix
        let term = Term::from_field_text(word_field, "hel");
        let fuzzy_query = FuzzyTermQuery::new_prefix(term, 1, true);
        let top_docs = searcher.search(&fuzzy_query, &TopDocs::with_limit(10).order_by_score())?;

        assert_eq!(top_docs.len(), 2);

        // Verify scores:
        // - "hel...lo" (exact, distance 0): 1.0 / (1 + 0) = 1.0
        // - "hal...lo" (distance 1): 1.0 / (1 + 1) = 0.5
        // - "hio...oo" (distance 2): no-match

        // First result should be exact match with score 1.0
        let (score_0, doc_addr_0) = top_docs[0];
        assert_nearly_equals!(1.0, score_0);
        assert_eq!(doc_addr_0.doc_id, 0);

        // Second should be distance-1 match with score 0.5
        let (score_1, doc_addr_1) = top_docs[1];
        assert_nearly_equals!(0.5, score_1);
        assert_eq!(doc_addr_1.doc_id, 1);

        Ok(())
    }

    #[test]
    pub fn test_fuzzy_scoring_uses_closest_term_of_document() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let word_field = schema_builder.add_text_field("word", TEXT);
        let index = Index::create_in_ram(schema_builder.build());
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(doc!(word_field => "hallo hello"))?;
        index_writer.add_document(doc!(word_field => "hallo"))?;
        index_writer.commit()?;
        let searcher = index.reader()?.searcher();

        let fuzzy_query = FuzzyTermQuery::new(Term::from_field_text(word_field, "hello"), 1, true);
        let top_docs = searcher.search(&fuzzy_query, &TopDocs::with_limit(10).order_by_score())?;

        assert_eq!(top_docs.len(), 2);
        assert_nearly_equals!(1.0, top_docs[0].0);
        assert_eq!(top_docs[0].1.doc_id, 0);
        assert_nearly_equals!(0.5, top_docs[1].0);
        assert_eq!(top_docs[1].1.doc_id, 1);
        assert_eq!(searcher.search(&fuzzy_query, &Count)?, 2);
        Ok(())
    }
}
