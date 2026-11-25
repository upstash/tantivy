use std::collections::BTreeSet;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use common::BitSet;
use levenshtein_automata::{Distance, LevenshteinAutomatonBuilder, DFA};
use once_cell::sync::OnceCell;
use tantivy_fst::Automaton;

use crate::index::SegmentReader;
use crate::postings::SegmentPostings;
use crate::query::phrase_prefix_query::prefix_end;
use crate::query::score_combiner::{DoNothingCombiner, ScoreCombiner};
use crate::query::union::BufferedUnionScorer;
use crate::query::{
    BitSetDocSet, ConstScorer, EmptyScorer, EnableScoring, Explanation, Query, Scorer, Weight,
};
use crate::schema::{Field, IndexRecordOption, Term, Type};
use crate::TantivyError::InvalidArgument;
use crate::{DocId, DocSet, Score, TantivyError};

/// The number of terms a [`FuzzyTermQuery`] expands to: the closest matching terms over all
/// segments, ties broken by the smaller term bytes.
pub const MAX_EXPANSIONS: usize = 50;

/// Memory shared by the fuzzy term queries of one query: the automata they build and the terms
/// they expand to. An automaton over the budget fails its build. Expanded terms over the budget
/// are dropped, farthest first.
#[derive(Debug)]
pub struct FuzzyBudget {
    max_bytes: usize,
    used: AtomicUsize,
}

impl FuzzyBudget {
    pub fn new(max_bytes: usize) -> FuzzyBudget {
        FuzzyBudget {
            max_bytes,
            used: AtomicUsize::new(0),
        }
    }

    fn remaining(&self) -> usize {
        self.max_bytes
            .saturating_sub(self.used.load(Ordering::Relaxed))
    }
}

/// The Levenshtein automaton a [`FuzzyTermQuery`] matches terms with.
///
/// Obtained from [`FuzzyTermQuery::automaton`], e.g. to stream the term dictionary
/// and collect the terms within the query's distance.
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
#[derive(Clone)]
pub struct FuzzyTermQuery {
    /// What term are we searching
    term: Term,
    /// How many changes are we going to allow
    distance: u8,
    /// Should a transposition cost 1 or 2?
    transposition_cost_one: bool,
    /// is a starts with query
    prefix: bool,
    /// Set by `build_with_limit`, which builds the automata up front and charges them here:
    /// one per distance from 0 to `distance`, indexed by distance.
    budget: Option<(Arc<FuzzyBudget>, Arc<Vec<DfaWrapper>>)>,
}

impl fmt::Debug for FuzzyTermQuery {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("FuzzyTermQuery")
            .field("term", &self.term)
            .field("distance", &self.distance)
            .field("transposition_cost_one", &self.transposition_cost_one)
            .field("prefix", &self.prefix)
            .finish()
    }
}

impl FuzzyTermQuery {
    /// Creates a new Fuzzy Query
    pub fn new(term: Term, distance: u8, transposition_cost_one: bool) -> FuzzyTermQuery {
        FuzzyTermQuery {
            term,
            distance,
            transposition_cost_one,
            prefix: false,
            budget: None,
        }
    }

    /// Creates a new Fuzzy Query of the Term prefix
    pub fn new_prefix(term: Term, distance: u8, transposition_cost_one: bool) -> FuzzyTermQuery {
        FuzzyTermQuery {
            term,
            distance,
            transposition_cost_one,
            prefix: true,
            budget: None,
        }
    }

    /// Creates a fuzzy query (a prefix one when `prefix` is set) and builds its automata now,
    /// charged to `budget`. The terms it expands to are charged to `budget` while its weight
    /// lives. Returns `None` when the automata do not fit in what is left of `budget`.
    pub fn build_with_limit(
        term: Term,
        distance: u8,
        transposition_cost_one: bool,
        prefix: bool,
        budget: Arc<FuzzyBudget>,
    ) -> crate::Result<Option<FuzzyTermQuery>> {
        let mut query = FuzzyTermQuery {
            term,
            distance,
            transposition_cost_one,
            prefix,
            budget: None,
        };
        let mut automata = Vec::with_capacity(distance as usize + 1);
        let mut bytes = 0;
        for level in 0..=distance {
            let max_bytes = budget.remaining().saturating_sub(bytes);
            let Some(automaton) = query.build_automaton(level, max_bytes)? else {
                return Ok(None);
            };
            bytes += automaton.0.memory_usage();
            automata.push(automaton);
        }
        budget.used.fetch_add(bytes, Ordering::Relaxed);
        query.budget = Some((budget, Arc::new(automata)));
        Ok(Some(query))
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
        let automaton = self.build_automaton(self.distance, usize::MAX)?;
        Ok(automaton.expect("an unlimited automaton build cannot fail"))
    }

    fn build_automaton(&self, distance: u8, max_bytes: usize) -> crate::Result<Option<DfaWrapper>> {
        static AUTOMATON_BUILDER: [[OnceCell<LevenshteinAutomatonBuilder>; 2]; 3] = [
            [OnceCell::new(), OnceCell::new()],
            [OnceCell::new(), OnceCell::new()],
            [OnceCell::new(), OnceCell::new()],
        ];

        let automaton_builder = AUTOMATON_BUILDER
            .get(distance as usize)
            .ok_or_else(|| {
                InvalidArgument(format!(
                    "Levenshtein distance of {} is not allowed. Choose a value less than {}",
                    distance,
                    AUTOMATON_BUILDER.len()
                ))
            })?
            .get(self.transposition_cost_one as usize)
            .unwrap()
            .get_or_init(|| {
                LevenshteinAutomatonBuilder::new(distance, self.transposition_cost_one)
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
        let automaton = automaton_builder.build_dfa_with_limit(term_text, self.prefix, max_bytes);
        Ok(automaton.map(DfaWrapper))
    }

    /// Returns the `MAX_EXPANSIONS` closest terms over all segments, with their distances.
    /// `automata` holds one automaton per distance from 0 to `self.distance`.
    fn expand(
        &self,
        automata: &[DfaWrapper],
        enable_scoring: &EnableScoring<'_>,
    ) -> crate::Result<BTreeSet<(u8, Vec<u8>)>> {
        let searcher = enable_scoring.searcher().ok_or_else(|| {
            InvalidArgument("A fuzzy term query needs a searcher to expand its terms".to_string())
        })?;
        let term_value = self.term.value();
        let json_path_bytes = term_value
            .as_json()
            .map(|(json_path_bytes, _)| json_path_bytes);

        // The distance-0 walk is cheap (the exact term, or a prefix's completions), and when it
        // finds `MAX_EXPANSIONS` terms no farther term can enter.
        let mut kept: BTreeSet<(u8, Vec<u8>)> = BTreeSet::new();
        let mut max_levels = vec![0];
        if self.distance > 0 {
            max_levels.push(self.distance);
        }
        for max_level in max_levels {
            for segment_reader in searcher.segment_readers() {
                let inverted_index = segment_reader.inverted_index(self.term.field())?;
                let alive_bitset = segment_reader.alive_bitset();
                let num_deleted_docs = segment_reader.num_deleted_docs();
                let mut resume: Option<Vec<u8>> = None;
                loop {
                    // Once the set is full with worst (w, t), a term before t needs a distance of
                    // at most w, and a term after t one below w. The walk uses the smallest
                    // automaton that still finds them.
                    let mut level = max_level;
                    if kept.len() == MAX_EXPANSIONS {
                        let worst = kept.last().unwrap();
                        let past_worst = resume.as_deref() >= Some(worst.1.as_slice());
                        if past_worst && worst.0 == 0 {
                            break;
                        }
                        level = if past_worst { worst.0 - 1 } else { worst.0 };
                    }
                    let automaton = &automata[level as usize];
                    let mut term_stream_builder =
                        inverted_index.terms().search_with_state(automaton);
                    if let Some(json_path_bytes) = json_path_bytes {
                        term_stream_builder = term_stream_builder.ge(json_path_bytes);
                        if let Some(end) = prefix_end(json_path_bytes) {
                            term_stream_builder = term_stream_builder.lt(&end);
                        }
                    }
                    if let Some(resume) = &resume {
                        term_stream_builder = term_stream_builder.ge(resume);
                    }
                    let mut term_stream = term_stream_builder.into_stream()?;

                    let mut switch = false;
                    while let Some((key, term_info, state)) = term_stream.next() {
                        let distance = automaton.0.distance(*state).to_u8();
                        if kept.len() < MAX_EXPANSIONS
                            || (distance, key) < {
                                let worst = kept.last().unwrap();
                                (worst.0, worst.1.as_slice())
                            }
                        {
                            // A term stays in the dictionary until a merge drops its deleted
                            // documents, so a term without a live document must not take a slot.
                            // A term with more documents than the segment has deleted has one.
                            let maybe_dead = term_info.doc_freq <= num_deleted_docs;
                            if let (Some(alive_bitset), true) = (alive_bitset, maybe_dead) {
                                let mut postings = inverted_index
                                    .read_block_postings_from_terminfo(
                                        term_info,
                                        IndexRecordOption::Basic,
                                    )?;
                                let mut live = false;
                                while !live && !postings.docs().is_empty() {
                                    for &doc in postings.docs() {
                                        if alive_bitset.is_alive(doc) {
                                            live = true;
                                            break;
                                        }
                                    }
                                    postings.advance();
                                }
                                if !live {
                                    continue;
                                }
                            }
                            kept.insert((distance, key.to_vec()));
                            if kept.len() > MAX_EXPANSIONS {
                                kept.pop_last();
                            }
                        }
                        if kept.len() == MAX_EXPANSIONS {
                            let worst = kept.last().unwrap();
                            if worst.0 < level || (worst.0 == level && key >= worst.1.as_slice()) {
                                // The smallest key after this one.
                                let mut next = key.to_vec();
                                next.push(0);
                                resume = Some(next);
                                switch = true;
                                break;
                            }
                        }
                    }
                    if !switch {
                        break;
                    }
                }
            }
            if kept.len() == MAX_EXPANSIONS {
                break;
            }
        }
        Ok(kept)
    }
}

impl Query for FuzzyTermQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        let kept = match &self.budget {
            Some((_, automata)) => self.expand(automata, &enable_scoring)?,
            None => {
                let mut automata = Vec::with_capacity(self.distance as usize + 1);
                for level in 0..=self.distance {
                    automata.push(self.build_automaton(level, usize::MAX)?.unwrap());
                }
                self.expand(&automata, &enable_scoring)?
            }
        };

        let remaining = match &self.budget {
            Some((budget, _)) => budget.remaining(),
            None => usize::MAX,
        };
        let mut charged = 0;
        let mut terms = Vec::with_capacity(kept.len());
        for (distance, key) in kept {
            let bytes = key.len() + std::mem::size_of::<(Vec<u8>, u8)>();
            if charged + bytes > remaining {
                break;
            }
            charged += bytes;
            terms.push((key, distance));
        }
        if let Some((budget, _)) = &self.budget {
            budget.used.fetch_add(charged, Ordering::Relaxed);
        }
        Ok(Box::new(FuzzyTermWeight {
            field: self.term.field(),
            terms,
            scoring_enabled: enable_scoring.is_scoring_enabled(),
            budget: self
                .budget
                .as_ref()
                .map(|(budget, _)| (budget.clone(), charged)),
        }))
    }
}

/// The weight of a [`FuzzyTermQuery`]: its expanded terms, sorted by distance. A document scores
/// `boost / (1 + distance)`, where `distance` is the smallest distance among its matching terms.
struct FuzzyTermWeight {
    field: Field,
    terms: Vec<(Vec<u8>, u8)>,
    scoring_enabled: bool,
    /// The budget the expanded terms are charged to, and how much they took.
    budget: Option<(Arc<FuzzyBudget>, usize)>,
}

impl Drop for FuzzyTermWeight {
    fn drop(&mut self) {
        if let Some((budget, charged)) = &self.budget {
            budget.used.fetch_sub(*charged, Ordering::Relaxed);
        }
    }
}

/// Keeps the smallest `distance + 1` among the matching terms of a document.
#[derive(Clone, Copy)]
struct MinCombiner {
    min: Score,
}

impl Default for MinCombiner {
    fn default() -> MinCombiner {
        MinCombiner { min: Score::MAX }
    }
}

impl ScoreCombiner for MinCombiner {
    fn update<TScorer: Scorer>(&mut self, scorer: &mut TScorer) {
        self.min = self.min.min(scorer.score());
    }

    fn clear(&mut self) {
        self.min = Score::MAX;
    }

    fn score(&self) -> Score {
        self.min
    }
}

impl Weight for FuzzyTermWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let inverted_index = reader.inverted_index(self.field)?;
        let term_dict = inverted_index.terms();
        let mut found = Vec::with_capacity(self.terms.len());
        let mut postings_bytes = 0;
        let mut doc_freq = 0;
        let mut distinct_distances = 0;
        for (key, distance) in &self.terms {
            let Some(term_info) = term_dict.get(key)? else {
                continue;
            };
            postings_bytes += term_info.postings_range.len();
            doc_freq += term_info.doc_freq as usize;
            if found.last().map(|(_, last)| last) != Some(distance) {
                distinct_distances += 1;
            }
            found.push((term_info, *distance));
        }
        if found.is_empty() {
            return Ok(Box::new(EmptyScorer));
        }

        // An open posting cursor holds its whole list, so the scorer opens all of them at once
        // only when they take less memory than the bitsets. Dense lists go to the bitsets even
        // when they are smaller, because filling a bitset is faster than walking a union.
        let max_doc = reader.max_doc();
        let num_bitsets = if self.scoring_enabled {
            distinct_distances
        } else {
            1
        };
        let bitsets_bytes = num_bitsets * (max_doc as usize / 8 + 1);
        if postings_bytes < bitsets_bytes && doc_freq < max_doc as usize / 16 {
            let mut scorers = Vec::with_capacity(found.len());
            for (term_info, distance) in &found {
                let postings = inverted_index
                    .read_postings_from_terminfo(term_info, IndexRecordOption::Basic)?;
                scorers.push(ConstScorer::new(postings, (*distance + 1) as Score));
            }
            if !self.scoring_enabled {
                let union =
                    BufferedUnionScorer::build(scorers, DoNothingCombiner::default, max_doc);
                return Ok(Box::new(union));
            }
            let union = BufferedUnionScorer::build(scorers, MinCombiner::default, max_doc);
            return Ok(Box::new(FuzzyUnionScorer { union, boost }));
        }

        // `found` is sorted by distance, so the closer bitsets come out closest first. Without
        // scoring, all documents go to `matching_docs`.
        let mut matching_docs = BitSet::with_max_value(max_doc);
        let mut closer: Vec<(Vec<u64>, u8)> = Vec::new();
        let farthest = found.last().unwrap().1;
        for (term_info, distance) in &found {
            let is_closer = self.scoring_enabled && *distance != farthest;
            if is_closer && closer.last().map(|(_, last)| last) != Some(distance) {
                closer.push((vec![0u64; max_doc as usize / 64 + 1], *distance));
            }
            let mut block_segment_postings = inverted_index
                .read_block_postings_from_terminfo(term_info, IndexRecordOption::Basic)?;
            loop {
                let docs = block_segment_postings.docs();
                if docs.is_empty() {
                    break;
                }
                if is_closer {
                    let closer_docs = &mut closer.last_mut().unwrap().0;
                    for &doc in docs {
                        matching_docs.insert(doc);
                        closer_docs[doc as usize / 64] |= 1u64 << (doc % 64);
                    }
                } else {
                    for &doc in docs {
                        matching_docs.insert(doc);
                    }
                }
                block_segment_postings.advance();
            }
        }
        Ok(Box::new(FuzzyBitSetScorer {
            matching_docs: BitSetDocSet::from(matching_docs),
            closer,
            farthest,
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

struct FuzzyUnionScorer {
    union: BufferedUnionScorer<ConstScorer<SegmentPostings>, MinCombiner>,
    boost: Score,
}

impl DocSet for FuzzyUnionScorer {
    fn advance(&mut self) -> DocId {
        self.union.advance()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        self.union.seek(target)
    }

    fn doc(&self) -> DocId {
        self.union.doc()
    }

    fn size_hint(&self) -> u32 {
        self.union.size_hint()
    }

    fn cost(&self) -> u64 {
        self.union.cost()
    }
}

impl Scorer for FuzzyUnionScorer {
    fn score(&mut self) -> Score {
        self.boost / self.union.score()
    }
}

struct FuzzyBitSetScorer {
    matching_docs: BitSetDocSet,
    /// The documents of each distance below `farthest`, closest first, one bit per document.
    closer: Vec<(Vec<u64>, u8)>,
    farthest: u8,
    boost: Score,
}

impl DocSet for FuzzyBitSetScorer {
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

impl Scorer for FuzzyBitSetScorer {
    fn score(&mut self) -> Score {
        let doc = self.matching_docs.doc();
        for (docs, distance) in &self.closer {
            if docs[doc as usize / 64] & (1u64 << (doc % 64)) != 0 {
                return self.boost / (*distance + 1) as Score;
            }
        }
        self.boost / (self.farthest + 1) as Score
    }
}

#[cfg(test)]
mod test {
    use std::collections::BTreeMap;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    use super::{FuzzyBudget, FuzzyTermQuery};
    use crate::collector::{Count, TopDocs};
    use crate::indexer::NoMergePolicy;
    use crate::query::{BooleanQuery, BoostQuery, Occur, Query, QueryParser, TermQuery};
    use crate::schema::{Field, IndexRecordOption, Schema, Value, STORED, STRING, TEXT};
    use crate::{assert_nearly_equals, Index, IndexWriter, TantivyDocument, Term};

    /// One document per term, one segment per inner list.
    fn index_with_segments(segments: &[Vec<String>]) -> crate::Result<(Index, Field)> {
        let mut schema_builder = Schema::builder();
        let t = schema_builder.add_text_field("t", STRING | STORED);
        let index = Index::create_in_ram(schema_builder.build());
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.set_merge_policy(Box::new(NoMergePolicy));
        for segment in segments {
            for term in segment {
                index_writer.add_document(doc!(t => term.as_str()))?;
            }
            index_writer.commit()?;
        }
        Ok((index, t))
    }

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

    #[test]
    pub fn test_fuzzy_keeps_the_closest_terms_over_all_segments() -> crate::Result<()> {
        let mut distance_one = vec!["ammmm".to_string(), "mmm".to_string()];
        for position in 0..4 {
            for letter in 'a'..='l' {
                let mut term: Vec<char> = "mmmm".chars().collect();
                term[position] = letter;
                distance_one.push(term.into_iter().collect());
            }
        }
        assert_eq!(distance_one.len(), 50);
        let exact = vec!["mmmm".to_string()];

        let mut expected: Vec<String> = distance_one.clone();
        expected.retain(|term| term != "mmml");
        expected.push("mmmm".to_string());
        expected.sort();

        for (segments, count) in [
            (vec![distance_one.clone(), exact.clone()], 50),
            (vec![exact.clone(), distance_one.clone()], 50),
            (vec![[distance_one.clone(), exact.clone()].concat()], 50),
            (
                vec![distance_one.clone(), distance_one.clone(), exact.clone()],
                99,
            ),
        ] {
            let (index, t) = index_with_segments(&segments)?;
            let query = FuzzyTermQuery::new(Term::from_field_text(t, "mmmm"), 1, true);
            let searcher = index.reader()?.searcher();
            let mut matched = BTreeMap::new();
            for (score, address) in
                searcher.search(&query, &TopDocs::with_limit(1000).order_by_score())?
            {
                let doc: TantivyDocument = searcher.doc(address)?;
                matched.insert(
                    doc.get_first(t).unwrap().as_str().unwrap().to_string(),
                    score,
                );
            }
            assert_eq!(matched.into_keys().collect::<Vec<String>>(), expected);
            assert_eq!(searcher.search(&query, &Count)?, count);
        }
        Ok(())
    }

    #[test]
    pub fn test_fuzzy_prefix_keeps_the_smallest_terms() -> crate::Result<()> {
        let mut terms = Vec::new();
        for i in 0..60 {
            terms.push(format!("la{i:02}"));
        }
        let (index, t) = index_with_segments(&[terms[30..].to_vec(), terms[..30].to_vec()])?;
        let query = FuzzyTermQuery::new_prefix(Term::from_field_text(t, "la"), 0, true);
        let searcher = index.reader()?.searcher();
        let mut matched = BTreeMap::new();
        for (score, address) in
            searcher.search(&query, &TopDocs::with_limit(1000).order_by_score())?
        {
            let doc: TantivyDocument = searcher.doc(address)?;
            matched.insert(
                doc.get_first(t).unwrap().as_str().unwrap().to_string(),
                score,
            );
        }
        assert_eq!(
            matched.into_keys().collect::<Vec<String>>(),
            terms[..50].to_vec()
        );
        assert_eq!(searcher.search(&query, &Count)?, 50);
        Ok(())
    }

    #[test]
    pub fn test_fuzzy_skips_terms_without_live_documents() -> crate::Result<()> {
        let mut terms = Vec::new();
        for i in 0..51 {
            terms.push(format!("sku{i:02}"));
        }
        let (index, t) = index_with_segments(&[terms.clone()])?;
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.set_merge_policy(Box::new(NoMergePolicy));
        for term in &terms[..50] {
            index_writer.delete_term(Term::from_field_text(t, term));
        }
        index_writer.commit()?;
        let query = FuzzyTermQuery::new_prefix(Term::from_field_text(t, "sku"), 0, true);
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        assert_eq!(searcher.search(&query, &Count)?, 1);
        Ok(())
    }

    #[test]
    pub fn test_fuzzy_scores_in_both_scoring_modes() -> crate::Result<()> {
        let terms = vec![
            "laptop".to_string(),
            "laptap".to_string(),
            "laptip".to_string(),
            "other".to_string(),
        ];
        let mut many = terms.clone();
        for i in 0..20_000 {
            many.push(format!("filler{i}"));
        }
        for segment in [terms, many] {
            let (index, t) = index_with_segments(&[segment])?;
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            index_writer.delete_term(Term::from_field_text(t, "laptip"));
            index_writer.commit()?;
            let query = FuzzyTermQuery::new(Term::from_field_text(t, "laptop"), 1, true);

            let searcher = index.reader()?.searcher();
            let mut matched = BTreeMap::new();
            for (score, address) in
                searcher.search(&query, &TopDocs::with_limit(1000).order_by_score())?
            {
                let doc: TantivyDocument = searcher.doc(address)?;
                matched.insert(
                    doc.get_first(t).unwrap().as_str().unwrap().to_string(),
                    score,
                );
            }
            assert_eq!(matched.len(), 2);
            assert_nearly_equals!(matched["laptop"], 1.0);
            assert_nearly_equals!(matched["laptap"], 0.5);
            assert_eq!(searcher.search(&query, &Count)?, 2);

            let negative = BoostQuery::new(Box::new(query.clone()), -2.0);
            let searcher = index.reader()?.searcher();
            let mut negative_matched = BTreeMap::new();
            for (score, address) in
                searcher.search(&negative, &TopDocs::with_limit(1000).order_by_score())?
            {
                let doc: TantivyDocument = searcher.doc(address)?;
                negative_matched.insert(
                    doc.get_first(t).unwrap().as_str().unwrap().to_string(),
                    score,
                );
            }
            assert_nearly_equals!(negative_matched["laptop"], -2.0);
            assert_nearly_equals!(negative_matched["laptap"], -1.0);

            let excluding = BooleanQuery::new(vec![
                (Occur::Must, Box::new(query.clone()) as Box<dyn Query>),
                (
                    Occur::MustNot,
                    Box::new(TermQuery::new(
                        Term::from_field_text(t, "laptap"),
                        IndexRecordOption::Basic,
                    )),
                ),
            ]);
            let searcher = index.reader()?.searcher();
            let mut excluding_matched = BTreeMap::new();
            for (score, address) in
                searcher.search(&excluding, &TopDocs::with_limit(1000).order_by_score())?
            {
                let doc: TantivyDocument = searcher.doc(address)?;
                excluding_matched.insert(
                    doc.get_first(t).unwrap().as_str().unwrap().to_string(),
                    score,
                );
            }
            assert_eq!(excluding_matched.len(), 1);
            assert_nearly_equals!(excluding_matched["laptop"], 1.0);
            assert_eq!(searcher.search(&excluding, &Count)?, 1);
        }
        Ok(())
    }

    #[test]
    pub fn test_fuzzy_budget() -> crate::Result<()> {
        let (index, t) = index_with_segments(&[vec!["laptop".to_string(), "laptap".to_string()]])?;
        let searcher = index.reader()?.searcher();
        let term = Term::from_field_text(t, "laptop");

        let budget = Arc::new(FuzzyBudget::new(10));
        let query = FuzzyTermQuery::build_with_limit(term.clone(), 1, true, false, budget)?;
        assert!(query.is_none());

        let budget = Arc::new(FuzzyBudget::new(1 << 20));
        let query = FuzzyTermQuery::build_with_limit(term.clone(), 1, true, false, budget.clone())?
            .unwrap();
        let automaton_bytes = budget.used.load(Ordering::Relaxed);
        assert!(automaton_bytes > 0);
        assert_eq!(searcher.search(&query, &Count)?, 2);
        assert_eq!(budget.used.load(Ordering::Relaxed), automaton_bytes);

        let budget = Arc::new(FuzzyBudget::new(automaton_bytes + 40));
        let query =
            FuzzyTermQuery::build_with_limit(term.clone(), 1, true, false, budget)?.unwrap();
        assert_eq!(searcher.search(&query, &Count)?, 1);

        let budget = Arc::new(FuzzyBudget::new(2 * automaton_bytes + 40));
        let first = FuzzyTermQuery::build_with_limit(term.clone(), 1, true, false, budget.clone())?
            .unwrap();
        let second =
            FuzzyTermQuery::build_with_limit(term, 1, true, false, budget.clone())?.unwrap();
        let both = BooleanQuery::new(vec![
            (Occur::Should, Box::new(first) as Box<dyn Query>),
            (Occur::Should, Box::new(second)),
        ]);
        assert_eq!(searcher.search(&both, &Count)?, 1);
        assert_eq!(budget.used.load(Ordering::Relaxed), 2 * automaton_bytes);
        Ok(())
    }
}
