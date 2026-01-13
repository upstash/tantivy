use std::any::TypeId;
use std::io;
use std::sync::Arc;

use common::BitSet;
use tantivy_fst::Automaton;

use super::fuzzy_query::DfaWrapper;
use super::phrase_prefix_query::prefix_end;
use crate::docset::COLLECT_BLOCK_BUFFER_LEN;
use crate::index::SegmentReader;
use crate::postings::TermInfo;
use crate::query::{BitSetDocSet, Explanation, Scorer, Weight};
use crate::schema::{Field, IndexRecordOption};
use crate::termdict::{TermDictionary, TermStreamer, TermStreamerWithState};
use crate::{DocId, DocSet, Score, TantivyError};

/// A weight struct for Fuzzy Term and Regex Queries
pub struct AutomatonWeight<A> {
    field: Field,
    automaton: Arc<A>,
    // For JSON fields, the term dictionary include terms from all paths.
    // We apply additional filtering based on the given JSON path, when searching within the term
    // dictionary. This prevents terms from unrelated paths from matching the search criteria.
    json_path_bytes: Option<Box<[u8]>>,
}

impl<A> AutomatonWeight<A>
where
    A: Automaton + Send + Sync + 'static,
    A::State: Clone,
{
    /// Create a new AutomationWeight
    pub fn new<IntoArcA: Into<Arc<A>>>(field: Field, automaton: IntoArcA) -> AutomatonWeight<A> {
        AutomatonWeight {
            field,
            automaton: automaton.into(),
            json_path_bytes: None,
        }
    }

    /// Create a new AutomationWeight for a json path
    pub fn new_for_json_path<IntoArcA: Into<Arc<A>>>(
        field: Field,
        automaton: IntoArcA,
        json_path_bytes: &[u8],
    ) -> AutomatonWeight<A> {
        AutomatonWeight {
            field,
            automaton: automaton.into(),
            json_path_bytes: Some(json_path_bytes.to_vec().into_boxed_slice()),
        }
    }

    fn automaton_stream<'a>(
        &'a self,
        term_dict: &'a TermDictionary,
    ) -> io::Result<TermStreamer<'a, &'a A>> {
        let automaton: &A = &self.automaton;
        let mut term_stream_builder = term_dict.search(automaton);

        if let Some(json_path_bytes) = &self.json_path_bytes {
            term_stream_builder = term_stream_builder.ge(json_path_bytes);
            if let Some(end) = prefix_end(json_path_bytes) {
                term_stream_builder = term_stream_builder.lt(&end);
            }
        }

        term_stream_builder.into_stream()
    }

    fn automaton_stream_with_state<'a>(
        &'a self,
        term_dict: &'a TermDictionary,
    ) -> io::Result<TermStreamerWithState<'a, &'a A>> {
        let automaton: &A = &self.automaton;
        let mut term_stream_builder = term_dict.search_with_state(automaton);

        if let Some(json_path_bytes) = &self.json_path_bytes {
            term_stream_builder = term_stream_builder.ge(json_path_bytes);
            if let Some(end) = prefix_end(json_path_bytes) {
                term_stream_builder = term_stream_builder.lt(&end);
            }
        }

        term_stream_builder.into_stream()
    }

    /// Returns the term infos that match the automaton
    pub fn get_match_term_infos(&self, reader: &SegmentReader) -> crate::Result<Vec<TermInfo>> {
        let inverted_index = reader.inverted_index(self.field)?;
        let term_dict = inverted_index.terms();
        let mut term_stream = self.automaton_stream(term_dict)?;
        let mut term_infos = Vec::new();
        while term_stream.advance() {
            term_infos.push(term_stream.value().clone());
        }
        Ok(term_infos)
    }

    /// Computes a score for a matched term based on the automaton state.
    /// For fuzzy queries (DfaWrapper), this returns a score inversely proportional
    /// to the Levenshtein distance. For other automatons, returns 1.0.
    fn compute_score(&self, state: &A::State) -> Score {
        // Check if this is a DfaWrapper (fuzzy query)
        if TypeId::of::<A>() == TypeId::of::<DfaWrapper>() {
            // Safe cast since we verified the type
            // SAFETY: We verified that A is DfaWrapper and A::State is u32
            let dfa = unsafe { &*(&*self.automaton as *const A as *const DfaWrapper) };
            let state_id = unsafe { *(state as *const A::State as *const u32) };
            dfa.score_from_state(state_id)
        } else {
            // Non-fuzzy automatons (regex, etc.) get constant score
            1.0
        }
    }
}

impl<A> Weight for AutomatonWeight<A>
where
    A: Automaton + Send + Sync + 'static,
    A::State: Clone,
{
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let max_doc = reader.max_doc();
        let mut doc_bitset = BitSet::with_max_value(max_doc);
        let mut doc_scores: Vec<Score> = vec![0.0; max_doc as usize];
        let inverted_index = reader.inverted_index(self.field)?;
        let term_dict = inverted_index.terms();

        // Use state-aware streaming to get distance information
        let mut term_stream = self.automaton_stream_with_state(term_dict)?;
        while term_stream.advance() {
            let term_info = term_stream.value();

            // Compute score based on automaton state (e.g., Levenshtein distance)
            let term_score = if let Some(state) = term_stream.state() {
                self.compute_score(state)
            } else {
                1.0
            };

            let mut block_segment_postings = inverted_index
                .read_block_postings_from_terminfo(term_info, IndexRecordOption::Basic)?;
            loop {
                let docs = block_segment_postings.docs();
                if docs.is_empty() {
                    break;
                }
                for &doc in docs {
                    doc_bitset.insert(doc);
                    // Keep the maximum score (closest match wins)
                    let current_score = &mut doc_scores[doc as usize];
                    if term_score > *current_score {
                        *current_score = term_score;
                    }
                }
                block_segment_postings.advance();
            }
        }
        let doc_bitset = BitSetDocSet::from(doc_bitset);
        Ok(Box::new(AutomatonScorer::new(
            doc_bitset, doc_scores, boost,
        )))
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let mut scorer = self.scorer(reader, 1.0)?;
        if scorer.seek(doc) == doc {
            let score = scorer.score();
            Ok(Explanation::new("AutomatonScorer", score))
        } else {
            Err(TantivyError::InvalidArgument(
                "Document does not exist".to_string(),
            ))
        }
    }
}

/// Scorer that returns per-document scores based on automaton match quality.
/// For fuzzy queries, this allows documents with closer matches to score higher.
pub struct AutomatonScorer {
    doc_bitset: BitSetDocSet,
    doc_scores: Vec<Score>,
    boost: Score,
}

impl AutomatonScorer {
    /// Creates a new `AutomatonScorer`.
    pub fn new(doc_bitset: BitSetDocSet, doc_scores: Vec<Score>, boost: Score) -> Self {
        AutomatonScorer {
            doc_bitset,
            doc_scores,
            boost,
        }
    }
}

impl DocSet for AutomatonScorer {
    fn advance(&mut self) -> DocId {
        self.doc_bitset.advance()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        self.doc_bitset.seek(target)
    }

    fn fill_buffer(&mut self, buffer: &mut [DocId; COLLECT_BLOCK_BUFFER_LEN]) -> usize {
        self.doc_bitset.fill_buffer(buffer)
    }

    fn doc(&self) -> DocId {
        self.doc_bitset.doc()
    }

    fn size_hint(&self) -> u32 {
        self.doc_bitset.size_hint()
    }
}

impl Scorer for AutomatonScorer {
    fn score(&mut self) -> Score {
        let doc = self.doc();
        self.doc_scores[doc as usize] * self.boost
    }
}

#[cfg(test)]
mod tests {
    use tantivy_fst::Automaton;

    use super::AutomatonWeight;
    use crate::docset::TERMINATED;
    use crate::query::Weight;
    use crate::schema::{Schema, STRING};
    use crate::{Index, IndexWriter};

    fn create_index() -> crate::Result<Index> {
        let mut schema = Schema::builder();
        let title = schema.add_text_field("title", STRING);
        let index = Index::create_in_ram(schema.build());
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(doc!(title=>"abc"))?;
        index_writer.add_document(doc!(title=>"bcd"))?;
        index_writer.add_document(doc!(title=>"abcd"))?;
        index_writer.commit()?;
        Ok(index)
    }

    #[derive(Clone, Copy)]
    enum State {
        Start,
        NotMatching,
        AfterA,
    }

    struct PrefixedByA;

    impl Automaton for PrefixedByA {
        type State = State;

        fn start(&self) -> Self::State {
            State::Start
        }

        fn is_match(&self, state: &Self::State) -> bool {
            matches!(*state, State::AfterA)
        }

        fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
            match *state {
                State::Start => {
                    if byte == b'a' {
                        State::AfterA
                    } else {
                        State::NotMatching
                    }
                }
                State::AfterA => State::AfterA,
                State::NotMatching => State::NotMatching,
            }
        }
    }

    #[test]
    fn test_automaton_weight() -> crate::Result<()> {
        let index = create_index()?;
        let field = index.schema().get_field("title").unwrap();
        let automaton_weight = AutomatonWeight::new(field, PrefixedByA);
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let mut scorer = automaton_weight.scorer(searcher.segment_reader(0u32), 1.0)?;
        assert_eq!(scorer.doc(), 0u32);
        assert_eq!(scorer.score(), 1.0);
        assert_eq!(scorer.advance(), 2u32);
        assert_eq!(scorer.doc(), 2u32);
        assert_eq!(scorer.score(), 1.0);
        assert_eq!(scorer.advance(), TERMINATED);
        Ok(())
    }

    #[test]
    fn test_automaton_weight_boost() -> crate::Result<()> {
        let index = create_index()?;
        let field = index.schema().get_field("title").unwrap();
        let automaton_weight = AutomatonWeight::new(field, PrefixedByA);
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let mut scorer = automaton_weight.scorer(searcher.segment_reader(0u32), 1.32)?;
        assert_eq!(scorer.doc(), 0u32);
        assert_eq!(scorer.score(), 1.32);
        Ok(())
    }
}
