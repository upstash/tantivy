use std::fmt;
use std::sync::Arc;

use regex_automata::dfa::{dense, Automaton as _, StartKind};
use regex_automata::nfa::thompson::{self, WhichCaptures};
use regex_automata::util::primitives::StateID;
use regex_automata::util::{start, syntax};
use regex_automata::{Anchored, MatchKind};
use tantivy_fst::Automaton;

use crate::query::{AutomatonWeight, EnableScoring, Query, Weight};
use crate::schema::Field;

/// The error returned when a [`RegexAutomaton`] cannot be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegexAutomatonError {
    /// The pattern is not a valid regex.
    Syntax(String),
    /// The pattern uses an anchor or a word boundary.
    LookAround,
    /// Building the automaton went over the size limit.
    TooComplex,
}

impl fmt::Display for RegexAutomatonError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RegexAutomatonError::Syntax(message) => f.write_str(message),
            RegexAutomatonError::LookAround => {
                f.write_str("anchors and word boundaries are not supported")
            }
            RegexAutomatonError::TooComplex => f.write_str("the pattern is too complex"),
        }
    }
}

impl std::error::Error for RegexAutomatonError {}

/// A regex compiled to a dense DFA that matches whole terms.
///
/// Building it is bounded by `size_limit`: a pattern that needs more fails with
/// [`RegexAutomatonError::TooComplex`].
pub struct RegexAutomaton {
    pattern: String,
    dfa: dense::DFA<Vec<u32>>,
    start: RegexAutomatonState,
    // Indexed by `state.as_usize() >> dfa.stride2()`. Holds the dead state, at index 0, and every
    // state reachable from `start`.
    flags: Vec<StateFlags>,
}

/// A state of a [`RegexAutomaton`].
///
/// It carries the flags of its DFA state, so the term dictionary walk gets them from the lookup
/// that computes the state.
#[derive(Clone, Copy, Debug)]
pub struct RegexAutomatonState {
    id: StateID,
    flags: StateFlags,
}

#[derive(Clone, Copy, Debug, Default)]
struct StateFlags {
    matching: bool,
    // True when some suffix leads to a match. The DFA reports a match one byte late, so the
    // state after a full match is not the dead state but is not live either.
    live: bool,
}

impl RegexAutomaton {
    /// `size_limit` applies separately to the NFA, to the determinization and to the DFA.
    pub fn new(pattern: &str, size_limit: usize) -> Result<RegexAutomaton, RegexAutomatonError> {
        let hir = match syntax::parse(pattern) {
            Ok(hir) => hir,
            Err(e) => return Err(RegexAutomatonError::Syntax(e.to_string())),
        };
        if !hir.properties().look_set().is_empty() {
            return Err(RegexAutomatonError::LookAround);
        }

        // With captures off and look-arounds rejected, every remaining build error is a size
        // limit.
        let nfa_config = thompson::Config::new()
            .which_captures(WhichCaptures::None)
            .nfa_size_limit(Some(size_limit));
        let nfa = match thompson::Compiler::new()
            .configure(nfa_config)
            .build_from_hir(&hir)
        {
            Ok(nfa) => nfa,
            Err(_) => return Err(RegexAutomatonError::TooComplex),
        };

        // `MatchKind::All` keeps every alternative, so `a|ab` matches both "a" and "ab".
        let dfa_config = dense::Config::new()
            .match_kind(MatchKind::All)
            .start_kind(StartKind::Anchored)
            .accelerate(false)
            .determinize_size_limit(Some(size_limit))
            .dfa_size_limit(Some(size_limit));
        let dfa = match dense::Builder::new()
            .configure(dfa_config)
            .build_from_nfa(&nfa)
        {
            Ok(dfa) => dfa,
            Err(_) => return Err(RegexAutomatonError::TooComplex),
        };

        let start = match dfa.start_state(&start::Config::new().anchored(Anchored::Yes)) {
            Ok(start) => start,
            Err(_) => return Err(RegexAutomatonError::TooComplex),
        };

        let mut representatives = Vec::new();
        for unit in dfa.byte_classes().representatives(..) {
            if let Some(byte) = unit.as_u8() {
                representatives.push(byte);
            }
        }

        let stride2 = dfa.stride2();
        let mut visited = vec![false; (start.as_usize() >> stride2) + 1];
        let mut predecessors: Vec<Vec<StateID>> = vec![Vec::new(); visited.len()];
        let mut flags = vec![StateFlags::default(); visited.len()];
        let mut accepting = Vec::new();
        let mut queue = vec![start];
        visited[start.as_usize() >> stride2] = true;
        let mut next = 0;
        while next < queue.len() {
            let state = queue[next];
            next += 1;

            if dfa.is_match_state(dfa.next_eoi_state(state)) {
                flags[state.as_usize() >> stride2].matching = true;
                accepting.push(state);
            }

            for &byte in &representatives {
                let target = dfa.next_state(state, byte);
                if dfa.is_dead_state(target) {
                    continue;
                }

                let index = target.as_usize() >> stride2;
                if index >= visited.len() {
                    visited.resize(index + 1, false);
                    predecessors.resize(index + 1, Vec::new());
                    flags.resize(index + 1, StateFlags::default());
                }

                predecessors[index].push(state);
                if !visited[index] {
                    visited[index] = true;
                    queue.push(target);
                }
            }
        }

        for &state in &accepting {
            flags[state.as_usize() >> stride2].live = true;
        }
        while let Some(state) = accepting.pop() {
            for &predecessor in &predecessors[state.as_usize() >> stride2] {
                let index = predecessor.as_usize() >> stride2;
                if !flags[index].live {
                    flags[index].live = true;
                    accepting.push(predecessor);
                }
            }
        }

        let start = RegexAutomatonState {
            id: start,
            flags: flags[start.as_usize() >> stride2],
        };
        Ok(RegexAutomaton {
            pattern: pattern.to_string(),
            dfa,
            start,
            flags,
        })
    }

    /// Returns the pattern the automaton was built from.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }
}

impl Automaton for RegexAutomaton {
    type State = RegexAutomatonState;

    fn start(&self) -> RegexAutomatonState {
        self.start
    }

    fn is_match(&self, state: &RegexAutomatonState) -> bool {
        state.flags.matching
    }

    fn can_match(&self, state: &RegexAutomatonState) -> bool {
        state.flags.live
    }

    fn accept(&self, state: &RegexAutomatonState, byte: u8) -> RegexAutomatonState {
        let id = self.dfa.next_state(state.id, byte);
        RegexAutomatonState {
            id,
            flags: self.flags[id.as_usize() >> self.dfa.stride2()],
        }
    }
}

impl PartialEq for RegexAutomaton {
    fn eq(&self, other: &RegexAutomaton) -> bool {
        self.pattern == other.pattern
    }
}

impl fmt::Debug for RegexAutomaton {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("RegexAutomaton")
            .field(&self.pattern)
            .finish()
    }
}

/// Matches the documents containing a term that the [`RegexAutomaton`] matches.
#[derive(Debug, Clone)]
pub struct RegexAutomatonQuery {
    automaton: Arc<RegexAutomaton>,
    field: Field,
}

impl RegexAutomatonQuery {
    /// Creates a query over `field`.
    pub fn new(automaton: Arc<RegexAutomaton>, field: Field) -> RegexAutomatonQuery {
        RegexAutomatonQuery { automaton, field }
    }
}

impl Query for RegexAutomatonQuery {
    fn weight(&self, _enabled_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        Ok(Box::new(AutomatonWeight::<RegexAutomaton>::new(
            self.field,
            self.automaton.clone(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tantivy_fst::{Automaton, Regex};

    use super::{RegexAutomaton, RegexAutomatonError, RegexAutomatonQuery};
    use crate::collector::Count;
    use crate::schema::{Schema, STRING};
    use crate::{Index, IndexWriter};

    #[test]
    fn test_matches_and_prunes_like_tantivy_fst() {
        let alphabet = [
            "a", "b", "c", "e", "h", "k", "l", "o", "0", "_", "K", "ç", "ş", "İ", "ı", "é", "日",
            "本", "\u{212A}", "😀", " ",
        ];
        let mut words = vec!["hello".to_string(), "ab".to_string()];
        let mut seed: u64 = 7;
        for _ in 0..20_000 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let len = 1 + (seed >> 61) as usize;
            let mut word = String::new();
            for _ in 0..len {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                word.push_str(alphabet[((seed >> 33) % alphabet.len() as u64) as usize]);
            }
            words.push(word);
        }

        let patterns = [
            "(?i)hel+o",
            "(?i).*o.*",
            "(?i)\\w+",
            "(?i)[a-z0-9_]{1,3}",
            "(?i)ç.*",
            "(?i).*ı",
            "(?i)k",
            "(?i)[^a]+",
            "(?i).{2,4}",
            "(?i)\\pL+",
            "(?i)\\d+",
            "(?i)ş?e?l*",
            "(?i)日.",
            "İ.*",
            "a|ab",
            "(|a)b*",
            "h|hello",
            "0|1|01",
            "(日|日本)本?",
            "ab.",
            "(?s).?",
        ];
        for pattern in patterns {
            let expected = Regex::new(pattern).unwrap();
            let automaton = RegexAutomaton::new(pattern, 1 << 20).unwrap();
            assert_eq!(
                automaton.can_match(&automaton.start()),
                expected.can_match(&expected.start()),
                "can_match of {pattern:?} at the start"
            );
            assert_eq!(
                automaton.is_match(&automaton.start()),
                expected.is_match(&expected.start()),
                "is_match of {pattern:?} on the empty term"
            );
            for word in &words {
                let mut expected_state = expected.start();
                let mut state = automaton.start();
                for &byte in word.as_bytes() {
                    expected_state = expected.accept(&expected_state, byte);
                    state = automaton.accept(&state, byte);
                    assert_eq!(
                        automaton.can_match(&state),
                        expected.can_match(&expected_state),
                        "can_match of {pattern:?} after a prefix of {word:?}"
                    );
                    assert_eq!(
                        automaton.is_match(&state),
                        expected.is_match(&expected_state),
                        "is_match of {pattern:?} on a prefix of {word:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_look_arounds_are_rejected() {
        for pattern in ["^a", "a$", "\\bfoo", "a\\B"] {
            assert_eq!(
                RegexAutomaton::new(pattern, 1 << 20).unwrap_err(),
                RegexAutomatonError::LookAround,
                "{pattern}"
            );
        }
    }

    #[test]
    fn test_size_limit() {
        let dots = ".?".repeat(400);
        for pattern in ["(|0){0,99999}", "\\w(0?){1000}", dots.as_str()] {
            assert_eq!(
                RegexAutomaton::new(pattern, 512 << 10).unwrap_err(),
                RegexAutomatonError::TooComplex,
                "{pattern}"
            );
        }
        RegexAutomaton::new("(?i)\\w+@\\w+", 512 << 10).unwrap();
    }

    #[test]
    fn test_lazy_quantifier_ascii_class_and_empty_class() {
        let lazy = RegexAutomaton::new("a+?b", 1 << 20).unwrap();
        let mut state = lazy.start();
        for &byte in b"aab" {
            state = lazy.accept(&state, byte);
        }
        assert!(lazy.is_match(&state));

        let ascii_word = RegexAutomaton::new("(?-u:\\w)+", 1 << 20).unwrap();
        let mut state = ascii_word.start();
        for &byte in "é".as_bytes() {
            state = ascii_word.accept(&state, byte);
        }
        assert!(!ascii_word.can_match(&state));

        let empty_class = RegexAutomaton::new("[a&&b]", 1 << 20).unwrap();
        assert!(!empty_class.can_match(&empty_class.start()));
    }

    #[test]
    fn test_query() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let country = schema_builder.add_text_field("country", STRING);
        let index = Index::create_in_ram(schema_builder.build());
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(doc!(country => "japan"))?;
        index_writer.add_document(doc!(country => "korea"))?;
        index_writer.add_document(doc!(country => "jamaica"))?;
        index_writer.commit()?;
        let searcher = index.reader()?.searcher();

        let automaton = Arc::new(RegexAutomaton::new("ja.*", 1 << 20).unwrap());
        let count = searcher.search(&RegexAutomatonQuery::new(automaton, country), &Count)?;
        assert_eq!(count, 2);
        Ok(())
    }
}
