use super::Distance;

/// Sink state. See [DFA](./index.html)
pub const SINK_STATE: u32 = 0u32;

/// Implementation of a Deterministic Finite Automaton for
/// a Levenshtein Automaton targeting UTF-8 encoded strings.
///
/// The automaton does not validate `utf-8`.
/// It will not return errors when fed with invalid `utf-8`
///
/// The only `sink` state is guaranteed to be `SINK`.
///
/// This means that if you reach the sink state you are
/// guaranteed that regardless of the sequence of bytes
/// you might consume in the future, you will always
/// remain in the same state.
///
/// This property can be exploited to abort further
/// evaluation.
///
///
/// # Usage
///
/// ```rust
/// # extern crate levenshtein_automata;
/// # use levenshtein_automata::{LevenshteinAutomatonBuilder, Distance};
/// # use levenshtein_automata::SINK_STATE;
/// # fn main() {
/// #   // Building this factory is not free.
/// #   let lev_automaton_builder = LevenshteinAutomatonBuilder::new(2, true);
/// #  // We can now build an entire dfa.
/// #  let dfa = lev_automaton_builder.build_dfa("Levenshtein");
/// # let str = "Levenshtain";
/// let mut state = dfa.initial_state();
/// for &byte in str.as_bytes() {
///     state = dfa.transition(state, byte);
///     if state == SINK_STATE {
///         break;
///     }
/// }
/// let distance = dfa.distance(state);
/// # }
//```
pub struct DFA {
    // Bytes that every state sends to the same next state share a class.
    classes: [u8; 256],
    // One row of `stride` entries per state: the next state of each class, then the encoded
    // distance of the state. A state id is the offset of its row, so `SINK_STATE` (row 0) is 0.
    stride: u32,
    table: Vec<u32>,
    initial_state: u32,
}

const AT_LEAST_FLAG: u32 = 1 << 8;

fn encode_distance(distance: Distance) -> u32 {
    match distance {
        Distance::Exact(d) => u32::from(d),
        Distance::AtLeast(d) => AT_LEAST_FLAG | u32::from(d),
    }
}

impl DFA {
    /// Returns the initial state
    pub fn initial_state(&self) -> u32 {
        self.initial_state
    }

    /// Helper function that consumes all of the bytes
    /// a sequence of bytes and returns the resulting
    /// distance.
    pub fn eval<B: AsRef<[u8]>>(&self, text: B) -> Distance {
        let mut state = self.initial_state();
        for &b in text.as_ref() {
            state = self.transition(state, b);
        }
        self.distance(state)
    }

    /// Returns the Levenshtein distance associated to the
    /// current state.
    pub fn distance(&self, state_id: u32) -> Distance {
        let encoded = self.table[(state_id + self.stride - 1) as usize];
        if encoded & AT_LEAST_FLAG != 0 {
            Distance::AtLeast(encoded as u8)
        } else {
            Distance::Exact(encoded as u8)
        }
    }

    /// Returns the number of states in the `DFA`.
    pub fn num_states(&self) -> usize {
        self.table.len() / self.stride as usize
    }

    /// Returns the destination state reached after consuming a given byte.
    pub fn transition(&self, from_state_id: u32, b: u8) -> u32 {
        self.table[(from_state_id + u32::from(self.classes[b as usize])) as usize]
    }

    /// Returns the bytes of the transition table and the class map.
    pub fn memory_usage(&self) -> usize {
        self.table.capacity() * 4 + self.classes.len()
    }
}

#[cfg(feature = "fst_automaton")]
use fst;
#[cfg(feature = "fst_automaton")]
impl fst::Automaton for DFA {
    type State = u32;

    fn start(&self) -> u32 {
        self.initial_state()
    }

    fn is_match(&self, state: &u32) -> bool {
        match self.distance(*state) {
            Distance::Exact(_) => true,
            Distance::AtLeast(_) => false,
        }
    }

    fn can_match(&self, state: &u32) -> bool {
        *state != SINK_STATE
    }

    fn accept(&self, state: &u32, byte: u8) -> u32 {
        self.transition(*state, byte)
    }
}

// The byte ranges `add_state` fills with one value each: 1-byte characters and continuation
// bytes, then the lead bytes of 2-, 3- and 4-byte characters.
const BYTE_RANGES: [(usize, usize); 4] = [(0, 192), (192, 224), (224, 240), (240, 256)];

pub struct Utf8DFAStateBuilder<'a> {
    dfa_builder: &'a mut Utf8DFABuilder,
    state_id: u32,
    default_successor: [u32; 4],
}

impl<'a> Utf8DFAStateBuilder<'a> {
    /// Returns `None` when the `DFA` would pass the builder's byte limit.
    pub fn add_transition(&mut self, chr: char, to_state_id: u32) -> Option<()> {
        let mut buffer = [0u8; 4];

        // The char may translate into more than one bytes.
        // We create a chain for this reason.
        let bytes: &[u8] = chr.encode_utf8(&mut buffer).as_bytes();
        let mut from_state_id_decoded = self.state_id;
        for (i, b) in bytes[..bytes.len() - 1].iter().cloned().enumerate() {
            let remaining_num_bytes = bytes.len() - i as usize - 1 as usize;
            let default_successor = self.default_successor[remaining_num_bytes];
            let mut intermediary_state_id: u32 = self.dfa_builder.get(from_state_id_decoded, b);
            if intermediary_state_id == default_successor {
                intermediary_state_id = self.dfa_builder.allocate()?;
                self.dfa_builder.fill_row(
                    intermediary_state_id,
                    self.default_successor[remaining_num_bytes - 1],
                );
            }
            self.dfa_builder
                .set(from_state_id_decoded, b, intermediary_state_id);
            from_state_id_decoded = intermediary_state_id;
        }

        let to_state_id_decoded = self
            .dfa_builder
            .get_or_allocate(Utf8StateId::original(to_state_id))?;
        self.dfa_builder.set(
            from_state_id_decoded,
            bytes[bytes.len() - 1],
            to_state_id_decoded,
        );
        Some(())
    }
}

/// `Utf8DFABuilder` makes it possible to define a DFA
/// that takes unicode character, and build a `DFA`
/// that operates on utf-8 encoded `&[u8]`.
pub struct Utf8DFABuilder {
    index: Vec<Option<u32>>,
    classes: [u8; 256],
    // The classes of each of `BYTE_RANGES`.
    range_classes: [Vec<u8>; 4],
    stride: u32,
    table: Vec<u32>,
    // The `table` length the byte limit allows.
    max_table_len: usize,
    initial_state: u32,
    max_num_states: u32,
}

#[derive(Eq, PartialEq, Hash, Clone, Copy)]
struct Utf8StateId(u32);
impl Utf8StateId {
    pub fn original(state_id: u32) -> Utf8StateId {
        Utf8StateId::predecessor(state_id, 0u8)
    }

    pub fn predecessor(state_id: u32, num_steps: u8) -> Utf8StateId {
        Utf8StateId(state_id * 4u32 + u32::from(num_steps))
    }
}

impl Utf8DFABuilder {
    /// Creates a new dictionary.
    ///
    /// The `builder` will only accept `state_id` that are
    /// lower than `max_num_states`. `alphabet` holds every character a transition can name.
    /// The built `DFA` takes at most `max_bytes`.
    pub fn new(max_num_states: usize, alphabet: &[char], max_bytes: usize) -> Utf8DFABuilder {
        // Only `add_transition` gives a byte a value of its own: the bytes of the alphabet's
        // characters. Every other byte keeps the value its range got in `add_state`, or the one
        // value an intermediary or predecessor row holds for all bytes.
        let mut is_named = [false; 256];
        for chr in alphabet {
            let mut buffer = [0u8; 4];
            for &b in chr.encode_utf8(&mut buffer).as_bytes() {
                is_named[b as usize] = true;
            }
        }
        let mut classes = [0u8; 256];
        let mut range_classes: [Vec<u8>; 4] = Default::default();
        let mut num_classes: u32 = 0;
        for (range_id, &(start, end)) in BYTE_RANGES.iter().enumerate() {
            let mut range_class = None;
            for b in start..end {
                let class = if is_named[b] {
                    num_classes += 1;
                    (num_classes - 1) as u8
                } else {
                    match range_class {
                        Some(class) => class,
                        None => {
                            num_classes += 1;
                            range_class = Some((num_classes - 1) as u8);
                            (num_classes - 1) as u8
                        }
                    }
                };
                classes[b] = class;
                if !range_classes[range_id].contains(&class) {
                    range_classes[range_id].push(class);
                }
            }
        }

        let stride = num_classes + 1;
        let max_table_len = max_bytes.saturating_sub(classes.len()) / 4;
        Utf8DFABuilder {
            index: vec![None; max_num_states * 4 + 3],
            classes,
            range_classes,
            stride,
            table: Vec::new(),
            max_table_len: max_table_len - max_table_len % stride as usize,
            initial_state: 0u32,
            max_num_states: max_num_states as u32,
        }
    }

    fn get(&self, state_id: u32, b: u8) -> u32 {
        self.table[(state_id + u32::from(self.classes[b as usize])) as usize]
    }

    fn set(&mut self, state_id: u32, b: u8, to_state_id: u32) {
        self.table[(state_id + u32::from(self.classes[b as usize])) as usize] = to_state_id;
    }

    fn fill_row(&mut self, state_id: u32, to_state_id: u32) {
        let start = state_id as usize;
        for slot in &mut self.table[start..start + self.stride as usize - 1] {
            *slot = to_state_id;
        }
    }

    fn allocate(&mut self) -> Option<u32> {
        let new_state = self.table.len() as u32;
        let new_len = self.table.len() + self.stride as usize;
        if new_len > self.max_table_len {
            return None;
        }
        if new_len > self.table.capacity() {
            let target = (self.table.capacity() * 2)
                .max(new_len)
                .min(self.max_table_len);
            self.table.reserve_exact(target - self.table.len());
        }
        self.table.resize(new_len, 0u32);
        self.table[new_len - 1] = encode_distance(Distance::AtLeast(255));
        Some(new_state)
    }

    fn get_or_allocate(&mut self, state: Utf8StateId) -> Option<u32> {
        let state_bucket = state.0 as usize;
        if let Some(state) = self.index[state_bucket] {
            return Some(state);
        }
        let new_state = self.allocate()?;
        self.index[state_bucket] = Some(new_state);
        Some(new_state)
    }

    pub fn set_initial_state(&mut self, initial_state: u32) -> Option<()> {
        let state_id_decoded = self.get_or_allocate(Utf8StateId::original(initial_state))?;
        self.initial_state = state_id_decoded;
        Some(())
    }

    /// Define a new state. Returns `None` when the `DFA` would pass the byte limit.
    pub fn add_state(
        &mut self,
        state: u32,
        distance: Distance,
        default_successor_orig: u32,
    ) -> Option<Utf8DFAStateBuilder<'_>> {
        assert!(
            state < self.max_num_states,
            "State id is larger than max_num_states"
        );
        let state_id = self.get_or_allocate(Utf8StateId::original(state))?;
        self.table[(state_id + self.stride - 1) as usize] = encode_distance(distance);

        let default_successor_id =
            self.get_or_allocate(Utf8StateId::original(default_successor_orig))?;

        // creates a chain of states of predecessors of `default_successor_orig`.
        // Accepting k-bytes (whatever the bytes are) from `predecessor_states[k-1]`
        // leads to the `default_successor_orig` state.
        let mut predecessor_states = [default_successor_id; 4];

        {
            for num_bytes in 1..4 {
                let predecessor_state =
                    Utf8StateId::predecessor(default_successor_orig, num_bytes as u8);
                let predecessor_state_id = self.get_or_allocate(predecessor_state)?;
                predecessor_states[num_bytes] = predecessor_state_id;
                let succ = predecessor_states[num_bytes - 1];
                self.fill_row(predecessor_state_id, succ);
            }
        }

        for range_id in 0..BYTE_RANGES.len() {
            for &class in &self.range_classes[range_id] {
                self.table[(state_id + u32::from(class)) as usize] = predecessor_states[range_id];
            }
        }

        Some(Utf8DFAStateBuilder {
            dfa_builder: self,
            state_id,
            default_successor: predecessor_states,
        })
    }

    pub fn build(mut self) -> DFA {
        self.table.shrink_to_fit();
        DFA {
            classes: self.classes,
            stride: self.stride,
            table: self.table,
            initial_state: self.initial_state,
        }
    }
}

#[cfg(test)]
mod tests {

    use super::{Distance, Utf8DFABuilder};

    #[test]
    fn test_utf8_dfa_builder() {
        let mut dfa_builder = Utf8DFABuilder::new(2, &[], usize::MAX);
        dfa_builder.add_state(0, Distance::Exact(1u8), 1).unwrap();
        dfa_builder.add_state(1, Distance::Exact(0u8), 0).unwrap();
        dfa_builder.set_initial_state(1u32).unwrap();
        let dfa = dfa_builder.build();
        let parity_num_letters = |s: &str| dfa.eval(s).to_u8();
        assert_eq!(parity_num_letters("abcdef"), 0u8);
        assert_eq!(parity_num_letters("a"), 1u8);
        assert_eq!(parity_num_letters("aあ"), 0u8);
        assert_eq!(parity_num_letters("❤"), 1u8);
        assert_eq!(parity_num_letters("❤❤"), 0u8);
        assert_eq!(parity_num_letters("❤a"), 0u8);
        assert_eq!(parity_num_letters("あ"), 1u8);
        assert_eq!(parity_num_letters("ああ"), 0u8);
    }
}
