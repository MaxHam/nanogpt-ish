use std::{
    collections::{BinaryHeap, HashMap},
    hash::Hash,
    vec,
};

type Utf8Byte = u8;

#[derive(Eq, Ord, PartialEq, PartialOrd, Hash, Debug, Clone)]
pub struct Token {
    pub id: u16,
    value: Vec<Utf8Byte>,
}

impl Token {
    pub fn new(id: u16, value: Vec<Utf8Byte>) -> Token {
        Token { id, value }
    }
    fn from_pair(id: &u16, pair: &(Token, Token)) -> Token {
        let mut new_bytes = Vec::new();
        new_bytes.extend_from_slice(&pair.0.value);
        new_bytes.extend_from_slice(&pair.1.value);
        Token {
            id: *id,
            value: new_bytes,
        }
    }
    pub fn from_byte(byte: Utf8Byte) -> Token {
        Token {
            id: byte as u16,
            value: vec![byte],
        }
    }
}

#[derive(PartialEq, Debug)]
pub struct Tokenizer {
    pub vocabulary: HashMap<u16, Token>,
    merge_rules: Vec<((Token, Token), u16)>,
    // (pair, merged_id), the rank of the merge rule is defind by its pos in the vector
}

impl Tokenizer {
    pub fn from_bytes() -> Tokenizer {
        // init a vocab from the given u8 bytes
        let mut tokens: HashMap<u16, Token> = HashMap::new();
        for id in 0u8..=255u8 {
            tokens.insert(id as u16, Token::from_byte(id));
        }
        Tokenizer {
            vocabulary: tokens,
            merge_rules: vec![],
        }
    }

    pub fn from_merges(merge_rules: Vec<((u16, u16), u16)>) -> Tokenizer {
        if merge_rules.is_empty() {
            panic!("You need to train the byte pair encoder before you can get a vocabulary.");
        }

        let mut tokenizer: Tokenizer = Tokenizer::from_bytes();

        let mut token_merge_rules: Vec<((Token, Token), u16)> = Vec::with_capacity(merge_rules.len());

        for ((a, b), token_id) in merge_rules {
            let token_a = tokenizer.vocabulary

                .get(&a)
                .unwrap_or_else(|| panic!("Token id {} not found when building merge rule for id {}", a, token_id))
                .clone();
            let token_b = tokenizer.vocabulary
                .get(&b)
                .unwrap_or_else(|| panic!("Token id {} not found when building merge rule for id {}", b, token_id))
                .clone();

            let pair = (token_a, token_b);

            if !tokenizer.vocabulary.contains_key(&pair.0.id) || !tokenizer.vocabulary.contains_key(&pair.1.id) {
                panic!(
                    "Token pair ({:?}, {:?}) not both present in vocabulary when applying merge rule for token_id {}.",
                    pair.0, pair.1, token_id
                );
            }

            let new_token = Token::from_pair(&token_id, &pair);
            tokenizer.vocabulary.insert(token_id, new_token);
            token_merge_rules.push((pair, token_id));
        }

        tokenizer.merge_rules = token_merge_rules;
        tokenizer
    }

    pub fn to_json(&self) {
        // Map token ids to decoded UTF-8 strings for readable JSON
        let vocab_strings: HashMap<&u16, String> = self
            .vocabulary
            .iter()
            .map(|(token_id,token)| match String::from_utf8(token.value.clone()) {
                Ok(s) => (token_id, s),
                Err(_) => (token_id, "UNKNOWN".to_string()),
            })
            .collect();

        let json = serde_json::to_string_pretty(&vocab_strings).expect("Failed to serialize vocab");
        std::fs::write("vocab.json", json).expect("Failed to write vocab.json");
    }

    pub fn encode(&self, text: &str) -> Vec<Token> {
        let mut tokens: Vec<Token> = text.bytes().map(Token::from_byte).collect();
        for (pair, token_id) in &self.merge_rules {
            tokens = replace_pair(&tokens, pair, token_id);
        }

        tokens
    }

    pub fn decode(&self, tokens: &[Token]) -> String {
        let mut target_tokens: Vec<Token> = tokens.to_vec();
        for (pair, token_id) in self.merge_rules.iter().rev() {
            target_tokens = expand_token(target_tokens, pair.clone(), *token_id);
        }
        let mut bytes = Vec::new();
        for token in &target_tokens {
            bytes.extend_from_slice(&token.value);
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    pub fn train(corpus: &str, vocab_size: u16) -> Result<Tokenizer, &str> {
        if vocab_size <= 256 {
            return Err(
                "Vocabulary size must be greater than 256 tokens; otherwise, no training is possible.",
            );
        }
        let num_merges = vocab_size - 256;
        let mut corpus_tokens: Vec<u16> = corpus.as_bytes().iter().map(|&b| b as u16).collect();
        // if not a single pair can be found then exit early
        // e.g. corpus "a"
        if corpus_tokens.len() <= 1 {
            return Ok(Tokenizer::from_bytes());
        }

        // init count of pairs
        let mut pair_counts: HashMap<(u16, u16), usize> = HashMap::new();
        for i in 0..corpus_tokens.len() - 1 {
            let pair = (corpus_tokens[i], corpus_tokens[i + 1]);
            *pair_counts.entry(pair).or_insert(0) += 1;
        }

        // store the counts of pairs in a heap
        // allows us to get the most frequent pair with O(1)
        let mut heap: BinaryHeap<(usize, (u16, u16))> =
            BinaryHeap::with_capacity(pair_counts.len());
        for (pair, count) in &pair_counts {
            heap.push((*count, *pair));
        }

        let mut merge_rules: Vec<((u16, u16), u16)> = Vec::with_capacity(num_merges.into());
        let mut next_token_id: u16 = 256;

        for (i, _) in (0..num_merges).enumerate() {
            eprintln!("Merge {}/{}", i + 1, num_merges);
            // find the most frequent pair
            // if its stale then we discard it (lazy deletion)
            let most_frequent_pair = loop {
                match heap.pop() {
                    None => break None,
                    Some((count, pair)) => {
                        // Check if this entry is still valid
                        let actual_count = pair_counts.get(&pair).copied().unwrap_or(0);
                        if actual_count == count && actual_count > 0 {
                            break Some((pair, count));
                        }
                        // Otherwise it's stale — discard and try again
                    }
                }
            };

            let (most_frequent_pair, _count) = match most_frequent_pair {
                Some(p) => p,
                None => break, // No more pairs to merge
            };
            apply_merge(
                &mut corpus_tokens,
                &mut pair_counts,
                &mut heap,
                &most_frequent_pair,
                next_token_id,
            );
            merge_rules.push((most_frequent_pair, next_token_id));

            next_token_id += 1;
        }

        Ok(Tokenizer::from_merges(merge_rules))
    }
}
fn apply_merge(
    tokens: &mut Vec<u16>,
    pair_counts: &mut HashMap<(u16, u16), usize>,
    heap: &mut BinaryHeap<(usize, (u16, u16))>,
    pair: &(u16, u16),
    new_id: u16,
) {
    // Iterate the token array and search for adjacent token pairs that match the pair
    // If pair is found then decrement count of old pairs and increment count of given pair
    let mut new_tokens: Vec<u16> = Vec::with_capacity(tokens.len());
    let mut i = 0;
    let mut prev_merged_right_pair: Option<(u16, u16)> = None;
    while i < tokens.len() - 1 {
        if tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
            let left_neighbor = if i > 0 { Some(tokens[i - 1]) } else { None };
            let right_neighbor = if i + 2 < tokens.len() {
                Some(tokens[i + 2])
            } else {
                None
            };

            // --- Decrement old neighbor pairs ---
            // Skip left_pair if it equals prev right_pair (consecutive merges double-count boundary)
            let left_pair = if i > 0 {
                Some((tokens[i - 1], tokens[i]))
            } else {
                None
            };
            if let Some(lp) = left_pair
                && prev_merged_right_pair != Some(lp) {
                    decrement_pair(pair_counts, lp);
                }
            let right_pair = if i + 2 < tokens.len() {
                Some((tokens[i + 1], tokens[i + 2]))
            } else {
                None
            };
            if let Some(rp) = right_pair {
                decrement_pair(pair_counts, rp);
            }

            decrement_pair(pair_counts, *pair);

            new_tokens.push(new_id);
            i += 2;
            prev_merged_right_pair = right_pair;

            if let Some(left) = left_neighbor {
                increment_pair(pair_counts, heap, (left, new_id));
            }
            if let Some(right) = right_neighbor {
                increment_pair(pair_counts, heap, (new_id, right));
            }
        } else {
            new_tokens.push(tokens[i]);
            i += 1;
            prev_merged_right_pair = None;
        }
    }
    if i == tokens.len() - 1 {
        new_tokens.push(tokens[i]);
    }
    *tokens = new_tokens
}

fn decrement_pair(counts: &mut HashMap<(u16, u16), usize>, pair: (u16, u16)) {
    if let Some(c) = counts.get_mut(&pair) {
        *c -= 1;
        // Don't push to heap — the old entry will be detected as stale
    }
}

fn increment_pair(
    counts: &mut HashMap<(u16, u16), usize>,
    heap: &mut BinaryHeap<(usize, (u16, u16))>,
    pair: (u16, u16),
) {
    let count = counts.entry(pair).or_insert(0);
    *count += 1;
    // Push new count to heap (old entries become stale automatically)
    heap.push((*count, pair));
}
fn replace_pair(tokens: &[Token], pair: &(Token, Token), token_id: &u16) -> Vec<Token> {
    let mut result: Vec<Token> = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        if i < tokens.len() - 1 && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
            result.push(Token::from_pair(token_id, pair));
            i += 2;
        } else {
            result.push(tokens[i].clone());
            i += 1;
        }
    }
    result
}

fn expand_token(mut tokens: Vec<Token>, pair: (Token, Token), token_id: u16) -> Vec<Token> {
    let mut i = 0;
    while i < tokens.len() {
        if tokens[i].id == token_id {
            // Replace token_id with the original pair
            tokens[i] = pair.0.clone();
            tokens.insert(i + 1, pair.1.clone());
            i += 2; // Skip both inserted tokens
        } else {
            i += 1;
        }
    }
    tokens
}

#[cfg(test)]
mod tests {
    use std::{fs::read_to_string, path::Path};

    use crate::bpe::{Token, Tokenizer};

    #[test]
    fn test_train() {
        // Given
        let corpus = "foo bar baz";

        // When
        let tokenizer = Tokenizer::train(corpus, 257).unwrap();

        // Then
        let merged_token =
            Token::from_pair(&256, &(Token::from_byte(b'b'), Token::from_byte(b'a')));
        assert!(tokenizer.vocabulary.contains_key(&merged_token.id));
    }

    #[test]
    fn test_train_too_little_vocab_size() {
        // Given
        let corpus = "foo bar baz";

        // When
        let tokenizer = Tokenizer::train(corpus, 1);

        // Then error
        assert!(tokenizer.is_err());
    }
    #[test]
    fn test_train_single_byte_corpus() {
        // Given
        let corpus = "a";

        // When
        let tokenizer = Tokenizer::train(corpus, 257);

        // Then
        assert_eq!(tokenizer, Ok(Tokenizer::from_bytes()));
    }

    #[test]
    fn test_encode() {
        // Given
        let corpus = "foo bar baz";
        let encoder = Tokenizer::train(corpus, 257).unwrap();
        let text = "foo bar baz"; // Use the same corpus to ensure merge rules apply

        // When
        let encoded = encoder.encode(text);

        // Then
        let decoded = encoder.decode(&encoded);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_decode() {
        // Given
        let corpus = "foo bar baz";
        let encoder = Tokenizer::train(corpus, 257).unwrap();

        // When
        let encoded = encoder.encode(corpus);
        let decoded = encoder.decode(&encoded);

        // Then
        assert_eq!(decoded, corpus);
    }

    #[test]
    fn test_vocabulary() {
        // Given
        let corpus = "foo bar baz";

        // When
        let tokenizer = Tokenizer::train(corpus, 257).unwrap();

        // Then
        // Vocabulary should contain all 256 base bytes plus 1 merged token
        assert_eq!(tokenizer.vocabulary.len(), 257);
        assert!(tokenizer.vocabulary.contains_key(&Token::from_byte(b'f').id));
        assert!(
            tokenizer
                .vocabulary
                .contains_key(&Token::new(256, vec![b'b', b'a']).id)
        );
    }


    #[test]
    fn test_no_duplicates_in_vocab() {
        // Given
        let path = Path::new("test_corpus.txt");
        let corpus = read_to_string(path).expect("Failed to read corpus file");
        
        // When
        let tokenizer = Tokenizer::train(corpus.as_str(), 512).unwrap();

        // Then
        // Assert that every token in the vocabulary appears only once
        for token in &tokenizer.vocabulary {
            let count = tokenizer.vocabulary.iter().filter(|t| *t == token).count();
            assert_eq!(count, 1, "Token {:?} is stored more than once", token);
        }
    }
    #[test]
    fn test_decode_with_corpus() {
        // Given
        let path = Path::new("test_corpus.txt");
        let corpus = read_to_string(path).expect("Failed to read corpus file");
        let tokenizer = Tokenizer::train(corpus.as_str(), 259).unwrap();

        // When
        let encoded = tokenizer.encode("Hi World!");
        let decoded = tokenizer.decode(&encoded);

        // Then
        assert_eq!(decoded, "Hi World!");
    }
}
