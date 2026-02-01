use std::{
    collections::{BinaryHeap, HashMap, HashSet}, hash::Hash, vec
};

type Utf8Byte = u8;

#[derive(Eq, Ord, PartialEq, PartialOrd, Hash, Debug, Clone)]
pub struct Token {
    id: u16,
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
    fn from_byte(byte: Utf8Byte) -> Token {
        Token {
            id: byte as u16,
            value: vec![byte],
        }
    }
}

#[derive(PartialEq, Debug)]
pub struct Tokenizer {
    pub vocabulary: HashSet<Token>,
    merge_rules: Vec<((Token, Token), u16)>,
    // (pair, merged_id), the rank of the merge rule is defind by its pos in the vector
}

impl Tokenizer {
    fn from_bytes() -> Tokenizer {
        // init a vocab from the given u8 bytes
        let tokens: HashSet<Token> = (0u8..=255u8).map(Token::from_byte).collect();
        Tokenizer {
            vocabulary: tokens,
            merge_rules: vec![],
        }
    }

    pub fn from_merges(merge_rules: Vec<((Token, Token), u16)>) -> Tokenizer {
        if merge_rules.is_empty() {
            panic!("You need to train the byte pair encoder before you can get a vocabulary.");
        }

        let mut tokenizer: Tokenizer = Tokenizer::from_bytes();
        // iterate merge rules in rank order
        for (pair, token_id) in &merge_rules {
            if !tokenizer.vocabulary.contains(&pair.0) || !tokenizer.vocabulary.contains(&pair.1) {
                panic!(
                    "Token pair ({:?}, {:?}) not both present in vocabulary when applying merge rule for token_id {}.",
                    pair.0, pair.1, token_id
                );
            }
            tokenizer
                .vocabulary
                .insert(Token::from_pair(token_id, pair));
        }
        tokenizer.merge_rules = merge_rules;
        tokenizer
    }

    pub fn to_json(&self) {
        // Map token ids to decoded UTF-8 strings for readable JSON
        let vocab_strings: HashMap<u16, String> = self
            .vocabulary
            .iter()
            .map(|token| match String::from_utf8(token.value.clone()) {
                Ok(s) => (token.id, s),
                Err(_) => (token.id, "UNKNOWN".to_string()),
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
        String::from_utf8(bytes).unwrap()
    }
}
#[derive(PartialEq, Debug)]
pub struct BytePairEncoder {
    pub merge_rules: Vec<((Token, Token), u16)>,
    // (pair, merged_id), the rank of the merge rule is defind by its pos in the vector
}

impl BytePairEncoder {
    pub fn train(corpus: &str, vocab_size: u16) -> Tokenizer {
        if vocab_size <= 256 {
            panic!("Vocabulary must need to be greater than 256 tokens otherwise no training is possible.")
        }
        let num_merges = vocab_size - 256;
        let mut corpus_tokens: Vec<u16> = corpus
        .as_bytes()
        .iter()
        .map(|&b| b as u16).collect();
        // if not a single pair can be found then exit early
        // e.g. corpus "a"
        if corpus_tokens.len() <= 1 {
            return Tokenizer::from_bytes()
        }
    
        // init count of pairs
        let mut pair_counts: HashMap<(u16, u16), usize> = HashMap::new();
        for i in 0..corpus_tokens.len() - 1 {
            let pair = (corpus_tokens[i], corpus_tokens[i+1]);
            *pair_counts.entry(pair).or_insert(0) += 1;
        };

        // store the counts of pairs in a heap
        // allows us to get the most frequent pair with O(1)
        let mut heap: BinaryHeap<(usize, (u16, u16))> = BinaryHeap::new();
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
            apply_merge(&mut corpus_tokens, &mut pair_counts, &mut heap, &most_frequent_pair, next_token_id);
            merge_rules.push((most_frequent_pair, next_token_id));
            
            next_token_id += 1;
        }

        // Convert merge_rules from Vec<((u16, u16), u16)> to Vec<((Token, Token), u16)>
        let merge_rules: Vec<((Token, Token), u16)> = merge_rules
            .into_iter()
            .map(|((a, b), id)| {
                (
                    (Token::from_byte(a as u8), Token::from_byte(b as u8)),
                    id,
                )
            })
            .collect();
        Tokenizer::from_merges(merge_rules)
    }
}
fn apply_merge(
    tokens: &mut Vec<u16>,
    pair_counts: &mut HashMap<(u16, u16), usize>,
    heap: &mut BinaryHeap<(usize, (u16, u16))>,
    pair: &(u16, u16),
    new_id: u16
) {
    // Iterate the token array and search for adjacent token pairs that match the pair
    // If pair is found then decrement count of old pairs and increment count of given pair
    let mut new_tokens: Vec<u16> = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() - 1 {
        if tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
            // --- Decrement old neighbor pairs ---
            if i > 0 {
                let left_pair = (tokens[i - 1], tokens[i]);
                decrement_pair(pair_counts, left_pair);
            }
            if i + 2 < tokens.len() {
                let right_pair = (tokens[i + 1], tokens[i + 2]);
                decrement_pair(pair_counts, right_pair);
            }
            
            // Decrement the merged pair itself
            decrement_pair(pair_counts, *pair);
            
            new_tokens.push(new_id);
            i += 2;
            
            // --- Increment new neighbor pairs ---
            if i > 0 {
                let new_left = (tokens[i - 1], tokens[i]);
                increment_pair(pair_counts, heap, new_left);
            }
            if i + 1 < tokens.len() {
                let new_right = (tokens[i], tokens[i + 1]);
                increment_pair(pair_counts, heap, new_right);
            }
            
        } else {
            i += 1; 
            new_tokens.push(tokens[i]);
        }
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

    use crate::bpe::{BytePairEncoder, Token, Tokenizer};


    #[test]
    fn test_train() {
        // Given
        let corpus = "foo bar baz";

        // When
        let tokenizer = BytePairEncoder::train(corpus, 257);

        // Then
        let merged_token =
            Token::from_pair(&256, &(Token::from_byte(b'b'), Token::from_byte(b'a')));
        assert!(tokenizer.vocabulary.contains(&merged_token));
    }


    #[test]
    #[should_panic]
    fn test_train_too_little_vocab_size() {
        // Given
        let corpus = "foo bar baz";

        // When
        BytePairEncoder::train(corpus, 1);

        // Then panic
    }
    #[test]
    fn test_train_single_byte_corpus() {
        // Given
        let corpus = "a";

        // When
        let tokenizer = BytePairEncoder::train(corpus, 257);

        // Then
        assert_eq!(tokenizer, Tokenizer::from_bytes());
    }

    #[test]
    fn test_encode() {
        // Given
        let corpus = "foo bar baz";
        let encoder = BytePairEncoder::train(corpus, 257);
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
        let encoder = BytePairEncoder::train(corpus, 257);

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
        let tokenizer = BytePairEncoder::train(corpus, 257);

        // Then
        // Vocabulary should contain all 256 base bytes plus 1 merged token
        assert_eq!(tokenizer.vocabulary.len(), 257);
        assert!(tokenizer.vocabulary.contains(&Token::from_byte(b'f')));
        assert!(
            tokenizer
                .vocabulary
                .contains(&Token::new(256, vec![b'b', b'a']))
        );
    }

    #[test]
    fn test_decode_with_corpus() {
        // Given
        let path = Path::new("test_corpus.txt");
        let corpus = read_to_string(path).expect("Failed to read corpus file");
        let tokenizer = BytePairEncoder::train(corpus.as_str(), 259);

        // When
        let encoded = tokenizer.encode("Hi World!");
        let decoded = tokenizer.decode(&encoded);

        // Then
        assert_eq!(decoded, "Hi World!");
    }
}
