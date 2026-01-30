use std::{
    collections::{HashMap, HashSet},
    vec,
};

type Utf8Byte = u8;

#[derive(PartialEq, Debug)]
pub struct BytePairEncoder {
    merge_rules: Vec<((u16, u16), u16)>, // (pair, merged_id)
    vocab_size: usize,                   // 256 + num_merges
}

impl BytePairEncoder {
    fn encode(&self, text: &str) -> Vec<u16> {
        let mut tokens: Vec<u16> = text.bytes().map(|b| b as u16).collect();

        for (pair, token_id) in &self.merge_rules {
            tokens = replace_pair(&tokens, *pair, *token_id);
        }

        tokens
    }

    fn decode(&self, tokens: &[u16]) -> String {
        let mut target_tokens: Vec<u16> = tokens.to_vec();
        for (pair, token_id) in self.merge_rules.iter().rev() {
            target_tokens = expand_token(target_tokens, *pair, *token_id);
        }
        // Convert u16 tokens back to bytes and then to string
        let bytes: Vec<Utf8Byte> = target_tokens.iter().map(|&t| t as u8).collect();
        String::from_utf8(bytes).unwrap()
    }

    pub fn train(corpus: &str, num_merges: u8) -> BytePairEncoder {
        let mut tokens: Vec<u16> = corpus.bytes().map(|b| b as u16).collect();

        let mut vocab: HashSet<u16> = (0u16..=255u16).collect();
        let mut merge_rules: Vec<((u16, u16), u16)> = vec![];
        let mut next_token_id: u16 = 256;

        (0..num_merges).for_each(|_| {
            // if not a single pair can be found then exit early
            // e.g. corpus "a"
            if tokens.len() <= 1 {
                return;
            }
            let mut pair_frequencies: HashMap<(u16, u16), i32> = HashMap::new();
            (0..tokens.len() - 1).for_each(|j| {
                let pair = (tokens[j], tokens[j + 1]);
                let count = pair_frequencies.get(&pair).unwrap_or(&0) + 1;
                pair_frequencies.insert(pair, count);
            });

            let most_frequent_pair = pair_frequencies
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| a.0.cmp(b.0)))
                .map(|(k, _)| *k)
                .unwrap();

            merge_rules.push((most_frequent_pair, next_token_id));
            vocab.insert(next_token_id);

            tokens = replace_pair(&tokens, most_frequent_pair, next_token_id);

            next_token_id += 1;
        });

        BytePairEncoder {
            merge_rules,
            vocab_size: vocab.len(),
        }
    }

    pub fn vocabulary(&self) -> HashMap<u16, Vec<Utf8Byte>> {
        if self.merge_rules.is_empty() {
            panic!("You need to train the byte pair encoder before you can get a vocabulary.");
        }

        // Prefill vocab: each u16 in 0..=255 is a vector of Vec<u8> containing its byte value
        let mut vocab: HashMap<u16, Vec<Utf8Byte>> =
            (0u16..=255u16).map(|i| (i, vec![i as u8])).collect();

        // For each merge, create the bytes representation by concatenating its parts
        for (pair, token_id) in &self.merge_rules {
            let mut bytes = vec![];
            if let Some(left) = vocab.get(&pair.0) {
                bytes.extend_from_slice(left);
            }
            if let Some(right) = vocab.get(&pair.1) {
                bytes.extend_from_slice(right);
            }
            vocab.insert(*token_id, bytes);
        }

        vocab
    }
}

fn replace_pair(tokens: &[u16], pair: (u16, u16), token_id: u16) -> Vec<u16> {
    let mut result = vec![];
    let mut i = 0;
    while i < tokens.len() {
        if i < tokens.len() - 1 && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
            result.push(token_id);
            i += 2;
        } else {
            result.push(tokens[i]);
            i += 1;
        }
    }
    result
}

fn expand_token(mut tokens: Vec<u16>, pair: (u16, u16), token_id: u16) -> Vec<u16> {
    let mut i = 0;
    while i < tokens.len() {
        if tokens[i] == token_id {
            // Replace token_id with the original pair
            tokens[i] = pair.0;
            tokens.insert(i + 1, pair.1);
            i += 2; // Skip both inserted tokens
        } else {
            i += 1;
        }
    }
    tokens.to_vec()
}

#[test]
fn test_train() {
    // Given
    let corpus = "foo bar baz";

    // When
    let encoder = BytePairEncoder::train(corpus, 1);

    // Then
    assert_eq!(
        encoder,
        BytePairEncoder {
            merge_rules: vec![(((b'b' as u16), (b'a' as u16)), 256u16)],
            vocab_size: 257
        }
    );
}

#[test]
fn test_train_single_byte_corpus() {
    // Given
    let corpus = "a";

    // When
    let rule = BytePairEncoder::train(corpus, 1);

    // Then
    assert_eq!(
        rule,
        BytePairEncoder {
            merge_rules: vec![],
            vocab_size: 256
        }
    );
}

#[test]
fn test_encode() {
    // Given
    let corpus = "foo bar baz";
    let encoder = BytePairEncoder::train(corpus, 1);
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
    let encoder = BytePairEncoder::train(corpus, 1);

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
    let encoder = BytePairEncoder::train(corpus, 1);

    // When
    let vocab = encoder.vocabulary();

    // Then
    // Vocabulary should contain all 256 base bytes plus 1 merged token
    assert_eq!(vocab.len(), 257);
    assert_eq!(vocab.get(&102u16), Some(vec![b'f'].as_ref()));
    // The merged token 256 should be "ba" (most frequent pair)
    assert_eq!(vocab.get(&256u16), Some(vec![b'b', b'a'].as_ref()));
}
