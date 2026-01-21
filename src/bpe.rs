use std::{
    collections::{HashMap, HashSet},
    vec,
};

#[derive(PartialEq, Debug)]
struct BytePairEncoder {
    merge_rules: Vec<((u16, u16), u16)>, // (pair, merged_id)
    vocab_size: usize,                   // 256 + num_merges
}

impl BytePairEncoder {
    fn encode(&self, text: &str) -> Vec<u16>{
        let mut tokens: Vec<u16> = text.bytes().map(|b| b as u16).collect();

        for (pair, token_id) in &self.merge_rules {
            tokens = replace_pair(&tokens, *pair, *token_id);
        }

        tokens
    }

    fn decode(&self, tokens: &Vec<u16>) -> String {
        let mut target_tokens = tokens.clone();
        for (pair, token_id) in self.merge_rules.iter().rev() {
            target_tokens = expand_token(target_tokens, *pair, *token_id);
        }
        // Convert u16 tokens back to bytes and then to string
        let bytes: Vec<u8> = target_tokens.iter().map(|&t| t as u8).collect();
        String::from_utf8(bytes).unwrap()
    }
}

fn train(corpus: &str, num_merges: u8) -> BytePairEncoder {
    // initialize tokens with character bytes as u16 in a Rust-idiomatic way
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
            let pair = (tokens[j] as u16, tokens[j + 1] as u16);
            let count = pair_frequencies.get(&pair).unwrap_or(&0) + 1;
            pair_frequencies.insert(pair, count);
        });

        let most_frequent_pair = pair_frequencies
            .iter()
            .max_by_key(|entry| entry.1)
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

fn replace_pair(tokens: &Vec<u16>, pair: (u16, u16), token_id: u16) -> Vec<u16> {
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
    tokens
}

#[test]
fn test_train() {
    // Given
    let corpus = "foo bar baz";

    // When
    let encoder = train(corpus, 1);

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
    let rule = train(corpus, 1);

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
    let encoder = train(corpus, 1);
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
    let encoder = train(corpus, 1);

    // When
    let encoded = encoder.encode(corpus);
    let decoded = encoder.decode(&encoded);

    // Then
    assert_eq!(decoded, corpus);
}
