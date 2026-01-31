use std::{
    collections::{HashMap, HashSet},
    vec,
};

type Utf8Byte = u8;

#[derive(Eq, Ord, PartialEq, PartialOrd, Hash, Debug, Clone)]
pub struct Token {
    id: u16,
    value: Vec<Utf8Byte>,
}

impl Token {
    fn new(id: u16, value: Vec<Utf8Byte>) -> Token {
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

    fn to_utf8_bytes(&self) -> Vec<Utf8Byte> {
        let bytes: Vec<Utf8Byte> = self.value.iter().map(|&t| t as Utf8Byte).collect();
        bytes
    }
}

pub struct Tokenizer {
    pub vocabulary: HashSet<Token>,
}

impl Tokenizer {
    fn from_bytes() -> Tokenizer {
        // init a vocab from the given u8 bytes
        let tokens: HashSet<Token> = (0u8..=255u8).map(Token::from_byte).collect();
        Tokenizer { vocabulary: tokens }
    }

    pub fn from_merges(merge_rules: Vec<((Token, Token), u16)>) -> Tokenizer {
        if merge_rules.is_empty() {
            panic!("You need to train the byte pair encoder before you can get a vocabulary.");
        }

        let mut vocab: Tokenizer = Tokenizer::from_bytes();
        // iterate merge rules in rank order
        for (pair, token_id) in &merge_rules {
            if !vocab.vocabulary.contains(&pair.0) || !vocab.vocabulary.contains(&pair.1) {
                panic!(
                    "Token pair ({:?}, {:?}) not both present in vocabulary when applying merge rule for token_id {}.",
                    pair.0, pair.1, token_id
                );
            }
            vocab.vocabulary.insert(Token::from_pair(token_id, pair));
        }
        vocab
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
}
#[derive(PartialEq, Debug)]
pub struct BytePairEncoder {
    pub merge_rules: Vec<((Token, Token), u16)>, // (pair, merged_id)
}

impl BytePairEncoder {
    fn encode(&self, text: &str) -> Vec<Token> {
        let mut tokens: Vec<Token> = text.bytes().map(Token::from_byte).collect();
        for (pair, token_id) in &self.merge_rules {
            tokens = replace_pair(&tokens, pair, token_id);
        }

        tokens
    }

    fn decode(&self, tokens: &[Token]) -> String {
        let mut target_tokens: Vec<Token> = tokens.to_vec();
        for (pair, token_id) in self.merge_rules.iter().rev() {
            target_tokens = expand_token(target_tokens, pair.clone(), *token_id);
        }
        let bytes = target_tokens
            .iter()
            .map(|t| t.to_utf8_bytes())
            .flatten()
            .collect();
        String::from_utf8(bytes).unwrap()
    }

    pub fn train(corpus: &str, num_merges: u8) -> BytePairEncoder {
        let mut tokens: Vec<Token> = corpus
            .as_bytes()
            .iter()
            .map(|&b| Token::from_byte(b))
            .collect();

        let mut vocab: Tokenizer = Tokenizer::from_bytes();
        let mut merge_rules: Vec<((Token, Token), u16)> = vec![];
        let mut next_token_id: u16 = 256;

        (0..num_merges).for_each(|_| {
            // if not a single pair can be found then exit early
            // e.g. corpus "a"
            if tokens.len() <= 1 {
                return;
            }
            let mut pair_frequencies: HashMap<(Token, Token), i32> = HashMap::new();
            (0..tokens.len() - 1).for_each(|j| {
                let pair = (tokens[j].clone(), tokens[j + 1].clone());
                let count = pair_frequencies.get(&pair).unwrap_or(&0) + 1;
                pair_frequencies.insert(pair, count);
            });

            let most_frequent_pair = pair_frequencies
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| a.0.cmp(b.0)))
                .map(|(k, _)| k.clone())
                .unwrap();

            merge_rules.push((most_frequent_pair.clone(), next_token_id));
            vocab
                .vocabulary
                .insert(Token::from_pair(&next_token_id, &most_frequent_pair));

            tokens = replace_pair(&tokens, &most_frequent_pair, &next_token_id);

            next_token_id += 1;
        });

        BytePairEncoder { merge_rules }
    }
}

fn replace_pair(tokens: &[Token], pair: &(Token, Token), token_id: &u16) -> Vec<Token> {
    let mut result: Vec<Token> = vec![];
    let mut i = 0;
    while i < tokens.len() {
        if i < tokens.len() - 1 && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
            result.push(Token::from_pair(token_id, &pair));
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
            merge_rules: vec![(((Token::from_byte(b'b'), Token::from_byte(b'a')), 256u16))],
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
    let vocab = Tokenizer::from_merges(encoder.merge_rules);

    // Then
    // Vocabulary should contain all 256 base bytes plus 1 merged token
    assert_eq!(vocab.vocabulary.len(), 257);
    assert!(vocab.vocabulary.contains(&Token::from_byte(b'f')));
    assert!(vocab.vocabulary.contains(&Token::new(256, vec![b'b', b'a'])));
}
