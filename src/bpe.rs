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

    pub fn to_utf8_bytes(&self) -> Vec<Utf8Byte> {
        let bytes: Vec<Utf8Byte> = self.value.iter().map(|&t| t as Utf8Byte).collect();
        bytes
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
        // initial tokenizer vocab has 256 ids
        let mut next_token_id: u16 = 256;
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
                .insert(Token::from_pair(&next_token_id, pair));
            next_token_id += 1;
        }
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
        let bytes = target_tokens
            .iter()
            .flat_map(|t| t.to_utf8_bytes())
            .collect();
        String::from_utf8(bytes).unwrap()
    }
}
#[derive(PartialEq, Debug)]
pub struct BytePairEncoder {
    pub merge_rules: Vec<((Token, Token), u16)>,
    // (pair, merged_id), the rank of the merge rule is defind by its pos in the vector
}

impl BytePairEncoder {
    pub fn train(corpus: &str, num_merges: u8) -> Tokenizer {
        let corpus_tokens: Vec<Token> = corpus
            .as_bytes()
            .iter()
            .map(|&b| Token::from_byte(b))
            .collect();

        let mut merge_rules: Vec<((Token, Token), u16)> = vec![];
        let mut next_token_id: u16 = 256;

        for (i, _) in (0..num_merges).enumerate() {
            eprintln!("Merge {}/{}", i + 1, num_merges);

            // if not a single pair can be found then exit early
            // e.g. corpus "a"
            if corpus_tokens.len() <= 1 {
                break;
            }
            let mut pair_frequencies: HashMap<(Token, Token), i32> = HashMap::new();
            (0..corpus_tokens.len() - 1).for_each(|j| {
                let pair = (corpus_tokens[j].clone(), corpus_tokens[j + 1].clone());
                let count = pair_frequencies.get(&pair).unwrap_or(&0) + 1;
                pair_frequencies.insert(pair, count);
            });

            let most_frequent_pair = pair_frequencies
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| a.0.cmp(b.0)))
                .map(|(k, _)| k.clone())
                .unwrap();

            merge_rules.push((most_frequent_pair.clone(), next_token_id));
            next_token_id += 1;
        }

        if merge_rules.is_empty() {
            Tokenizer::from_bytes()
        } else {
            Tokenizer::from_merges(merge_rules)
        }
    }
}

fn replace_pair(tokens: &[Token], pair: &(Token, Token), token_id: &u16) -> Vec<Token> {
    let mut result: Vec<Token> = vec![];
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
    tokens.to_vec()
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
        let tokenizer = BytePairEncoder::train(corpus, 1);

        // Then
        let merged_token =
            Token::from_pair(&256, &(Token::from_byte(b'b'), Token::from_byte(b'a')));
        assert!(tokenizer.vocabulary.contains(&merged_token));
    }

    #[test]
    fn test_train_single_byte_corpus() {
        // Given
        let corpus = "a";

        // When
        let tokenizer = BytePairEncoder::train(corpus, 1);

        // Then
        assert_eq!(tokenizer, Tokenizer::from_bytes());
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

        // When
        let tokenizer = BytePairEncoder::train(corpus, 1);

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
        let tokenizer = BytePairEncoder::train(corpus.as_str(), 3);

        // When
        let encoded = tokenizer.encode("Hi World!");
        let decoded = tokenizer.decode(&encoded);

        // Then
        assert_eq!(decoded, "Hi World!");
    }
}
