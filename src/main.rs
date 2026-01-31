mod bpe;
use std::{fs::read_to_string, path::Path};

use bpe::{BytePairEncoder, Tokenizer};

fn main() {
    let path = Path::new("gutenberg_txts/corpus.txt");
    let corpus = read_to_string(path).expect("Failed to read corpus file");
    let bpe = BytePairEncoder::train(corpus.as_str(), 100);
    let vocab = Tokenizer::from_merges(bpe.merge_rules);
    vocab.to_json()
}
