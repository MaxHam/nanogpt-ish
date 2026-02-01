mod bpe;
use std::{fs::read_to_string, path::Path};

use bpe::{Tokenizer};

fn main() {
    let path = Path::new("gutenberg_txts/corpus.txt");
    let corpus = read_to_string(path).expect("Failed to read corpus file");
    let tokenizer = Tokenizer::train(corpus.as_str(), 1000);
    tokenizer.unwrap().to_json()
}
