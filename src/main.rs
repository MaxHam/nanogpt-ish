mod bpe;
use std::{fs::read_to_string, path::Path};

use bpe::BytePairEncoder;

fn main() {
    let path = Path::new("gutenberg_txts/corpus.txt");
    let corpus = read_to_string(path).expect("Failed to read corpus file");
    let bpe = BytePairEncoder::train(corpus.as_str(), 100);
    let vocab = bpe.vocabulary();

    // Map token ids to decoded UTF-8 strings for readable JSON
    let vocab_strings: std::collections::HashMap<u16, String> = vocab
        .into_iter()
        .map(|(id, bytes)| (id, String::from_utf8_lossy(&bytes).into_owned()))
        .collect();

    let json =
        serde_json::to_string_pretty(&vocab_strings).expect("Failed to serialize vocab");
    std::fs::write("vocab.json", json).expect("Failed to write vocab.json");
}
