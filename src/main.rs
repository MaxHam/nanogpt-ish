use std::io::{self, Write};

use candle_core::{Device, IndexOp, Tensor};
use llm_rs::bpe::{TokenTranslation, Tokenizer};
use llm_rs::dataset::Dataset;
use llm_rs::gpt::Transformer;
use llm_rs::sampling::Generator;
use llm_rs::training::Training;

fn main() -> anyhow::Result<()> {
    let corpus = std::fs::read_to_string("./data/shakespeare.txt")?;
    // Small vocab + small model so we can train on CPU quickly.
    // Byte-level BPE generally improves "speaking" a lot vs ASCII tokens.
    let vocab_size = 320u16;
    let tokenizer = Tokenizer::train(&corpus, vocab_size).expect("failed to train tokenizer");
    let device = Device::Cpu;
    let mut model = Transformer::new(tokenizer.vocabulary.len(), &device, 64, 32)?;
    let mut dataset = Dataset::from_file("./data/shakespeare.txt", 0.9, &tokenizer)?;
    println!(
        "Training data shape: {:?}, dtype: {:?}",
        dataset.training_data.shape(),
        dataset.training_data.dtype()
    );
    println!(
        "Validation data shape: {:?}, dtype: {:?}",
        dataset.validation_data.shape(),
        dataset.validation_data.dtype()
    );

    let block_size = 8usize;
    println!(
        "First block of training data: {:?}, decoded {:?}",
        &dataset.training_data.i(0..block_size).unwrap(),
        tokenizer.decode(&dataset.training_data.i(0..block_size).unwrap().to_tokens(&tokenizer))
    );
    model.train(&mut dataset, 1024, 16)?;

    println!("Chitchat with your GPT");
    println!("Type something and press enter. Ctrl+C to exit.\n");

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        let encoded = tokenizer.encode(input);
        let mut input = Tensor::from_tokens(&encoded, &device)?;
        input = input.unsqueeze(0)?; // add batch back in (1, S)

        let output = model.generate(input, 32, 0.9, 40)?;
        let decoded = tokenizer.decode(&output.to_tokens(&tokenizer));

        println!("{decoded}\n");
    }
}
