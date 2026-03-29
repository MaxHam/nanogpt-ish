use std::io::{self, Write};

use candle_core::{Device, IndexOp, Tensor};
use llm_rs::bigram::Bigram;
use llm_rs::bpe::{TokenTranslation, Tokenizer};
use llm_rs::dataset::Dataset;
use llm_rs::sampling::Generator;
use llm_rs::training::Training;

fn main() -> anyhow::Result<()> {
    let tokenizer = Tokenizer::ascii();
    let device = Device::Cpu;
    // let config = GPTConfig::default(tokenizer.vocabulary.len());
    // let model = Transformer::new(&config)?;
    let mut model = Bigram::new(tokenizer.vocabulary.len(), &device)?;
    let mut dataset = Dataset::from_file("./data/gutenberg_txts/corpus.txt", 0.8, &tokenizer)?;
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
    let _ = model.train(&mut dataset, 128, 64);

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

        let output = model.generate(input, 32)?;
        let decoded = tokenizer.decode(&output.to_tokens(&tokenizer));

        println!("{decoded}\n");
    }
}
