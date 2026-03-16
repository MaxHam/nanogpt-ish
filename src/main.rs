use std::io::{self, Write};

use candle_core::{Device, Tensor};
use llm_rs::gpt::{GPTConfig, Transformer};
use llm_rs::bigram::{Bigram};
use llm_rs::bpe::{TokenTranslation, Tokenizer};

fn main() -> anyhow::Result<()> {
    let tokenizer = Tokenizer::ascii();
    let device = Device::Cpu;
    // let config = GPTConfig::default(tokenizer.vocabulary.len());
    // let model = Transformer::new(&config)?;
    let mut model= Bigram::new(tokenizer.vocabulary.len(), &device)?;

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
        let encoded  =  tokenizer.encode(input);
        let mut input = Tensor::from_tokens(&encoded, &device)?;
        let output = model.generate(input, 20)?;
        let decoded = tokenizer.decode(&output.to_tokens(&tokenizer));
        

        println!("{decoded}\n");
    }
}