use std::io::{self, Write};

use llm_rs::gpt::{GPTConfig, Transformer};
use llm_rs::bpe::{Tokenizer};

fn main() -> anyhow::Result<()> {
    let tokenizer = match Tokenizer::train("", 257) {
        Ok(tok) => tok,
        Err(e) => {
            eprintln!("Failed to train tokenizer: {}", e);
            return Ok(());
        }
    };
    let config = GPTConfig::default(tokenizer.vocabulary.len());
    let model = Transformer::new(&config)?;

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

        let output = model.generate(&tokenizer, input, 20)?;

        println!("{output}\n");
    }
}