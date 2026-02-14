use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};

use crate::bpe::{Token, Tokenizer};

#[derive(Clone)]
pub struct GPTConfig {
    vocab_size: usize,
    n_embd: usize,
    device: Device,
}

pub struct Transformer<'a> {
    pub config: &'a GPTConfig,
    tok_emb: Embedding,
    lm_head: Linear
}

trait CustomTokenizer {
    fn from_tokens(tokens: Vec<Token>, device: &Device) -> Result<Tensor>;
    fn most_likely_token(&self, tokenizer: &Tokenizer) -> Token;
}

impl CustomTokenizer for Tensor {
    fn from_tokens(tokens: Vec<Token>, device: &Device) -> Result<Tensor> {
        Tensor::from_vec(
            tokens.iter().map(|t| t.id as u32).collect::<Vec<u32>>(),
            (1, tokens.len()),
            &device,
        )
    }
    fn most_likely_token(&self, tokenizer: &Tokenizer) -> Token {
        // Converts a Tensor of shape (1, N) (as created by from_tokens) back into a Vec<Token>
        // Assumes the original token IDs are stored as u32 in the tensor.
        // Since Token struct only has an id, we'll reconstruct tokens from the IDs.
        let ids = self.to_vec1::<u32>().unwrap_or_default();
        // Find the index of the highest value in the tensor (assumes shape (1, N)).
        // If the tensor is empty, return a default token (or panic).
        if ids.is_empty() {
            panic!("Tensor is empty, cannot determine most likely token");
        }
        let (_, &max_id) = ids
            .iter()
            .enumerate()
            .max_by_key(|&(_, val)| val)
            .expect("Failed to find max in ids");
        // Map back to Token
        let token = tokenizer
            .vocabulary
            .iter()
            .find(|t| t.id == max_id as u16)
            .cloned()
            .expect("Token id not found in vocabulary");
        token
    }
}

impl<'a> Transformer<'a> {
    pub fn new(config: &'a GPTConfig) -> Result<Self> {
        // random initial weights
        let device = config.device.clone();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let tok_emb = candle_nn::embedding(config.vocab_size, config.n_embd, vb.pp("tok_emb"))?;
        // weight tying, we reuse the token embedding for the lm_head
        let lm_head = Linear::new(tok_emb.embeddings().clone(), None);

        Ok(Self { config, tok_emb, lm_head})
    }

    fn forward(&self, idx: &Tensor) -> Result<Tensor> {
        let x = self.tok_emb.forward(idx)?;
        let (batch, tokens, embeddings) = x.dims3()?;
        let x2d = x.reshape((batch * tokens, embeddings))?;
        // (B*T, C)

        let logits2d = self.lm_head.forward(&x2d)?;
        // (B*T, V)

        let logits = logits2d.reshape((batch, tokens, self.config.vocab_size))?;
        // (B, T, V)

        Ok(logits)
    }
}

#[test]
fn test_tok_emb_tieing() {
    // Given
    let config = GPTConfig {
        device: Device::Cpu,
        vocab_size: 64,
        n_embd: 32,
    };
    let input = Tensor::from_vec(vec![1u32, 5, 42, 9], (1, 4), &config.device).unwrap();
    // When
    let model = Transformer::new(&config).unwrap();
    let output = model.forward(&input).unwrap();

    // Then the shape should be (batch_size, sequence_length, vocab_size):
    // [
    //   token 0 logits over 64 vocab items,
    //   token 1 logits over 64 vocab items,
    //   token 2 logits over 64 vocab items,
    //   token 3 logits over 64 vocab items
    // ]
    // shape = (1, 4, 64)
    // Meaning:
    // output[batch_index][token_position][vocab_index] = logit for that token
    let shape = output.shape();
    assert_eq!(shape.dims(), &[1, 4, config.vocab_size]);
}

// #[test]
// fn test_with_tokenizer() {
//     // Given
//     let tokenizer = Tokenizer::train("", 257).unwrap();
//     let config = &GPTConfig {
//         vocab_size: tokenizer.vocabulary.len(),
//         n_embd: 32,
//         device: Device::Cpu,
//     };
//     let gpt = Transformer::new(config).unwrap();
//     let tokens = tokenizer.encode("foo bar");
//     let input = Tensor::from_tokens(tokens, &config.device).unwrap();

//     // When
//     let output = gpt.forward(&input).unwrap();

//     // Then
//     let output_tokens = output.most_likely_token(&tokenizer);
//     debug_assert_eq!(tokenizer.decode(&vec![output_tokens]), "foo bar")
// }
