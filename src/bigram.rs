use candle_core::{Device, Error, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module, ops};
use rand::{distr::{Distribution, weighted::WeightedIndex}, rngs::ThreadRng};

pub fn sample_multinomial(rng: &mut ThreadRng, prs: &Vec<f32>) -> candle_core::Result<u32> {
    let distribution = WeightedIndex::new(prs).map_err(Error::wrap)?;
    let next_token = distribution.sample(rng) as u32;

    Ok(next_token)
}

/// A simple Bigram language model.
///
/// This model stores a bigram probability table of shape `[vocab_size, vocab_size]`,
/// where each row corresponds to the probability distribution of the next token
/// given the current token.
///
/// In a bigram model, the prediction of the next token depends **only on the current token**,
/// ignoring any earlier context. This is a simple form of a Markov model for sequences.
pub struct Bigram {
    tok_emb: Embedding,
    rng: ThreadRng
}

impl Bigram {
    pub fn new(vocab_size: usize, device: &Device) -> Result<Self> {
        let tok_emb_weights = Tensor::randn(0.0, 0.02f32, (vocab_size, vocab_size), &device)?;
        let tok_emb = Embedding::new(tok_emb_weights, vocab_size);
        let mut rng = rand::rng();
        Ok(Self { tok_emb, rng})
    }

    pub fn generate(&mut self, mut idx: Tensor, max_new_tokens: usize) -> Result<Tensor> {
        for _ in 0..max_new_tokens {
            let logits = self.forward(&idx)?;
            let (_, seq_len, _) = logits.dims3()?;
            let last_logits = logits.i((.., seq_len - 1, ..))?;
            let probabilities = ops::softmax(&last_logits, 0)?;
            let probabilities = probabilities.squeeze(0)?;     
            let probs_vec = probabilities.to_vec1()?;
            let next_token = sample_multinomial(&mut self.rng, &probs_vec)?;
            // reshape to [1,1]
            let next_tensor = Tensor::from_slice(&[next_token], &[1, 1], &Device::Cpu)?; 
            // append to sequence
            idx = Tensor::cat(&[&idx, &next_tensor], 1)?;
        }
        Ok(idx)
    }
}

impl Module for Bigram {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.tok_emb.forward(xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Result, Tensor};
    use candle_nn::Embedding;

    #[test]
    fn test_forward_shape() -> Result<()> {
        // Given
        let device = Device::Cpu;
        let vocab_size = 5;
        let emb_weights = Tensor::randn(0.0f32, 0.02, (vocab_size, vocab_size), &device)?;
        let tok_emb = Embedding::new(emb_weights, vocab_size);
        let model = Bigram { tok_emb, rng: rand::rng() };
        let idx = Tensor::from_slice(&[0u32, 1, 2], &[3], &device)?; // seq_len=3

        // When
        let logits = model.forward(&idx)?;

        // Then
        assert_eq!(logits.shape().dims(), &[3, vocab_size]);

        Ok(())
    }
}
