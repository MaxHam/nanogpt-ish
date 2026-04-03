use candle_core::{DType, Device, IndexOp, Result, Shape, Tensor};
use candle_nn::{
    AdamW, Embedding, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap, loss,
    ops::softmax,
};
use rand::{
    rngs::ThreadRng,
};

use crate::{dataset::Dataset, sampling::{Generator, sample_multinomial}};
use crate::{training::Training};


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
    vocab_size: usize,
    rng: ThreadRng,
    var_map: VarMap,
}

impl Bigram {
    pub fn new(vocab_size: usize, device: &Device) -> Result<Self> {
        let mut var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&mut var_map, DType::F32, device);
        let embeddings = vb.get((vocab_size, vocab_size), "embeddings")?;
        let tok_emb = Embedding::new(embeddings, vocab_size);

        Ok(Self {
            tok_emb,
            vocab_size,
            rng: rand::rng(),
            var_map,
        })
    }


}

impl Training for Bigram {
    fn train(&self, dataset: &mut Dataset, num_epochs: usize, batch_size: usize) -> Result<()> {
        let params = ParamsAdamW {
            lr: 0.1, // set extra high so we can result fast in this toy example
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        };
        let mut optimizer = AdamW::new(self.var_map.all_vars(), params)?;

        for epoch in 0..num_epochs {
            let (training_inputs, training_targets) =
                dataset.random_training_batch(self.vocab_size, batch_size)?;
            let logits = self.forward(&training_inputs)?;
            let (batch_size, time_size, channel_size) = logits.shape().dims3()?;
            let loss = loss::cross_entropy(
                &logits.reshape(Shape::from((batch_size * time_size, channel_size)))?,
                &training_targets.reshape(Shape::from((batch_size * time_size,)))?,
            )?;
            optimizer.backward_step(&loss)?;

            println!(
                "Epoch: {epoch:3} Train loss: {:8.5}",
                loss.to_scalar::<f32>()?
            );
        }
        Ok(())
    }
}

impl Generator for Bigram {
    fn generate(
        &mut self,
        mut idx: Tensor,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> Result<Tensor> {
        // Takes in shape (batch, sequence)
        // Returns in shape (batch, sequence)
        // input tensor is updated in place
        for _ in 0..max_new_tokens {
            let logits = self.forward(&idx)?; // (batch, seq_len, vocab)
            let (_, seq_len, _) = logits.dims3()?;
            let last_logits = logits.i((.., seq_len - 1, ..))?; // (batch, vocab)

            // REPL generator is tuned for `batch=1`.
            let last_logits = last_logits.squeeze(0)?; // (vocab)
            let temp = temperature.max(1e-5);
            let temp_tensor = Tensor::new(temp, last_logits.device())?;
            let scaled_logits = last_logits.broadcast_div(&temp_tensor)?;
            let vocab = scaled_logits.dim(0)?;

            let next_token = if top_k > 0 && top_k < vocab {
                let topk_indices = scaled_logits
                    .arg_sort_last_dim(false)?
                    .narrow(0, 0, top_k)?
                    .contiguous()?;
                let topk_logits = scaled_logits.gather(&topk_indices, 0)?;
                let probs = softmax(&topk_logits, 0)?;
                let probs_vec: Vec<f32> = probs.to_vec1()?;
                let sampled_in_topk = sample_multinomial(&mut self.rng, &probs_vec)? as usize;
                let topk_idx_vec: Vec<u32> = topk_indices.to_vec1::<u32>()?;
                topk_idx_vec[sampled_in_topk]
            } else {
                let probs = softmax(&scaled_logits, 0)?;
                let probs_vec: Vec<f32> = probs.to_vec1()?;
                sample_multinomial(&mut self.rng, &probs_vec)?
            };

            // reshape to [1,1]
            let next_tensor = Tensor::from_slice(&[next_token], &[1, 1], &idx.device())?;
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

    #[test]
    fn test_forward_shape() -> Result<()> {
        // Given
        let device = Device::Cpu;
        let vocab_size = 5;
        let model = Bigram::new(vocab_size, &device)?;
        let idx = Tensor::from_slice(&[0u32, 1, 2], &[1, 3], &device)?; // seq_len=3

        // When
        let logits = model.forward(&idx)?;

        // Then
        assert_eq!(logits.shape().dims(), &[1, 3, vocab_size]);

        Ok(())
    }

    #[test]
    fn test_generate_shape() -> Result<()> {
        // Given
        let device = Device::Cpu;
        let vocab_size = 5;
        let mut model = Bigram::new(vocab_size, &device)?;
        let max_new_tokens = 1;
        let seq_len = 3;
        let idx = Tensor::from_slice(&[0u32, 1, 2], &[1, 3], &device)?; // seq_len=3

        // When
        let output_idx = model.generate(idx, 1, 1.0, 0)?;

        // Then
        assert_eq!(output_idx.shape().dims(), &[1, seq_len+max_new_tokens]); // (batch, seq_len)

        Ok(())
    }
}
