use std::{fs::File, io::Read};

use candle_core::{Device, IndexOp, Result, Tensor};
use rand::{RngExt, rngs::ThreadRng};

use crate::bpe::{TokenTranslation, Tokenizer};

fn filter_ascii(input: &str) -> String {
    input
        .bytes()                  // iterate over bytes
        .filter(|&b| b.is_ascii() && b <= 128) // keep only ASCII <= 128
        .map(|b| b as char)       // convert back to char
        .collect()
}

#[derive(Debug)]
pub struct Dataset {
    pub training_data: Tensor,
    pub training_size: usize,
    pub validation_data: Tensor,
    pub validation_size: usize,
    device: Device,
    rng: ThreadRng,
}

impl Dataset {
    /// Create a dataset from a flat 1D tensor (sequence of token IDs)
    pub fn new(data: Tensor, training_ratio: f64, device: &Device) -> Self {
        // Ensure tensor is rank-1
        assert_eq!(data.rank(), 1, "Dataset tensor must be rank-1");

        let seq_len = *data.shape().dims().first().unwrap();
        let training_size = (seq_len as f64 * training_ratio) as usize;
        let training_data = data.i(0..training_size).unwrap();

        let validation_size = seq_len - training_size;
        let validation_data = data.i(training_size..seq_len).unwrap();

        let rng = rand::rng();

        Self {
            training_data,
            training_size,
            validation_data,
            validation_size,
            device: device.clone(),
            rng,
        }
    }

    /// Load a dataset from a text file using a tokenizer
    pub fn from_file(path: &str, training_ratio: f64, tokenizer: &Tokenizer, device: &Device) -> Result<Self> {
        let mut file = File::open(path)
            .map_err(|e| candle_core::Error::msg(format!("Failed to open file: {}", e)))?;

        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        // If we are using `Tokenizer::ascii()`, it can't represent bytes > 127.
        // For byte-level BPE tokenizers, we keep the full text.
        let contents = if tokenizer.is_ascii() {
            filter_ascii(contents.as_str())
        } else {
            contents
        };
        let tokens = tokenizer.encode(&contents);

        let data = Tensor::from_tokens(&tokens, &Device::Cpu)?;
        Ok(Dataset::new(data, training_ratio, device))
    }

    /// Sample a random training batch
    pub fn random_training_batch(
        &mut self,
        block_size: usize,
        batch_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        let max_indices: Vec<usize> = (0..batch_size)
            .map(|_| self.rng.random_range(0..self.training_size - block_size))
            .collect();

        // Context: slice of size `block_size`
        let contexts = max_indices.iter().map(|&idx| {
            self.training_data.i(idx..idx + block_size).unwrap()
        });
        let stacked_contexts = Tensor::stack(&contexts.collect::<Vec<_>>(), 0)?
            .to_device(&self.device)?;

        // Targets: same slice shifted by 1
        let targets = max_indices.iter().map(|&idx| {
            self.training_data.i(idx + 1..idx + block_size + 1).unwrap()
        });
        let stacked_targets = Tensor::stack(&targets.collect::<Vec<_>>(), 0)?
            .to_device(&self.device)?;

        Ok((stacked_contexts, stacked_targets))
    }
    pub fn validation_batch(
        &self,
        block_size: usize,
        batch_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        // evenly spaced indices across validation set
        let step = (self.validation_size - block_size) / batch_size;
    
        let indices: Vec<usize> = (0..batch_size)
            .map(|i| i * step)
            .collect();
    
        let contexts = indices.iter().map(|&idx| {
            self.validation_data.i(idx..idx + block_size).unwrap()
        });
        let stacked_contexts = Tensor::stack(&contexts.collect::<Vec<_>>(), 0)?
            .to_device(&self.device)?;
    
        let targets = indices.iter().map(|&idx| {
            self.validation_data.i(idx + 1..idx + block_size + 1).unwrap()
        });
        let stacked_targets = Tensor::stack(&targets.collect::<Vec<_>>(), 0)?
            .to_device(&self.device)?;
    
        Ok((stacked_contexts, stacked_targets))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor, Result};
    use crate::bpe::Tokenizer;

    #[test]
    fn test_dataset_new_and_random_batch() -> Result<()> {
        // Given
        let tokens: Vec<u32> = (0..10).collect();
        let tensor = Tensor::from_slice(&tokens, &[tokens.len()], &Device::Cpu)?;
        let training_ratio = 0.6;

        // When
        let mut dataset = Dataset::new(tensor, training_ratio, &Device::Cpu);

        // Then
        assert_eq!(dataset.training_size, 6);
        assert_eq!(dataset.validation_size, 4);
        assert_eq!(dataset.training_data.shape().dims(), &[6]);
        assert_eq!(dataset.validation_data.shape().dims(), &[4]);

        // When
        let block_size = 2;
        let batch_size = 3;
        let (contexts, targets) = dataset.random_training_batch(block_size, batch_size)?;

        // Then
        assert_eq!(contexts.shape().dims(), &[batch_size, block_size]);
        assert_eq!(targets.shape().dims(), &[batch_size, block_size]);

        Ok(())
    }

    #[test]
    fn test_dataset_from_file_ascii() -> Result<()> {
        // Given
        let tokenizer = Tokenizer::ascii();
        let contents = "Hello, World!";
        let tokens = tokenizer.encode(contents);
        println!("tokens = {:?}", tokens);
        let tensor = Tensor::from_tokens(&tokens, &Device::Cpu)?;
        println!("data shape = {:?}", tensor.shape().dims());
        let training_ratio = 0.8;

        // When
        let dataset = Dataset::new(tensor, training_ratio, &Device::Cpu);

        // Then
        let expected_training_size = (tokens.len() as f64 * training_ratio) as usize;
        let expected_validation_size = tokens.len() - expected_training_size;

        assert_eq!(dataset.training_size, expected_training_size);
        assert_eq!(dataset.validation_size, expected_validation_size);
        assert_eq!(dataset.training_data.shape().dims(), &[expected_training_size]);
        assert_eq!(dataset.validation_data.shape().dims(), &[expected_validation_size]);

        Ok(())
    }
}