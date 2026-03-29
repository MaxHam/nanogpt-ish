use candle_core::{Result};
use crate::{dataset::Dataset};

pub trait Training {
    fn train(&self, dataset: &mut Dataset, num_epochs: usize, batch_size: usize) -> Result<()>; 
}