use candle_core::{DType, Device, IndexOp, Result, Shape, Tensor};
use candle_nn::{
    AdamW, Embedding, Init, LayerNorm, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap,
    layer_norm_no_bias, linear_no_bias, loss, ops::softmax,
};
use rand::rngs::ThreadRng;

use crate::{
    dataset::Dataset,
    sampling::{Generator, sample_multinomial},
    training::Training,
};

#[derive(Clone)]
struct SelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
}

impl SelfAttention {
    fn new(n_embd: usize, head_size: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(SelfAttention {
            q_proj: linear_no_bias(n_embd, head_size, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(n_embd, head_size, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(n_embd, head_size, vb.pp("v_proj"))?,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _c) = x.dims3()?;

        // Linear projections
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Attention scores: Q @ K^T / sqrt(d) — scale prevents softmax saturation
        let scale = (x.dim(2)? as f64).sqrt();
        let k_t = k.transpose(1, 2)?; // (b, t, c) -> (b, c, t)
        let mut scores = (q.matmul(&k_t)? / scale)?;

        // -------- Causal mask (lower triangular: attend to current and past only) --------
        let device = x.device();
        let row = Tensor::arange(0f32, t as f32, device)?
            .unsqueeze(1)?
            .broadcast_as((t, t))?;
        let col = Tensor::arange(0f32, t as f32, device)?
            .unsqueeze(0)?
            .broadcast_as((t, t))?;
        let mask = row.ge(&col)?.to_dtype(DType::F32)?;
        let mask = mask.unsqueeze(0)?; // (t, t) -> (1, t, t)
        let mask = mask.broadcast_as((b, t, t))?; // explicit broadcast for Candle ops

        // Large negative constant for masked positions. Do not use -inf here:
        // `0 * (-inf)` is NaN (IEEE 754), and zero_mask is 0 on allowed cells, which
        // would poison the whole scores tensor before softmax.
        let neg_large = Tensor::full(-1e9f32, (b, t, t), device)?;

        let zero_mask = (mask.eq(0)?).to_dtype(DType::F32)?;
        scores = ((scores * mask)? + (neg_large * zero_mask)?)?;

        // Softmax over last dim (keys)
        let probs = softmax(&scores, scores.rank() - 1)?;

        // Weighted sum
        let attn_out = probs.matmul(&v)?;

        Ok(attn_out)
    }
}

#[derive(Clone)]
struct Block {
    // Layer Norm normalizes activations across the feature dimension.
    // Values can grow very large and the layer norm forces it back to mean ~0 and variance ~1
    ln1: LayerNorm,
    ln2: LayerNorm,
    attention_heads: Vec<SelfAttention>,
    mlp: Linear,
}

impl Block {
    fn new(n_embd: usize, num_heads: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Block {
            ln1: layer_norm_no_bias(n_embd, 0.01, vb.pp("ln1"))?,
            ln2: layer_norm_no_bias(n_embd, 0.01, vb.pp("ln2"))?,
            attention_heads: vec![
                SelfAttention::new(n_embd, n_embd / num_heads, vb.pp("attn"))?;
                num_heads
            ],
            mlp: linear_no_bias(n_embd, n_embd, vb.pp("mlp"))?,
        })
    }
}

pub struct Transformer {
    tok_emb: Embedding,
    pos_emb: Embedding,
    block: Block,
    lm_head: Linear,
    max_seq_len: usize,
    vocab_size: usize,
    n_emb: usize,
    rng: ThreadRng,
    var_map: VarMap,
}

impl Transformer {
    pub fn new(
        vocab_size: usize,
        device: &Device,
        max_seq_len: usize,
        n_emb: usize,
    ) -> Result<Self> {
        let mut var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&mut var_map, DType::F32, device);

        let emb_init = Init::Randn {
            mean: 0.,
            stdev: 0.02,
        };
        let tok_emb_weights = vb.get_with_hints((vocab_size, n_emb), "tok_emb", emb_init)?;
        let pos_emb_weights = vb.get_with_hints((max_seq_len, n_emb), "pos_emb", emb_init)?;
        let tok_emb = Embedding::new(tok_emb_weights, n_emb);
        let pos_emb = Embedding::new(pos_emb_weights, n_emb);

        let block = Block::new(n_emb, 4, vb.pp("block"))?;

        // weight tying, we reuse the token embedding for the lm_head
        // TODO: find out whether we can reuse the actual tensor and not just clone it, for efficiency sake
        let lm_head = Linear::new(tok_emb.embeddings().clone(), None);

        Ok(Self {
            tok_emb,
            pos_emb,
            block,
            lm_head,
            max_seq_len,
            vocab_size,
            n_emb,
            rng: rand::rng(),
            var_map,
        })
    }

    fn default(device: &Device) -> Result<Self> {
        Transformer::new(64, device, 512, 32)
    }

    fn input_embedding(&self, idx: &Tensor) -> Result<Tensor> {
        // This method takes care of adding up the token and positional embeddings
        // representation = meaning(token emb) + location(position emb)
        //
        // I moved this part of the forward step here only to not clutter the main loop
        let (batch, seq_len) = idx.dims2()?;
        assert!(
            seq_len <= self.max_seq_len,
            "sequence length exceeds max sequence length of {}",
            self.max_seq_len
        );

        let tok = self.tok_emb.forward(idx)?;
        let pos_idx = Tensor::arange(0u32, seq_len as u32, idx.device())?.unsqueeze(0)?;
        let pos = self.pos_emb.forward(&pos_idx)?;
        let pos = pos.broadcast_as((batch, seq_len, self.n_emb))?;

        let x = (tok + pos)?;
        // Keep (batch, seq_len, n_embd) for attention (needs 3D for Q@K^T)
        Ok(x)
    }
}

impl Generator for Transformer {
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
        // Note: this REPL-style generator is tuned for `batch=1`.

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

            // Apply top-k using Candle tensor ops, then sample from the (k) probs.
            let next_token = if top_k > 0 && top_k < vocab {
                // Indices sorted descending by logit.
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
                // No top-k filtering: sample from full softmax.
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

impl Module for Transformer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.input_embedding(xs)?;
        // (B, T, C)

        let x = self.block.forward(&x)?;

        let (batch, seq_len, _) = x.dims3()?;
        let x = x.reshape((batch * seq_len, self.n_emb))?;
        let logits = self.lm_head.forward(&x)?;
        logits.reshape((batch, seq_len, self.vocab_size))
    }
}

impl Module for Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x_norm = self.ln1.forward(xs)?;
        let attention_head_outputs: Vec<Tensor> = self
            .attention_heads
            .iter()
            .map(|h| h.forward(&x_norm))
            .collect::<Result<Vec<_>>>()?;
        let attn_out = Tensor::cat(&attention_head_outputs, 2)?;

        let x = (xs + attn_out)?;

        let x_norm = self.ln2.forward(&x)?;
        let mlp_out = self.mlp.forward(&x_norm)?;
        let output = (x + mlp_out)?;

        Ok(output)
    }
}

impl Training for Transformer {
    fn train(&self, dataset: &mut Dataset, num_epochs: usize, batch_size: usize) -> Result<()> {
        let params = ParamsAdamW {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        };
        let mut optimizer = AdamW::new(self.var_map.all_vars(), params)?;

        for epoch in 0..num_epochs {
            let (training_inputs, training_targets) =
                dataset.random_training_batch(self.max_seq_len, batch_size)?;
            let logits = self.forward(&training_inputs)?;
            let (batch_size, time_size, channel_size) = logits.shape().dims3()?;
            let loss = loss::cross_entropy(
                &logits.reshape(Shape::from((batch_size * time_size, channel_size)))?,
                &training_targets.reshape(Shape::from((batch_size * time_size,)))?,
            )?;
            optimizer.backward_step(&loss)?;

            let (training_inputs, training_targets) =
                dataset.random_training_batch(self.max_seq_len, batch_size)?;
            let logits = self.forward(&training_inputs)?;
            let (batch_size, time_size, channel_size) = logits.shape().dims3()?;
            let loss = loss::cross_entropy(
                &logits.reshape(Shape::from((batch_size * time_size, channel_size)))?,
                &training_targets.reshape(Shape::from((batch_size * time_size,)))?,
            )?;
            println!(
                "Epoch: {epoch:3} Train loss: {:8.5}",
                loss.to_scalar::<f32>()?
            );
        }
        Ok(())
    }
}

#[test]
fn test_tok_emb_tieing() {
    // Given
    let device = Device::Cpu;
    let input = Tensor::from_vec(vec![1u32, 5, 42, 9], (1, 4), &device).unwrap();

    // When
    let model = Transformer::default(&device).unwrap();
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
    assert_eq!(shape.dims(), &[1, 4, model.vocab_size]);
}

#[test]
fn test_generate_shape() -> Result<()> {
    // Given
    let device = Device::Cpu;
    let mut model = Transformer::default(&device).unwrap();
    let max_new_tokens = 1;
    let seq_len = 3;
    let idx = Tensor::from_slice(&[0u32, 1, 2], &[1, 3], &device)?; // seq_len=3

    // When
    let output_idx = model.generate(idx, 1, 1.0, 0)?;

    // Then
    assert_eq!(output_idx.shape().dims(), &[1, seq_len + max_new_tokens]); // (batch, seq_len)

    Ok(())
}

#[test]
fn test_attention_head_shape() -> Result<()> {
    // Given
    let device = Device::Cpu;
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let attn = SelfAttention::new(8, 8, vb).unwrap();
    let idx = Tensor::zeros((1, 32, 8), DType::F32, &device)?;

    // When
    let output_idx = attn.forward(&idx)?;

    // Then input_dim = output_dim
    assert_eq!(*output_idx.shape().dims(), *idx.shape().dims());

    Ok(())
}

#[test]
fn test_block_shape() -> Result<()> {
    // Given
    let device = Device::Cpu;
    let var_map = VarMap::new();
    let n_embd = 8;
    let n_heads = 4;
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let block = Block::new(n_embd, n_heads, vb).unwrap();
    let idx = Tensor::from_slice(&[0f32, 1.0, 2.0], &[1, 32, 8], &device)?; // (B, T, C)

    // When
    let output_idx = block.forward(&idx)?;

    // Then input_dim = output_dim
    assert_eq!(*output_idx.shape().dims(), *idx.shape().dims());

    Ok(())
}
