use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder, VarMap, ops::{self, softmax}};
use rand::rngs::ThreadRng;

use crate::sampling::{sample_multinomial, Generator};

#[derive(Clone)]
struct SelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
}

impl SelfAttention {
    fn new(n_embd: usize, device: Device) -> Result<Self> {
        Ok(SelfAttention {
            q_proj: Linear::new(
                Tensor::randn(0.5, 0.25f32, (n_embd, n_embd), &device)?,
                None,
            ),
            k_proj: Linear::new(
                Tensor::randn(0.5, 0.25f32, (n_embd, n_embd), &device)?,
                None,
            ),
            v_proj: Linear::new(
                Tensor::randn(0.5, 0.25f32, (n_embd, n_embd), &device)?,
                None,
            ),
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
        let mask = mask.unsqueeze(0)?; // (t, t) -> (1, t, t) to match scores (b, t, t)

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
    attention: SelfAttention,
    mlp: Linear
}

impl Block {
    fn new(n_embd: usize, device: &Device) -> Result<Self> {
        let ln1_weights = Tensor::rand(0.0, 1.0f32, (n_embd,), device)?;
        let ln2_weights = Tensor::rand(0.0, 1.0f32, (n_embd,), device)?;
        let mlp_weights = Tensor::randn(0.0, 0.02f32, (n_embd, n_embd), device)?;
        Ok(Block {
            ln1: LayerNorm::new_no_bias(ln1_weights, 0.01),
            ln2: LayerNorm::new_no_bias(ln2_weights, 0.01),
            attention: SelfAttention::new(n_embd, device.clone())?,
            mlp: Linear::new(mlp_weights, None)
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_norm = self.ln1.forward(x)?;

        let attn_out = self.attention.forward(&x_norm)?;

        let x = (x + attn_out)?;

        let x_norm = self.ln2.forward(&x)?;
        let mlp_out = self.mlp.forward(&x_norm)?;
        let output = (x + mlp_out)?;

        Ok(output)
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
}

impl Transformer {
    pub fn new(vocab_size: usize, device: &Device, max_seq_len: usize, n_emb: usize) -> Result<Self> {
        let mut var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&mut var_map, DType::F32, device);

        // Zero-mean init: avoids bias toward any token (e.g. space) with weight tying
        let tok_emb_weights = vb.get((vocab_size, n_emb), "tok_emb")?;
        let pos_emb_weights  = vb.get((max_seq_len, n_emb), "pos_emb")?;
        let tok_emb = Embedding::new(tok_emb_weights, n_emb);
        let pos_emb = Embedding::new(pos_emb_weights, n_emb);

        let block = Block::new(32, device)?;

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
        })
    }

    fn default(device: &Device)->Result<Self> {
        Transformer::new(
            64,
            device,
            512,
            32,
        )
    }

    fn input_embedding(&self, idx: &Tensor) -> Result<Tensor> {
        // This method takes care of adding up the token and positional embeddings
        // representation = meaning(token emb) + location(position emb)
        //
        // I moved this part of the forward step here only to not clutter the main loop
        let (_batch, seq_len) = idx.dims2()?;
        assert!(
            seq_len <= self.max_seq_len,
            "sequence length exceeds max sequence length of {}",
            self.max_seq_len
        );

        let tok = self.tok_emb.forward(idx)?;
        let pos_idx = Tensor::arange(0u32, seq_len as u32, idx.device())?.unsqueeze(0)?;
        let pos = self.pos_emb.forward(&pos_idx)?;

        let x = (tok + pos)?;
        // Keep (batch, seq_len, n_embd) for attention (needs 3D for Q@K^T)
        Ok(x)
    }

    fn forward(&self, idx: &Tensor) -> Result<Tensor> {
        let x = self.input_embedding(idx)?;
        // (B, T, C)

        let x = self.block.forward(&x)?;

        let (batch, seq_len, _) = x.dims3()?;
        let x = x.reshape((batch * seq_len, self.n_emb))?;
        let logits = self.lm_head.forward(&x)?;
        logits.reshape((batch, seq_len, self.vocab_size))
    }
}
impl Generator for Transformer {
    fn generate(&mut self, mut idx: Tensor, max_new_tokens: usize) -> Result<Tensor> {
        // Takes in shape (batch, sequence)
        // Returns in shape (batch, sequence)
        // input tensor is updated in place
        for _ in 0..max_new_tokens {
            let logits = self.forward(&idx)?; // (batch, seq_len, vocab)
            let (_, seq_len, _) = logits.dims3()?;
            let last_logits = logits.i((.., seq_len - 1, ..))?; // (batch, vocab)
            let probabilities = ops::softmax(&last_logits, 1)?; // (128)
            let probabilities = probabilities.squeeze(0)?;
            let probs_vec = probabilities.to_vec1()?;
            let next_token = sample_multinomial(&mut self.rng, &probs_vec)?;
            // reshape to [1,1]
            let next_tensor = Tensor::from_slice(&[next_token], &[1, 1], &idx.device())?;
            idx = Tensor::cat(&[&idx, &next_tensor], 1)?;
        }
        Ok(idx)
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
        let output_idx = model.generate(idx, 1)?;

        // Then
        assert_eq!(output_idx.shape().dims(), &[1, seq_len+max_new_tokens]); // (batch, seq_len)

        Ok(())
    }