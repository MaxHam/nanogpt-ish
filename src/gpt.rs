use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, ops::softmax};

use crate::bpe::{Token, Tokenizer};

#[derive(Clone)]
pub struct GPTConfig {
    vocab_size: usize,
    max_seq_len: usize,
    n_embd: usize,
    device: Device,
}

impl GPTConfig {
    pub fn default(vocab_size: usize) -> GPTConfig {
        GPTConfig {
            vocab_size,
            max_seq_len: 512,
            n_embd: 32,
            device: Device::Cpu,
        }
    }
}

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

        let neg_inf = Tensor::full(f32::NEG_INFINITY, (b, t, t), device)?;

        let zero_mask = (mask.eq(0)?).to_dtype(DType::F32)?;
        scores = ((scores * mask)? + (neg_inf * zero_mask)?)?;

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

pub struct Transformer<'a> {
    pub config: &'a GPTConfig,
    tok_emb: Embedding,
    pos_emb: Embedding,
    block: Block,
    lm_head: Linear,
}

trait TokenTranslation {
    fn from_tokens(tokens: &[Token], device: &Device) -> Result<Tensor>;
}

impl TokenTranslation for Tensor {
    fn from_tokens(tokens: &[Token], device: &Device) -> Result<Tensor> {
        Tensor::from_vec(
            tokens.iter().map(|t| t.id as u32).collect::<Vec<u32>>(),
            (1, tokens.len()),
            device,
        )
    }
}

impl<'a> Transformer<'a> {
    pub fn new(config: &'a GPTConfig) -> Result<Self> {
        // random initial weights
        let device = &config.device;

        // Zero-mean init: avoids bias toward any token (e.g. space) with weight tying
        let tok_emb_weights =
            Tensor::randn(0.0, 0.02f32, (config.vocab_size, config.n_embd), device)?;
        let tok_emb = Embedding::new(tok_emb_weights, config.n_embd);

        let pos_emb_weights =
            Tensor::randn(0.0, 0.02f32, (config.max_seq_len, config.n_embd), device)?;
        let pos_emb = Embedding::new(pos_emb_weights, config.n_embd);

        let block = Block::new(32, device)?;

        // weight tying, we reuse the token embedding for the lm_head
        // TODO: find out whether we can reuse the actual tensor and not just clone it, for efficiency sake
        let lm_head = Linear::new(tok_emb.embeddings().clone(), None);

        Ok(Self {
            config,
            tok_emb,
            pos_emb,
            block,
            lm_head,
        })
    }

    fn input_embedding(&self, idx: &Tensor) -> Result<Tensor> {
        // This method takes care of adding up the token and positional embeddings
        // representation = meaning(token emb) + location(position emb)
        //
        // I moved this part of the forward step here only to not clutter the main loop
        let (_batch, seq_len) = idx.dims2()?;
        assert!(
            seq_len <= self.config.max_seq_len,
            "sequence length exceeds max sequence length of {}",
            self.config.max_seq_len
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
        let x = x.reshape((batch * seq_len, self.config.n_embd))?;
        let logits = self.lm_head.forward(&x)?;
        logits.reshape((batch, seq_len, self.config.vocab_size))
    }

    pub fn generate(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<String> {
        let device = &self.config.device;
        let mut tokens = tokenizer.encode(prompt);
        let mut input = Tensor::from_tokens(&tokens, device)?;

        for _ in 0..max_new_tokens {
            let logits = self.forward(&input)?;

            let (_, seq_len, _) = logits.dims3()?;

            let last_logits = logits.i((0, seq_len - 1))?;

            // greed sampling, equals temperature=0
            let next_id = last_logits.argmax(0)?.to_scalar::<u32>()?;
            // Convert id to token
            let next_token = tokenizer
                .vocabulary
                .get(&(next_id as u16))
                .cloned()
                .expect("Token not found");

            // Add generated token to input and rebuild tensor
            tokens.push(next_token);
            input = Tensor::from_tokens(&tokens, device)?;
        }

        Ok(tokenizer.decode(&tokens))
    }
}

#[test]
fn test_tok_emb_tieing() {
    // Given
    let config = GPTConfig::default(64);
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

#[test]
fn test_generate() {
    let tokenizer = Tokenizer::from_bytes();
    let config = GPTConfig::default(tokenizer.vocabulary.len());
    let prompt = "Hi";
    // When
    let model = Transformer::new(&config).unwrap();
    let output = model.generate(&tokenizer, &prompt, 1).unwrap();

    // Then
    debug_assert!(
        !output.is_empty(),
        "Input prompt string should not be empty"
    );
}

#[test]
fn test_token_to_tensor() {
    // Given
    let tokens = vec![
        Token::from_byte(0),
        Token::from_byte(1),
        Token::from_byte(2),
    ];

    // When
    let input = Tensor::from_tokens(&tokens, &Device::Cpu).unwrap();

    // Then it should create 7 tokens since no merges happened
    let num_tokens = input.shape().dims();
    assert_eq!(num_tokens, &[1, 3]);
}
