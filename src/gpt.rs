use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Linear, Module};

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
            max_seq_len: 32,
            n_embd: 32,
            device: Device::Cpu,
        }
    }
}

pub struct Transformer<'a> {
    pub config: &'a GPTConfig,
    tok_emb: Embedding,
    pos_emb: Embedding,
    lm_head: Linear,
}

trait CustomTokenizer {
    fn from_tokens(tokens: &[Token], device: &Device) -> Result<Tensor>;
}

impl CustomTokenizer for Tensor {
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

        let tok_emb_weights = Tensor::zeros((config.vocab_size, config.n_embd), DType::F32, device)?;
        let tok_emb = Embedding::new(tok_emb_weights, config.n_embd);

        let pos_emb_weights = Tensor::zeros((config.max_seq_len, config.n_embd), DType::F32, device)?;
        let pos_emb = Embedding::new(pos_emb_weights, config.n_embd);
    
        // weight tying, we reuse the token embedding for the lm_head
        // TODO: find out whether we can reuse the actual tensor and not just clone it, for efficiency sake
        let lm_head = Linear::new(tok_emb.embeddings().clone(), None);

        Ok(Self {
            config,
            tok_emb,
            pos_emb,
            lm_head,
        })
    }

    fn input_embedding(&self, idx: &Tensor) -> Result<Tensor> {
        // This method takes care of adding up the token and positional embeddings
        // representation = meaning(token emb) + location(position emb)
        //
        // I moved this part of the forward step here only to not clutter the main loop
        let (batch, seq_len) = idx.dims2()?; 
        assert!(seq_len <= self.config.max_seq_len, "sequence length exceeds block size");
    
        let tok = self.tok_emb.forward(idx)?;
        let pos_idx = Tensor::arange(0u32, seq_len as u32, idx.device())?.unsqueeze(0)?;
        let pos = self.pos_emb.forward(&pos_idx)?;
        
        let x = (tok + pos)?;
        let x2d = x.reshape((batch * seq_len, self.config.n_embd))?;
    
        Ok(x2d)
    }

    fn forward(&self, idx: &Tensor) -> Result<Tensor> {
        let x2d = self.input_embedding(idx)?;
        // (B*T, C)

        let logits2d = self.lm_head.forward(&x2d)?;
        // (B*T, V)

        let (batch, tokens) = idx.dims2()?;
        
        // (B, T, V)

        logits2d.reshape((batch, tokens, self.config.vocab_size))
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
                .iter()
                .find(|t| t.id == next_id as u16)
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
