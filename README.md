# llm-rs
This repository will be a place for me to learn LLM basics. Target is to write a small gpt-style model that can run on CPU, written in candle(similar to python torch). Its purely educational and for fun.

## Requirements
- Python
- uv
- Rust

## Development
1. Download text corpus.
```bash
uv sync
uv run gutenberg.py
```

## How to use
Run 
```shell
cargo run --release # for optimal compilation and fast training
```

## Roadmap
- [x] Byte Pair Encoding
- [x] Tokenizer 
- [ ] Tokenize a big corpus
- [x] Dense Architecture LLM 
- [ ] Mixture of Expert Architecture