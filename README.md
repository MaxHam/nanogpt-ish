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

## Roadmap
- [x] Byte Pair Encoding
- [ ] Tokenize a big corpus
- [ ] Generate vocab file
- [ ] Tokenizer 
- [ ] Dense Architecture LLM 
- [ ] Mixture of Expert Architecture