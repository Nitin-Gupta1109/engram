# Contributing to Engram

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/Nitin-Gupta1109/engram.git
cd engram
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

## Development

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check engram/ tests/
ruff format engram/ tests/
```

## What to Work On

- Check [open issues](https://github.com/Nitin-Gupta1109/engram/issues) for bugs and feature requests
- Improving retrieval accuracy on weaker categories (temporal, multi-hop)
- Adding new backend integrations
- Documentation improvements

## Pull Requests

1. Fork the repo and create a branch from `main`
2. Add tests for any new functionality
3. Make sure `pytest` and `ruff check` pass
4. Submit a PR with a clear description of the change

## Running Benchmarks

```bash
# LongMemEval
python benchmarks/longmemeval_bench.py data/longmemeval_s_cleaned.json --mode hybrid

# LoCoMo
python benchmarks/locomo_bench.py data/locomo10.json --mode hybrid
```

## Code Style

- Line length: 100 characters
- Linter: ruff (E, F, W, I rules)
- Target: Python 3.9+

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
