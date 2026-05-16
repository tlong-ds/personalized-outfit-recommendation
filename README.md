# Alibaba iFashion - POG Reproduction Scaffold

This repo now implements a practical reproduction pipeline for the KDD'19 POG paper
("Personalized Outfit Generation for Fashion Recommendation at Alibaba iFashion").

## What Is Implemented

- Phase 1: canonical data builder from raw `item_data.txt`, `outfit_data.txt`, `user_data.txt`
- Phase 2: deterministic split manifest + FITB/CP evaluation set builders
- Phase 3: pragmatic-faithful multimodal embedding proxies with modality ablations
- Phase 4: FOM proxy evaluation and baseline comparisons (F-LSTM, Bi-LSTM, SetNN proxies)
- Phase 5: offline POG personalization proxy metrics (`Hit@K`, `MRR@K`, `NDCG@K`)

Notes:
- This is an offline reproduction framework. Alibaba's online Dida CTR experiment is out of scope.
- Proprietary embedding components are replaced by deterministic open proxies and documented as deltas.

## Project Layout

- `data/raw/`: original text files
- `data/interim/canonical/`: normalized tables
- `data/processed/`: split manifests + FITB/CP sets
- `artifacts/`: embeddings + metrics outputs
- `scripts/`: phase CLIs
- `src/graph_recsys/`: pipeline code

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run Multi-Phase Pipeline

```bash
python scripts/prepare_data.py --config configs/base.yaml
python scripts/build_eval_sets.py --config configs/base.yaml
python scripts/train_embeddings.py --config configs/base.yaml
python scripts/train_fom.py --config configs/base.yaml
python scripts/train_baselines.py --config configs/base.yaml
python scripts/train_pog.py --config configs/base.yaml
python scripts/evaluate.py --config configs/base.yaml --task all
```

## Main Outputs

- `data/interim/canonical/canonical_stats.json`
- `data/processed/split_manifest.json`
- `data/processed/fitb_eval.csv`
- `data/processed/cp_eval.csv`
- `artifacts/item_embeddings.csv`
- `artifacts/fom_metrics.json`
- `artifacts/baseline_metrics.json`
- `artifacts/pog_proxy_metrics.json`
