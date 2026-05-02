# tools/evidence

Evidence validation tools for the satellite research evidence pack.

## validate_research_pack.py

Validates the integrity of the research evidence pack without requiring the dataset
or model checkpoints.

### Usage

```bash
python tools/evidence/validate_research_pack.py
```

### What It Checks

1. All required research pack files exist in `docs/research_pack/`
2. `results/metrics.json` exists and is valid JSON
3. `results/model_comparison.csv` exists and contains all four expected model rows
4. Model names in `results/model_comparison.csv` are referenced in
   `docs/research_pack/METRIC_PROVENANCE_MATRIX.md`
5. Forbidden overclaim phrases are absent from all research pack documents
6. Required limitation phrases appear somewhere in the research pack documents

### Forbidden Phrases

The validator checks that none of the research pack documents contain:

- `state-of-the-art`
- `production deployment`
- `operational agricultural monitoring`
- `validated real-world deployment`
- `real-time monitoring platform`

### Required Phrases

The validator checks that the research pack documents collectively contain:

- `dataset is not bundled`
- `checkpoints are not bundled`
- `binary`
- `CNN-ViT hybrid`
- `preserved`

### Exit Codes

- `0` — all checks pass
- `1` — one or more checks failed (details printed to stdout)
