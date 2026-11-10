## What this PR does

<!-- One or two sentences. Example: "CRNN: add prefix beam search and lexicon decoding." -->

## Owner checklist

- [ ] I only changed files that I own (see Section 5 of the team plan / `.github/CODEOWNERS`)
- [ ] I did **not** change frozen files (`src/common/`, `CONTRACTS.md`, `requirements.txt`, `requirements/base.txt`, `.gitignore`)
- [ ] No dataset images, checkpoints (`*.pt`, `*.pth`) or large files are included
- [ ] The code runs: I tested it with the command below
- [ ] Outputs follow the contracts (CSV manifests / `save_results` / `predict.py` API)

## How I tested it

```
# paste the command(s) you ran and a short result, e.g.
# python -m src.crnn.train --config configs/crnn/toy.yaml   -> overfit test CER 0.00
```

## Results (if any)

<!-- e.g. val CER, test CER, link to results/<run_name>/metrics.json -->
