# How we work

## Branches and pull requests

- `main` always passes `python -m pytest -q`.
- Each member works on their own branch: `nikhil-crnn`, `prabhat-trocr`, `ravi-charcnn`, `monu-report`.
- Open a pull request to `main` when something works. Fill in the checklist of the PR template. Another member reads it before it is merged.
- Commit messages are short and plain: what changed and why. No attribution trailers.

## Who owns which files

Edit only your own files. Never run `git add .`; add files by name.

| Owner | Files |
|---|---|
| Nikhil | `scripts/`, `src/crnn/`, `configs/crnn/`, `data/processed/words/`, `src/common/` (frozen), `CONTRACTS.md` (frozen), `report/sections/03_data.md`, `report/sections/05_crnn.md` |
| Prabhat | `src/trocr/`, `configs/trocr/`, `report/sections/06_trocr.md` |
| Ravi | `src/char_cnn/`, `configs/char_cnn/`, `results/summary/`, `report/sections/04_char_cnn.md` |
| Monu | `README.md`, `report/` (all other sections and the merged report) |

Frozen files (`src/common/`, `CONTRACTS.md`, `requirements.txt`, `requirements/base.txt`, `.gitignore`) change only when all four agree, in one pull request by Nikhil.

## Rules that keep the numbers comparable

1. Use `src/common/metrics.py` for every error rate. Never write your own.
2. Save every run with `save_results` (`src/common/io.py`), named `<model>_<subset>_<decoding>`, for example `crnn_full_greedy`.
3. Labels are Unicode NFC. The character list comes from the training split only.
4. Tune on the validation split. Evaluate on the test split **once**, at the end.
5. Seed 42 (`src/common/seed.py`).
6. Never commit datasets, images or checkpoints (`*.pt`, `*.pth`). They are in `.gitignore`.

## Before you open a pull request

```bash
python -m pytest -q
```

Run the training command you changed on a tiny setting (for example `--overfit_n 20`) so it is clear the code runs.
