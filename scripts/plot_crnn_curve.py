"""Plot the CRNN learning curve from results/crnn_full_greedy/training_log.csv.

Where this file fits: a report helper. It runs after training (the log has one row per epoch,
the same columns train.py writes to log.csv) and saves a figure for the report.
Run from the repository root: python scripts/plot_crnn_curve.py
The left plot shows train and validation CTC loss. The right plot shows validation CER in percent.
If validation loss rises while train loss falls, the model is overfitting.
"""
# csv reads the log; matplotlib draws the figure.
import csv
import matplotlib
matplotlib.use("Agg")  # file-only backend: no window is needed, so it also works on a server
import matplotlib.pyplot as plt

# Read the log: one dict per epoch, with keys such as epoch, train_loss, val_loss, val_cer.
rows = list(csv.DictReader(open("results/crnn_full_greedy/training_log.csv", encoding="utf-8")))
# The log counts epochs from 0, so +1 makes the x-axis start at epoch 1.
ep = [int(r["epoch"]) + 1 for r in rows]
# Two plots side by side (1 row, 2 columns).
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
# Left: train and validation loss against epoch. CSV values are text, so float() converts them.
ax[0].plot(ep, [float(r["train_loss"]) for r in rows], label="train")
ax[0].plot(ep, [float(r["val_loss"]) for r in rows], label="validation")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("CTC loss"); ax[0].legend()
# Right: validation CER. The log stores a fraction (0.12), so multiply by 100 to show percent.
ax[1].plot(ep, [100 * float(r["val_cer"]) for r in rows], color="tab:green")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation CER (%)")
plt.tight_layout()  # stop the labels from overlapping
plt.savefig("report/figures/crnn_learning_curve.png", dpi=150)  # dpi=150: sharp enough for a report
