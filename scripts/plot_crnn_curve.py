"""Plot the CRNN learning curve from results/crnn_full_greedy/training_log.csv."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results/crnn_full_greedy/training_log.csv", encoding="utf-8")))
ep = [int(r["epoch"]) + 1 for r in rows]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot(ep, [float(r["train_loss"]) for r in rows], label="train")
ax[0].plot(ep, [float(r["val_loss"]) for r in rows], label="validation")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("CTC loss"); ax[0].legend()
ax[1].plot(ep, [100 * float(r["val_cer"]) for r in rows], color="tab:green")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation CER (%)")
plt.tight_layout()
plt.savefig("report/figures/crnn_learning_curve.png", dpi=150)
