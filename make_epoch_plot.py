from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import os

log_dir = "lightning_logs_per_epoch/lightning_logs_cats_pretrained_logs_2026-01-14_21:37:57/cats_pretrained_runname_2026-01-14_21:37:57/version_0"
ea = event_accumulator.EventAccumulator(
    log_dir,
    size_guidance={"scalars": 0},
)
ea.Reload()

# List available scalar tags
print(ea.Tags()["scalars"])

# Read your metric
events = ea.Scalars("acc/train_epoch")  # or "train_acc_epoch"

epochs = [i for i in range(len(events))]
values = [e.value for e in events]

plt.figure()
plt.plot(epochs, values, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy per Epoch")
plt.show()
