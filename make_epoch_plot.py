from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import os

log_dir = "lightning_logs_per_epoch"
#log_dir = "lightning_logs_per_epoch/lightning_logs_cats_pretrained_logs_2026-01-14_21:37:57/cats_pretrained_runname_2026-01-14_21:37:57/version_0"
plt.figure()
i = 0

for sub_dir in os.walk(log_dir):
    if "version_" in sub_dir[0]:
        ea = event_accumulator.EventAccumulator(
            sub_dir[0],
            size_guidance={"scalars": 0},
        )
        ea.Reload()

        # List available scalar tags
        print(ea.Tags()["scalars"])

        # Read your metric
        if len(ea.Tags()["scalars"]) != 0:
            events = ea.Scalars("acc/train_epoch")  # or "train_acc_epoch"

            epochs = [i for i in range(len(events))]
            values = [e.value for e in events]

            plt.plot(epochs, values, marker="o", label=f"run_{i}")
            i += 1

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy per Epoch")
plt.legend()
plt.show()
