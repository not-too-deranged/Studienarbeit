from lightning.pytorch.callbacks import Callback
import time
import torch
from codecarbon import EmissionsTracker

class ComputeCostLogger(Callback):
    def __init__(self, output_dir="./emission_logs"):
        self.start_time = None
        self.tracker = None
        self.output_dir = output_dir

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.tracker = EmissionsTracker(
            project_name="training_cost_comparison",
            output_dir=self.output_dir,
            measure_power_secs=15
        )
        self.tracker.start()

    def on_train_end(self, trainer, pl_module):
        duration = time.time() - self.start_time
        emissions = self.tracker.stop()

        print(f"\nTraining complete in {duration/60:.2f} min")
        print(f"Max GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        print(f"Estimated CO2 emissions: {emissions:.6f} kg")
        print(f"Total energy: {self.tracker.final_emissions_data.energy_consumed:.3f} kWh")
