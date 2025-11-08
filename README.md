Test Results 07.10. Self-Trained:

Epoch [2/10], Loss: 0.9610, Validation Accuracy: 0.6727 <br>
Epoch [3/10], Loss: 0.8141, Validation Accuracy: 0.6943 <br>
Epoch [4/10], Loss: 0.6974, Validation Accuracy: 0.7107 <br>
Epoch [5/10], Loss: 0.5997, Validation Accuracy: 0.7051 <br>
Epoch [6/10], Loss: 0.5059, Validation Accuracy: 0.7089 <br>
Epoch [7/10], Loss: 0.4197, Validation Accuracy: 0.7142 <br>
Epoch [8/10], Loss: 0.3404, Validation Accuracy: 0.7171 <br>
Epoch [9/10], Loss: 0.2737, Validation Accuracy: 0.7154 <br>
Epoch [10/10], Loss: 0.2123, Validation Accuracy: 0.7045 <br>

## Test Results 08.10. CIFAR-100 Pre-Trained:

**Parameters:**
- dropout_rate: 0.2903538439232585
- learning_rate: 0.00018777606774170717
- unfreeze_layers: 7
- weight_decay: 0.004878073851651753

**Test Metrics**
- acc/test: 0.9032999873161316
- loss/test: 0.40259337425231934

**Resources hunger metrics** \
_Optuna hyperparameter search_
- Total time: 252.76 min
- Total energy: 1.490 kWh
  - Energy consumed for All CPU : 0.184000 kWh
  - Energy consumed for all GPUs : 1.021125 kWh
- Estimated CO2 emissions: 0.567523 kg

_Training of Network with Tuned Hyperparameters_
- Training complete in 46.00 min
- Max GPU memory: 32.94 GB
- Total energy: 0.288 kWh
  - Energy consumed for All CPU : 0.033260 kWh
  - Energy consumed for all GPUs : 0.202614 kWh
- Estimated CO2 emissions: 0.109554 kg
