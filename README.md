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

## Test Results 08.11. CIFAR-100 Pre-Trained:

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

## Test Results 11.11. CIFAR-100 Pre-Trained:
**Parameters:**
- dropout_rate: 0.36466173895931375
- learning_rate: 0.0009032932923923763
- unfreeze_layers: 6
- weight_decay: 0.08867092775630593

**Test Metrics**
- TODO

**Resources hunger metrics** \
_Optuna hyperparameter search_
Total time: 120.62 min
Total energy: 0.539 kWh
Total CO2: 0.205520 kg
Energy consumed for All CPU : 0.097856 kWh
Energy consumed for all GPUs : 0.305809 kWh

_Training of Network with Tuned Hyperparameters_
Training complete in 7.59 min
Max GPU memory: 32.92 GB
Estimated CO2 emissions: 0.013710 kg
Total energy: 0.036 kWh

## Test Results 17.11. CIFAR-100 Self-Trained:
**Parameters:**
- dropout_rate: 0.48645517619542167
- learning_rate: 0.002913637707010568
- weight_decay: 0.04241168576887874
- 'stage1_repeats': 2, 'stage2_repeats': 2, 'stage3_repeats': 4, 'stage4_repeats': 5, 'stage5_repeats': 8, 'stage6_repeats': 6

**Test Metrics**
- acc/test: 0.6345999836921692 
- loss/test: 1.4687973260879517

**Resources hunger metrics** \
_Optuna hyperparameter search_
- Total time: ?

_Training of Network with Tuned Hyperparameters_
- Training complete in 31.26 min
- Max GPU memory: 13.02 GB
- Total energy: 0.167 kWh

## Test Results 18.11. CIFAR-100 sanity-check
**parameters:**
- {'learning_rate': 0.001572222062812689, 'dropout_rate': 0.2106854604258504, 'weight_decay': 0.09620862174008252}

**Test Metrics:**
- acc/test 0.6290000081062317
- loss/test 1.4781690835952759
