import matplotlib.pyplot as plt
import json

testscenario_names = ["CIFAR-100_pretrained", "CIFAR-100_selftrained", "cats_pretrained", "cats_selftrained", "places365_pretrained", "places365_selftrained"]
required_boxplots_results = {"acc/test_epoch": "accuracy",
                             "loss/test_epoch": "loss"}
required_boxplots_hyperparameters = {"DROPOUT_RATE": "Dropout",
                                     "WEIGHT_DECAY": "weight decay",
                                     "LEARNING_RATE": "Learning rate"}#"UNFREEZE_LAYERS" "NUM_LAYERS"

with open('test_results.json', 'r') as file:
    data = json.load(file)

    for scenario in testscenario_names:
        testscenario_entries = []
        for entry in data:
            if scenario in entry["model_name"]:
                testscenario_entries.append(entry)

        for result_boxplot in required_boxplots_results:
            testresults = []
            for entry in testscenario_entries:
                testresults.append(entry["test_results"][result_boxplot])

            if len(testresults) != 5:
                print(f"{result_boxplot} has {len(testresults)} results instead of 5. {scenario}_{required_boxplots_results[f"{result_boxplot}"]} plot was not created.")
            else:
                fig = plt.figure(figsize=(4, 6))
                plt.boxplot(testresults)
                plt.title(f"{scenario}")
                plt.xticks([1], labels=[f"{required_boxplots_results[f"{result_boxplot}"]}"])
                plt.savefig(f"statistical_analysis_pics/results/{scenario}_{required_boxplots_results[f"{result_boxplot}"]}.png", bbox_inches="tight")

        if "pretrained" in scenario:
            required_boxplots_hyperparameters["UNFREEZE_LAYERS"] = "Unfrozen layers"
            required_boxplots_hyperparameters.pop("NUM_LAYERS", None)

        elif "selftrained" in scenario:
            required_boxplots_hyperparameters["NUM_LAYERS"] = "Number of layers" #TODO how do you deal with that?
            required_boxplots_hyperparameters.pop("UNFREEZE_LAYERS", None)

        for hyperparameter_boxplot in required_boxplots_hyperparameters:
            hyperparameter_results = []
            for entry in testscenario_entries:
                hyperparameter_results.append(entry["hyperparameters"][hyperparameter_boxplot])

            fig = plt.figure(figsize=(4, 6))

            if "NUM_LAYERS" in hyperparameter_boxplot:
                fig = plt.figure(figsize=(6, 6))
                hyperparameter_results = list(map(list, zip(*hyperparameter_results)))

            if len(hyperparameter_results) == 0:
                print(f"no entries for {scenario}_{hyperparameter_boxplot}")
                continue

            if "NUM_LAYERS" not in hyperparameter_boxplot and len(hyperparameter_results) != 5:
                print(f"{hyperparameter_boxplot} has {len(hyperparameter_results)} entries instead of 5. {scenario}_{required_boxplots_hyperparameters[f"{hyperparameter_boxplot}"]} plot was not created.")
            elif "NUM_LAYERS" in hyperparameter_boxplot and len(hyperparameter_results[0]) != 5:
                print(f"{hyperparameter_boxplot} has {len(hyperparameter_results[0])} entries instead of 5. {scenario}_{required_boxplots_hyperparameters[f"{hyperparameter_boxplot}"]} plot was not created.")
            else:
                plt.boxplot(hyperparameter_results)
                plt.title(f"{scenario}")
                if "NUM_LAYERS" in hyperparameter_boxplot:
                    plt.xticks(
                        range(1, len(hyperparameter_results) + 1),
                        [f"Layer_{i + 1}" for i in range(len(hyperparameter_results))]
                    )
                else:
                    plt.xticks([1], labels=[f"{required_boxplots_hyperparameters[f"{hyperparameter_boxplot}"]}"])

                plt.savefig(f"statistical_analysis_pics/hyperparameters/{scenario}_{required_boxplots_hyperparameters[f"{hyperparameter_boxplot}"]}.png", bbox_inches="tight")
