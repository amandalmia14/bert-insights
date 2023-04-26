import json
import os

path = os.getcwd() + os.path.sep + "exps"
exp_name_list = ["reproduce-glue", "student"]
datasets_list = ["cola", "mrpc", "rte"]
layer_config_list = ["layer_config1", "layer_config2", "layer_config3", "layer_config4"]
exp_type_list = ["logit", "hidden", "embedding", "hid_embed", "log_hid", "log_embed", "log_hid_embed"]

results = {}
for exp_name in exp_name_list:
    if exp_name == "reproduce-glue":
        reproduced_results = {}
        for dataset in datasets_list:
            reproduced_results[dataset] = []
            file_dir = path + os.path.sep + exp_name + os.path.sep + dataset + os.path.sep
            for op_file in os.listdir(file_dir):
                if "results" in op_file:
                    with open(file_dir + op_file, "r") as f:
                        data = json.load(f)
                        if dataset == "cola":
                            reproduced_results[dataset].append(round(data["test_math_coef"], 3))
                        else:
                            reproduced_results[dataset].append(round(data["test_acc"], 3))

        results[exp_name] = reproduced_results
    else:
        student_results = {}
        for layer_config in layer_config_list:
            student_results[layer_config] = {}
            for exp_type in exp_type_list:
                student_results[layer_config][exp_type] = {}
                for dataset in datasets_list:
                    student_results[layer_config][exp_type][dataset] = []
                    file_dir = path + os.path.sep + exp_name + os.path.sep + layer_config + os.path.sep + exp_type + \
                               os.path.sep + dataset + os.path.sep
                    try:
                        for op_file in os.listdir(file_dir):
                            if "results" in op_file:
                                with open(file_dir + op_file, "r") as f:
                                    data = json.load(f)
                                if dataset == "cola":
                                    student_results[layer_config][exp_type][dataset].append(
                                        round(data["test_math_coef"], 3))
                                else:
                                    student_results[layer_config][exp_type][dataset].append(round(data["test_acc"], 3))
                    except Exception as e:
                        print(e)
                        print(file_dir)
            results[exp_name] = student_results

with open("results.json", "w") as f:
    json.dump(results, f, indent=4)
print(results)
