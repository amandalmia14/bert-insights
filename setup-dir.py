import os

scripts_list = ["student", "reproduce-glue"]

datasets_list = ["cola", "mrpc", "rte"]
config_list = ["layer_config1", "layer_config2", "layer_config3", "layer_config4"]
exp_type_list = ["logit", "hidden", "embedding", "hid_embed", "log_hid", "log_embed", "log_hid_embed"]

for scripts_for in scripts_list:
    path = os.getcwd() + os.path.sep + "output_logs" + os.path.sep + scripts_for + os.path.sep

    if scripts_for == "reproduce-glue":
        os.makedirs(path, exist_ok=True)

    else:
        for exp_type in exp_type_list:
            for config in config_list:
                os.makedirs(path + config + os.path.sep + exp_type + os.path.sep, exist_ok=True)

