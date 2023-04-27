import os


def write1(dataset, scripts_for, exp_type=None, config=None):
    rsh.writelines('#!/bin/bash\n')
    rsh.writelines('#SBATCH --gres=gpu:1 # Request 1 GPU core. This takes up the complete computation node\n')
    rsh.writelines('#SBATCH --cpus-per-task=4 # Request 4 CPU cores. This takes up the complete computation node\n')
    rsh.writelines('#SBATCH --mem=22000M # Memory proportional to GPUs: 22000M per GPU core\n')
    rsh.writelines('#SBATCH --time=1-00:00\n')
    rsh.writelines('#SBATCH --mail-user=email@email.com # Email me when job starts, ends or fails\n')
    rsh.writelines('#SBATCH --mail-type=ALL\n')
    rsh.writelines('#SBATCH --account=sponsorname # Resource Allocation Project Identifier\n')
    if exp_type is not None and config is not None:
        rsh.writelines(
            '#SBATCH -o ./output_logs/' + scripts_for + os.path.sep + config + os.path.sep + exp_type + os.path.sep
            + dataset + '.out # STDOUT\n')
    else:
        rsh.writelines('#SBATCH -o ./output_logs/' + scripts_for + os.path.sep + dataset + '.out # STDOUT\n')
    rsh.writelines('module load python/3.9 cuda cudnn gcc/9.3.0 arrow scipy-stack\n')
    rsh.writelines('SOURCEDIR=path\n')
    rsh.writelines('# Prepare virtualenv\n')
    rsh.writelines('virtualenv --no-download $SLURM_TMPDIR/env\n')
    rsh.writelines('source $SLURM_TMPDIR/env/bin/activate\n')
    rsh.writelines('# Install packages on the virtualenv\n')
    rsh.writelines('pip install --no-index --upgrade pip torch torchvision tqdm\n')
    rsh.writelines('pip install -r requirements.txt\n')
    rsh.writelines('# Start training\n')


scripts_list = ["reproduce-glue", "student"]  # reproduce-glue OR student
layer_config_list = ["layer_config1.json", "layer_config2.json", "layer_config3.json", "layer_config4.json"]
type_of_exp_list = ["logit", "hidden", "embedding", "hid_embed", "log_hid", "log_embed", "log_hid_embed"]
datasets_list = ["cola", "mrpc", "rte"]

for scripts_for in scripts_list:
    path = os.getcwd() + os.path.sep + "sh-scripts" + os.path.sep + scripts_for + os.path.sep
    os.makedirs(path, exist_ok=True)
    if scripts_for == "reproduce-glue":
        for dataset in datasets_list:
            with open(path + dataset + '_rg.sh', 'w') as rsh:
                write1(dataset, scripts_for)
                rsh.writelines('python ./src/reproducebase/reproduce_glue.py --dataset="' + dataset + '" --device="cuda" ')
                rsh.writelines(' --logdir="exps/' + scripts_for + os.path.sep + dataset + '/"\n')
    else:
        for layer_config in layer_config_list:
            for exp_type in type_of_exp_list:
                for dataset in datasets_list:
                    new_path = path + layer_config.split(".")[0] + os.path.sep + exp_type + os.path.sep
                    os.makedirs(new_path, exist_ok=True)
                    with open(new_path + dataset + '_' + exp_type + '.sh', 'w') as rsh:
                        write1(dataset, scripts_for, exp_type, layer_config.split(".")[0])
                        log_dir_path = "exps/" + scripts_for + os.path.sep + layer_config.split(".")[0] + os.path.sep + \
                                    exp_type + os.path.sep + dataset + os.path.sep
                        rsh.writelines(
                            'python ./src/distilmodel/student_train_' + exp_type + '_distill.py --dataset="' + dataset + '" --device="cuda" ')
                        rsh.writelines(' --logdir="' + log_dir_path + '"')
                        rsh.writelines(' --layer_config="config/' + layer_config + '"\n')
