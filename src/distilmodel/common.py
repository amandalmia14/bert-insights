import copy
import os

import torch.nn as nn

ps = os.path.sep


def get_student_dir(exp_type, dataset, layer_config):
    layer_config = layer_config.split('.')[0].split("/")[1]
    student_dir = os.getcwd() + ps + "src" + ps + "data" + ps + "studentmodel" + ps + layer_config.split(".")[0] + ps \
                  + exp_type + ps + dataset + ps
    return student_dir


def get_teacher_model_path(dataset):
    teacher_mdl_pth = os.getcwd() + ps + "src" + ps + "data" + ps + "reproducedmodel" + ps + dataset + ps + "2"
    return teacher_mdl_pth


def get_logs_path(logdir, time_now, exp_type, comfig_name, layer_config):
    layer_config = layer_config.split('.')[0].split("/")[1]
    config_new_path = os.path.join(logdir, time_now + "_student_" + exp_type + "_" + layer_config + "_" + comfig_name
                                   + '_config.json')
    result_new_path = os.path.join(logdir, time_now + "_student_" + exp_type + "_" + layer_config + "_" + comfig_name
                                   + '_results.json')
    return config_new_path, result_new_path


def create_student_model(teacher_model, student_to_teacher_layer):
    # adapted from https://github.com/huggingface/transformers/issues/2483
    oldModuleList = teacher_model.bert_model.encoder.layer
    newModuleList = nn.ModuleList()

    # copy the layers to keep
    # for student_layer_id in range(len(student_to_teacher_layer)):
    #     print("student_layer_id", type(student_layer_id), student_layer_id)
    #     newModuleList.append(oldModuleList[student_to_teacher_layer[int(student_layer_id)]])

    for student_layer_id, teacher_layer in student_to_teacher_layer.items():
        newModuleList.append(oldModuleList[teacher_layer])

    # create a copy of the model, modify it with the new list, and return
    student_model = copy.deepcopy(teacher_model)
    student_model.bert_model.encoder.layer = newModuleList

    return student_model
