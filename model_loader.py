import os
import cifar10.model_loader

def load(dataset, model_name, model_file, data_parallel=False, config={}):
    confopt_models = ["nb201", "darts", "robust_darts"]
    if model_name in confopt_models:
        net = cifar10.model_loader.load_confopt_model(model_name, model_file, data_parallel, config)
    elif dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)
    return net
