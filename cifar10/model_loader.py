import os
import torch, torchvision
import cifar10.models.vgg as vgg
import cifar10.models.resnet as resnet
import cifar10.models.densenet as densenet
from searchspace import (
    BabyDARTSSearchSpace,
    DARTSGenotype,  # noqa: F401
    DARTSImageNetModel,
    DARTSModel,
    DARTSSearchSpace,
    NAS201Genotype,
    NASBench201Model,
    NASBench201SearchSpace,
    RobustDARTSSearchSpace,
    SearchSpace,
    TransNASBench101SearchSpace,
)
import sys
from enum import Enum

from fvcore.common.checkpoint import Checkpointer



# map between model name and function
models = {
    'vgg9'                  : vgg.VGG9,
    'densenet121'           : densenet.DenseNet121,
    'resnet18'              : resnet.ResNet18,
    'resnet18_noshort'      : resnet.ResNet18_noshort,
    'resnet34'              : resnet.ResNet34,
    'resnet34_noshort'      : resnet.ResNet34_noshort,
    'resnet50'              : resnet.ResNet50,
    'resnet50_noshort'      : resnet.ResNet50_noshort,
    'resnet101'             : resnet.ResNet101,
    'resnet101_noshort'     : resnet.ResNet101_noshort,
    'resnet152'             : resnet.ResNet152,
    'resnet152_noshort'     : resnet.ResNet152_noshort,
    'resnet20'              : resnet.ResNet20,
    'resnet20_noshort'      : resnet.ResNet20_noshort,
    'resnet32_noshort'      : resnet.ResNet32_noshort,
    'resnet44_noshort'      : resnet.ResNet44_noshort,
    'resnet50_16_noshort'   : resnet.ResNet50_16_noshort,
    'resnet56'              : resnet.ResNet56,
    'resnet56_noshort'      : resnet.ResNet56_noshort,
    'resnet110'             : resnet.ResNet110,
    'resnet110_noshort'     : resnet.ResNet110_noshort,
    'wrn56_2'               : resnet.WRN56_2,
    'wrn56_2_noshort'       : resnet.WRN56_2_noshort,
    'wrn56_4'               : resnet.WRN56_4,
    'wrn56_4_noshort'       : resnet.WRN56_4_noshort,
    'wrn56_8'               : resnet.WRN56_8,
    'wrn56_8_noshort'       : resnet.WRN56_8_noshort,
    'wrn110_2_noshort'      : resnet.WRN110_2_noshort,
    'wrn110_4_noshort'      : resnet.WRN110_4_noshort,
}

class SearchSpaceType(Enum):
    DARTS = "darts"
    NB201 = "nb201"
    NB1SHOT1 = "nb1shot1"
    TNB101 = "tnb101"
    BABYDARTS = "baby_darts"
    RobustDARTS = "robust_darts"

def load(model_name, model_file=None, data_parallel=False):
    net = models[model_name]()
    if data_parallel: # the model is saved in data paralle mode
        net = torch.nn.DataParallel(net)

    if model_file:
        assert os.path.exists(model_file), model_file + " does not exist."
        stored = torch.load(model_file, map_location=lambda storage, loc: storage)
        if 'state_dict' in stored.keys():
            net.load_state_dict(stored['state_dict'])
        else:
            net.load_state_dict(stored)

    if data_parallel: # convert the model back to the single GPU version
        net = net.module

    net.eval()
    return net

def get_model(
    model_name: SearchSpaceType,
    config: dict,
) -> None:
    if model_name == SearchSpaceType.NB201:
        model = NASBench201SearchSpace(**config)
    elif model_name == SearchSpaceType.DARTS:
        model = DARTSSearchSpace(**config)
    elif model_name == SearchSpaceType.TNB101:
        model = TransNASBench101SearchSpace(**config)
    elif model_name == SearchSpaceType.BABYDARTS:
        model = BabyDARTSSearchSpace(**config)
    elif model_name == SearchSpaceType.RobustDARTS:
        model = RobustDARTSSearchSpace(**config)
    return model

def load_confopt_model(model_name, model_file=None, data_parallel=False, config={}):
    model = get_model(SearchSpaceType(model_name), config)
    
    if data_parallel and torch.cuda.is_available():
        net = torch.nn.DataParallel(model).cuda()
    else:
        net = model
    checkpointer = Checkpointer(
        model=net,
        save_dir=model_file,
        save_to_disk=True,
    )
    checkpoint_info = checkpointer._load_file(f=model_file)
    net.load_state_dict(checkpoint_info["model"])

    if data_parallel: # convert the model back to the single GPU version
        net = net.module

    net.eval()
    return net