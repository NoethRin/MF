from model.mlp import fMRIModule
from model.brainmagic import MEGModule
from model.fusion import FusionModule
from model.baseline import RandomModule

def select_model(args):
    if args.model_type == "meg":
        return MEGModule()
    if args.model_type == "fmri":
        return fMRIModule()
    elif args.model_type == "fusion":
        return FusionModule()
    elif args.model_type == "random":
        return RandomModule()
    else:
        NotImplemented
