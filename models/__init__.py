from .trans_vg import TransVG


def build_model(args):
    return TransVG(args)
