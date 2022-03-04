from .trans_vg import TransVG


def build_model(args):
    model = TransVG(args)
    print(model)
    return model
