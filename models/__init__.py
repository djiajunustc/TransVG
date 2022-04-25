from .lvit import LViT


def build_model(args):
    model = LViT(args)
    print(model)
    return model
