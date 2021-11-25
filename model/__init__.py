# can design the model by your own
# however, have to inherit from the `metaBaseModel`
from .meta_base_model import metaBaseModel
from .meta_transformer import metaTransformer
from .meta_bert import metaBert


def build_model(args):
    if args.model_type == 'bert':
        return metaBert(args)
    elif args.model_type == 'transformer':
        return metaTransformer(args)
    else:
        print("{} is not supported yet".format(args.model_type))
        exit(1)