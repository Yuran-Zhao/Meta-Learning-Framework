# can design the model by your own
# however, have to inherit from the `metaBaseModel`
from .removed_residual_connection_bert import removedResidualConnectionBertModel
from .meta_transformer import metaTransformer
from .meta_bert import metaBert
from .classifier import MLPClassifier


def build_model(args):
    if args.model_type == 'bert':
        return metaBert(args)
    elif args.model_type == 'transformer':
        return metaTransformer(args)
    else:
        print("{} is not supported yet".format(args.model_type))
        exit(1)