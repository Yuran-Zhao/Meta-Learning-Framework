from options import get_parser
from model import build_model
from dataset import MLDataset
from optim import build_optimizer
from utils import build_trainer
from tqdm import tqdm


def main():
    args = get_parser()
    train_dataset, valid_dataset = MLDataset(args)
    model = build_model(args)
    inner_optimizer, outer_optimizer = build_optimizer(args)
    trainer = build_trainer(args)

    trainer.train(model, train_dataset, valid_dataset, inner_optimizer,
                  outer_optimizer)


if __name__ == '__main__':
    main()
