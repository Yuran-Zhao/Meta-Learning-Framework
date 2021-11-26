import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    # Model setting
    parser.add_argument(
        '--model_type',
        '-model',
        type=str,
        default='bert',
        help="Which model to use in the task [bert | transformer]")

    parser.add_argument('--dropout',
                        type=float,
                        default=0.1,
                        help='Dropout rate in the model')

    # Dataset setting
    parser.add_argument('--n_train_class',
                        type=int,
                        required=True,
                        default=5,
                        help="The number of classes in the meta-train dataset")
    parser.add_argument('--n_valid_class',
                        type=int,
                        required=True,
                        default=5,
                        help="The number of classes in the meta-valid dataset")
    parser.add_argument('--n_test_class',
                        type=int,
                        required=True,
                        default=5,
                        help="The number of classes in the meta-test dataset")

    parser.add_argument('--N-way',
                        '-N',
                        type=int,
                        default=5,
                        help="The number of classes in each generated task")

    parser.add_argument(
        '--support',
        type=int,
        default=10,
        help=
        'The number of sampled `support` examples for each class in each task')

    parser.add_argument(
        '--shot',
        type=int,
        default=25,
        help='The number of sampled `query` examples for each class in each task'
    )

    # Training procedure
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='Batch size')

    parser.add_argument('--train_epochs',
                        type=int,
                        default=1000,
                        help='The number of training epochs')

    parser.add_argument(
        '--train_episodes',
        type=int,
        default=100,
        help='The number of generated task during each training epoch')

    parser.add_argument(
        '--valid_episodes',
        type=int,
        default=100,
        help='The number of generated task during each validating epoch')

    parser.add_argument(
        '--test_episodes',
        type=int,
        default=100,
        help='The number of generated task during each testing epoch')

    parser.add_argument('--seed',
                        type=int,
                        default=1234,
                        help='The random seed')

    parser.add_argument('--patience',
                        type=int,
                        default=20,
                        help='The patience used in scheduler')

    parser.add_argument('--clip_grad',
                        type=float,
                        default=None,
                        help='gradient clipping')

    parser.add_argument("--mode",
                        type=str,
                        default="test",
                        help=("Running mode."
                              "Options: [train, test]"
                              "[Default: test]"))

    parser.add_argument('--cuda',
                        type=int,
                        default=1,
                        help='cuda device numbers, -1 means running on cpu')

    parser.add_argument(
        '--inner_lr',
        type=float,
        default=1e-3,
        help=
        'The learning rate used in loss with `support` set (inner optimization)'
    )
    parser.add_argument(
        '--outer_lr',
        type=float,
        default=1e-3,
        help=
        'The learning rate used in loss with `query` set (outer optimization)')

    parser.add_argument('--lr_scheduler',
                        type=str,
                        default='ReduceLROnPlateau',
                        help="The scheduler for optimizers")

    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        help='The mode running the scripts [train | test]')

    return parser.parse_args()