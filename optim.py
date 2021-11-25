from torch.optim import Adam, lr_scheduler


def build_optimizer(params, args):
    Adam(params, lr=args.inner_lr), Adam(params, lr=args.outer_lr)


def build_scheduler(inner_optimizer, outer_optimizer, args):
    inner_scheduler = lr_scheduler.ReduceLROnPlateau(inner_optimizer,
                                                     'max',
                                                     patience=args.patience //
                                                     2,
                                                     factor=0.1,
                                                     verbose=True)
    outer_scheduler = lr_scheduler.ReduceLROnPlateau(outer_optimizer,
                                                     'max',
                                                     patience=args.patience //
                                                     2,
                                                     factor=0.1,
                                                     verbose=True)
    return inner_scheduler, outer_scheduler
