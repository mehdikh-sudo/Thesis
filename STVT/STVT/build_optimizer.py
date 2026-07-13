import torch.optim as optim


def build_optimizer(args, model):
    opt_name = getattr(args, "optimizer", None) or getattr(args, "optim", "sgd")
    opt_name = opt_name.lower()

    # Legacy selection (for easy rollback):
    # if args.optim == 'sgd':
    #     return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
    #                      weight_decay=args.weight_decay, nesterov=args.nesterov)
    # elif args.optimizer == 'Adam':
    #     return optim.Adam(model.parameters(), args.lr, betas=(args.adam_beta1, args.adam_beta2),
    #                       weight_decay=args.weight_decay, eps=args.eps)
    # elif args.optimizer == 'AdamW':
    #     return optim.AdamW(model.parameters(), args.lr, betas=(args.adam_beta1, args.adam_beta2),
    #                        weight_decay=args.weight_decay, eps=args.eps)
    # elif args.optimizer == 'RMSprop':
    #     return optim.RMSprop(model.parameters(), args.lr, alpha=args.rms_alpha, eps=args.eps,
    #                          weight_decay=args.weight_decay, momentum=args.momentum)

    if opt_name == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    if opt_name == 'adam':
        return optim.Adam(
            model.parameters(),
            args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.weight_decay,
            eps=args.eps,
        )
    if opt_name == 'adamw':
        return optim.AdamW(
            model.parameters(),
            args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.weight_decay,
            eps=args.eps,
        )
    if opt_name == 'rmsprop':
        return optim.RMSprop(
            model.parameters(),
            args.lr,
            alpha=args.rms_alpha,
            eps=args.eps,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )

    raise ValueError(f"Invalid optimizer specified: {opt_name}")


