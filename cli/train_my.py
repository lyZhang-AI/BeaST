"""Command-line entry point for the proposed method."""

import argparse

from aad.training.proposed import main


def parse_args():
    parser = argparse.ArgumentParser(description="Train the proposed model with trial-independent splits.")
    parser.add_argument("--dataset", choices=["KUL", "DTU", "AVGC"], required=True)
    parser.add_argument("--subject", default="S1")
    parser.add_argument("--time-len", type=float, default=2.0)
    parser.add_argument("--fold", type=int, default=None, help="Trial-CV fold; omit for one trial-independent split.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--condition", choices=["NV", "SV", "MV", "MTN"], default="NV")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--label-mode", choices=["half", "trial"], default="half")
    parser.add_argument("--avgc-root", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        name=args.subject,
        time_len=args.time_len,
        dataset=args.dataset,
        fold_id=args.fold,
        max_epoch=args.epochs,
        seed=args.seed,
        condition=args.condition,
        variant=args.variant,
        label_mode=args.label_mode,
        avgc_root=args.avgc_root,
    )
