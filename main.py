#!/usr/bin/env python3
"""Project entrypoint.

Delegates to evaluation by default and exposes a train subcommand:

`python main.py --config config/CAP_task.yaml --task cap`
`python main.py train --config config/CAP_task.yaml --task cap`
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"train", "fit", "fine-tune", "finetune"}:
        from train import main as train_main

        return train_main(args[1:])
    if args and args[0] in {"eval", "evaluate"}:
        from evaluate import main as evaluate_main

        return evaluate_main(args[1:])

    from evaluate import main as evaluate_main

    return evaluate_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
