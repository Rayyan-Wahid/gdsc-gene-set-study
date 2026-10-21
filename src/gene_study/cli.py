import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from .data import coverage, prepare, read_table
from .experiment import Config, evaluate


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Single-drug gene-set study")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run a small SYNTHETIC smoke experiment")
    demo.add_argument("--output", default="outputs/demo")
    for name in ["coverage", "run"]:
        command = commands.add_parser(name)
        command.add_argument("--expression", required=True)
        command.add_argument("--responses", required=True)
        command.add_argument("--dataset", required=True)
        if name == "run":
            command.add_argument("--drug", required=True)
            command.add_argument("--release", required=True)
            command.add_argument("--output", required=True)
            command.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.command == "demo":
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(size=(60, 80)), columns=[f"synthetic_gene_{i}" for i in range(80)],
                         index=[f"synthetic_line_{i}" for i in range(60)])
        y = pd.Series(2*X.iloc[:, 0]-X.iloc[:, 1]+rng.normal(size=60), index=X.index)
        X = X.mask(rng.random(X.shape) < 0.03)
        print(evaluate(X, y, args.output, Config(gene_counts=(5, 20, 50), outer_folds=3,
                       inner_folds=2, trees=20), {"synthetic": True, "purpose": "smoke test only"}).to_string(index=False))
        return
    expression, responses = read_table(args.expression), read_table(args.responses)
    if args.command == "coverage":
        print(coverage(expression, responses, args.dataset).to_string(index=False))
        return
    X, y, audit = prepare(expression, responses, args.drug, args.dataset)
    audit.update({"release": args.release, "synthetic": False,
                  "expression_sha256": digest(args.expression), "responses_sha256": digest(args.responses)})
    print(evaluate(X, y, args.output, Config(n_jobs=args.jobs), audit).to_string(index=False))


if __name__ == "__main__":
    main()
