"""Shared outer folds with fully nested preprocessing and hyperparameter search."""
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from joblib import Memory
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from .models import ConvergedElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .features import GeneSelector


@dataclass
class Config:
    gene_counts: tuple = (50, 200, 1000)
    outer_folds: int = 5
    inner_folds: int = 3
    seed: int = 42
    max_missing: float = 0.2
    n_jobs: int = 1
    trees: int = 200
    cache_dir: str | None = None


def evaluate(X, y, output, config=None, metadata=None):
    config = config or Config()
    if config.outer_folds < 2 or config.inner_folds < 2:
        raise ValueError("Both CV fold counts must be at least two.")
    if len(X) // config.outer_folds < 2:
        raise ValueError("Each test fold needs at least two samples for R2.")
    if not X.index.equals(y.index) or not X.index.is_unique:
        raise ValueError("X and y must have identical, unique cell-line indices.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Output directory must be empty to preserve earlier runs.")
    outer = list(KFold(config.outer_folds, shuffle=True, random_state=config.seed).split(X))
    memory = Memory(config.cache_dir, verbose=0) if config.cache_dir else None
    # Save the design before fitting so interruptions still leave a reproducible record.
    (output / 'run_design.json').write_text(json.dumps({'config': asdict(config), 'metadata': metadata or {}}, indent=2))
    rows, predictions, genes, assignments, tuning = [], [], [], [], []
    for fold, (train, test) in enumerate(outer, 1):
        print(f"Outer fold {fold}/{config.outer_folds}", flush=True)
        inner = list(KFold(config.inner_folds, shuffle=True, random_state=config.seed+fold).split(train))
        for idx in test:
            assignments.append({"cell_line_id": X.index[idx], "fold": fold})
        baseline = DummyRegressor(strategy="mean").fit(np.zeros((len(train), 1)), y.iloc[train])
        jobs = [("mean_baseline", 0, baseline.predict(np.zeros((len(test), 1))))]
        for k in config.gene_counts:
            models = {
                "elastic_net": (ConvergedElasticNet(max_iter=30000, selection='random', random_state=config.seed),
                                {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.1, 0.5, 0.9]}),
                "random_forest": (RandomForestRegressor(n_estimators=config.trees, random_state=config.seed, n_jobs=1),
                                  {"model__max_features": [0.3, 1.0], "model__min_samples_leaf": [2, 5]}),
            }
            for name, (model, grid) in models.items():
                print(f'  {name}, {k} genes: inner search', flush=True)
                pipe = Pipeline([("genes", GeneSelector(k, config.max_missing)),
                                 ("scale", StandardScaler() if name == "elastic_net" else "passthrough"),
                                 ("model", model)], memory=memory)
                search = GridSearchCV(pipe, grid, scoring="neg_root_mean_squared_error", cv=inner,
                                      n_jobs=config.n_jobs, error_score="raise")
                search.fit(X.iloc[train], y.iloc[train])
                best = search.best_estimator_
                jobs.append((name, k, best.predict(X.iloc[test])))
                tuning.append({"fold": fold, "model": name, "gene_count": k,
                               "inner_rmse": -search.best_score_, "params": search.best_params_})
                selector, fitted = best.named_steps["genes"], best.named_steps["model"]
                importance = fitted.coef_ if name == "elastic_net" else fitted.feature_importances_
                for index, score, weight in zip(selector.indices_, selector.scores_, importance):
                    genes.append({"fold": fold, "model": name, "gene_count": k, "gene": X.columns[index],
                                  "abs_pearson": score, "weight": weight})
        for name, k, pred in jobs:
            truth = y.iloc[test]
            rows.append({"fold": fold, "model": name, "gene_count": k,
                         "rmse": np.sqrt(mean_squared_error(truth, pred)),
                         "mae": mean_absolute_error(truth, pred), "r2": r2_score(truth, pred)})
            predictions.extend({"cell_line_id": ident, "fold": fold, "model": name, "gene_count": k,
                                "observed": actual, "predicted": estimate}
                               for ident, actual, estimate in zip(truth.index, truth, pred))
        pd.DataFrame(rows).to_csv(output / 'checkpoint_fold_metrics.csv', index=False)
        pd.DataFrame(predictions).to_csv(output / 'checkpoint_predictions.csv', index=False)
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby(["model", "gene_count"])[["rmse", "mae", "r2"]].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.reset_index()
    selected = pd.DataFrame(genes)
    frequency = selected.groupby(["model", "gene_count", "gene"]).fold.nunique().rename("fold_count").reset_index()
    frequency["selection_fraction"] = frequency.fold_count / config.outer_folds
    for filename, frame in {"fold_metrics": metrics, "summary": summary, "predictions": pd.DataFrame(predictions),
                            "selected_genes": selected, "selection_frequency": frequency,
                            "fold_assignments": pd.DataFrame(assignments)}.items():
        frame.to_csv(output / f"{filename}.csv", index=False)
    manifest = {"config": asdict(config), "metadata": metadata or {}, "python": platform.python_version(),
                "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "tuning": tuning}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot_results(summary, output, synthetic=(metadata or {}).get("synthetic", False))
    return summary


def plot_results(summary, output, synthetic=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ["elastic_net", "random_forest"]:
        values = summary.loc[summary.model == name].sort_values("gene_count")
        ax.errorbar(values.gene_count, values.rmse_mean, yerr=values.rmse_std,
                    marker="o", capsize=4, label=name.replace("_", " "))
    base = summary.loc[summary.model == "mean_baseline", "rmse_mean"].iloc[0]
    ax.axhline(base, color="gray", linestyle="--", label="training mean baseline")
    ax.set(xlabel="Selected genes", ylabel="Outer-fold RMSE (LN_IC50)",
           title=("SYNTHETIC DEMO — " if synthetic else "") + "Gene-set comparison\n(error bars: fold standard deviation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "rmse_by_gene_count.png", dpi=160)
    plt.close(fig)
