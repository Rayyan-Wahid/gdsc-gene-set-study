"""Validate normalized inputs and resolve replicates before splitting cell lines."""
from pathlib import Path
import numpy as np
import pandas as pd


def read_table(path):
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path, sep="\t" if ".tsv" in path.suffixes else ",")


def clean_ids(values):
    if values.isna().any():
        raise ValueError("Missing identifiers are not allowed.")
    result = values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    if result.eq("").any():
        raise ValueError("Empty identifiers are not allowed.")
    return result


def prepare(expression, responses, drug, dataset):
    if "cell_line_id" not in expression or expression.shape[1] < 2:
        raise ValueError("Expression requires cell_line_id plus numeric gene columns.")
    required = {"cell_line_id", "drug_id", "dataset", "LN_IC50"}
    if not required.issubset(responses):
        raise ValueError(f"Response columns missing: {sorted(required - set(responses))}")
    expression, responses = expression.copy(), responses.copy()
    expression["cell_line_id"] = clean_ids(expression.cell_line_id)
    responses["cell_line_id"] = clean_ids(responses.cell_line_id)
    responses["drug_id"] = clean_ids(responses.drug_id)
    if expression.cell_line_id.duplicated().any():
        raise ValueError("Duplicate expression cell lines: resolve explicitly before analysis.")
    X = expression.set_index("cell_line_id").apply(pd.to_numeric, errors="raise")
    if X.columns.duplicated().any() or np.isinf(X.to_numpy()).any():
        raise ValueError("Expression has duplicate gene names or infinite values.")
    selected = responses.loc[(responses.drug_id == str(drug)) & (responses.dataset == dataset)].copy()
    selected["LN_IC50"] = pd.to_numeric(selected.LN_IC50, errors="raise")
    valid = selected.loc[np.isfinite(selected.LN_IC50)]
    # Prespecified policy: average measured replicates within drug/dataset/cell line.
    y = valid.groupby("cell_line_id", sort=True).LN_IC50.mean()
    ids = X.index.intersection(y.index).sort_values()
    audit = {"drug_id": str(drug), "dataset": dataset,
             "response_rows": len(selected), "missing_response_rows": len(selected)-len(valid),
             "replicate_rows_collapsed": len(valid)-len(y),
             "response_cell_lines_without_expression": len(y.index.difference(X.index)),
             "matched_cell_lines": len(ids), "input_genes": X.shape[1]}
    if len(ids) < 10:
        raise ValueError(f"Only {len(ids)} matched cell lines; at least 10 required.")
    if y.loc[ids].nunique() < 2:
        raise ValueError("The response has no variation.")
    return X.loc[ids], y.loc[ids], audit


def coverage(expression, responses, dataset):
    """Rank drugs by matched unique cell-line count, never by model scores."""
    ids = set(clean_ids(expression.cell_line_id))
    r = responses.copy()
    r["cell_line_id"] = clean_ids(r.cell_line_id)
    r["drug_id"] = clean_ids(r.drug_id)
    r["LN_IC50"] = pd.to_numeric(r.LN_IC50, errors="raise")
    r = r.loc[(r.dataset == dataset) & r.cell_line_id.isin(ids) & np.isfinite(r.LN_IC50)]
    return (r.groupby("drug_id").cell_line_id.nunique().rename("matched_cell_lines")
            .reset_index().sort_values(["matched_cell_lines", "drug_id"], ascending=[False, True]))
