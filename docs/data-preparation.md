# Data preparation

Obtain expression and fitted drug-response data from the official
[GDSC portal](https://www.cancerrxgene.org/) or its linked Sanger distribution.
Older bulk-download URLs may no longer work. No download URL or current release
has been assumed valid by this starter repository.

Keep original files outside version control. Record the source URL, release,
download date, screening dataset, drug identifier, expression measurement type,
and any mapping or normalization decisions. Do not apply an additional log
transform to an already log-transformed expression matrix or to LN_IC50.

## Normalized inputs

`expression.csv`: one row per cell line, first column `cell_line_id`, remaining
columns unique gene identifiers with numeric expression or empty missing values.

```csv
cell_line_id,GENE_A,GENE_B
1001,5.2,7.4
1002,6.1,
```

`responses.csv`: `cell_line_id`, `drug_id`, `dataset`, `LN_IC50` are required.
Other metadata columns are permitted. Use drug identifiers rather than names.

```csv
cell_line_id,drug_id,dataset,LN_IC50
1001,123,GDSC2,-0.2
1002,123,GDSC2,1.3
```

Examples illustrate schema only and are not real measurements. CSV, TSV (including
gzip-compressed CSV/TSV), and XLSX are supported. For Excel, the first sheet is read.

## Source-file adaptation checklist

1. Inspect the downloaded release's actual column definitions. Preserve the raw
   input and record its checksum. Do not assume that all releases share a schema.
2. If expression genes are rows, separate gene annotation columns and transpose
   the cell-line columns. Choose one documented stable gene identifier. Resolve
   duplicate probes/genes explicitly and document the aggregation policy.
3. Map identifiers to one common namespace (for example COSMIC IDs) in both
   tables. Do not join approximate cell-line names. Review unmapped samples.
   Numeric IDs ending in `.0` are normalized by this code; prefixes are not.
4. Rename response fields to the contract above and assign the screening dataset
   from verified source metadata. Keep GDSC1/GDSC2 separate.
5. Run coverage and freeze a drug choice before evaluating predictive scores.

Missing/nonfinite responses are dropped; measured replicates are averaged within
the chosen drug and dataset per cell line before CV. Duplicate expression cell
lines and infinite expression values are rejected. Invalid numeric strings raise
errors. The manifest records unmatched response cell lines and collapsed rows.
All sample IDs are sorted before splitting for reproducibility.

The pipeline does not fit expression preprocessing globally: missingness filters,
medians, variance filters, correlations, and scaling are learned in each training
fold. Do not externally preselect genes by association with all response values.
