# SMILES Processing & Classification Utilities — Fork Updates

## Overview
This fork introduces several improvements and bug fixes to the SMILES-X utility functions and visualization tools for molecular data. The updates enhance **robustness, flexibility, and uncertainty estimation** in the pipeline.

---

## Key Changes & Enhancements

### 1. `smiles_concat` Validation
- Added input validation to prevent misusage when a single SMILES string is provided instead of a list.
- Logs clear error messages for incorrect input types.
- Ensures only valid lists or tuples of SMILES are concatenated using `'j'`.

```python
if isinstance(smiles, str):
    logging.error("Wrap your SMILES into a list, e.g. ['CCO']")

### 2. int_vec_encode with Dynamic Padding

Replaced truncation with dynamic padding to the length of the longest SMILES.

Ensures no information loss for longer SMILES strings.

Converts unknown tokens to 'unk' and pads sequences with 'pad'.

pad_len = max_length - len(ismiles)
ismiles_tmp = ismiles + ['pad'] * pad_len


Prepares data for future transformer-based models (e.g., SMILES-BERT) with potential attention masks.

### 3. sigma_classification_metrics — Monte Carlo Uncertainty

Added Monte Carlo simulation to estimate uncertainty in classification metrics.

Adds Gaussian noise to predictions repeatedly and computes standard deviation of accuracy, precision, recall, F1, and PR-AUC.

Provides robust uncertainty estimates for classification outputs.

pred_mc = pred + np.random.normal(0, err_pred, size=len(pred))
metrics_mat[i] = [acc, prec, rec, f1, pr_auc]

### 4. Code Cleanup & Maintenance

Flagged unnecessary imports in main.py.


Impact

Increased robustness: prevents common input errors and ensures consistent encoding.

Better data handling: avoids information loss from truncation.

Uncertainty-aware metrics: enables better interpretation of model performance under prediction noise.

Readiness for advanced models: dynamic padding supports transformer-based architectures.

Usage

Use smiles_concat(smiles_list) for safe concatenation of SMILES sequences.

Use int_vec_encode(tokenized_smiles_list, vocab) for integer encoding with dynamic padding.

Use sigma_classification_metrics(true, pred, err_pred, n_mc=1000) to compute Monte Carlo uncertainty of classification metrics.