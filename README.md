# SMILES Processing & Classification Utilities — Fork Updates

## Overview
This fork introduces multiple improvements and bug fixes to the **SMILES-X** utility functions and visualization tools used for molecular data processing and classification.  
The changes focus on improving **robustness**, **data integrity**, and **uncertainty estimation**.

---

## Key Changes & Enhancements

### 1. `smiles_concat` — Input Validation
- Added validation to prevent misuse when a **single SMILES string** is passed instead of a list.
- Logs clear error messages for incorrect input types.
- Ensures only lists or tuples of SMILES are concatenated using `'j'`.

```python
if isinstance(smiles, str):
    logging.error(
        "smiles_concat expected a list of SMILES per entry but got a STRING."
    )
    logging.error("Wrap your SMILES into a list, e.g. ['CCO']")
2. int_vec_encode — Dynamic Padding (No Truncation)
Replaced fixed-length truncation with dynamic padding.

All SMILES are padded to the length of the longest sequence in the batch.

Prevents information loss for longer molecules.

Unknown tokens are mapped to 'unk', padding uses 'pad'.

python
Copy code
pad_len = max_length - len(ismiles)
ismiles_tmp = ismiles + ['pad'] * pad_len
This design also prepares the pipeline for transformer-based models
(e.g., SMILES-BERT), where attention masks may be required.

3. sigma_classification_metrics — Monte Carlo Uncertainty Estimation
Added Monte Carlo simulation to quantify uncertainty in classification metrics.

Gaussian noise is injected into predictions using predicted error estimates.

Computes the standard deviation of:

Accuracy

Precision

Recall

F1-score

PR-AUC

python
Copy code
pred_mc = pred + np.random.normal(0, err_pred, size=len(pred))
metrics_mat[i] = [acc, prec, rec, f1, pr_auc]
Provides more reliable performance assessment under prediction uncertainty.

4. Code Cleanup & Maintenance
Identified and flagged unnecessary imports in main.py.

Improved code readability and maintainability.

Reduced potential confusion during experimentation and extension.

Impact
Increased robustness: prevents common input errors in SMILES handling.

Improved data handling: eliminates truncation-induced information loss.

Uncertainty-aware evaluation: enables statistically meaningful interpretation of model performance.

Future-ready design: dynamic padding supports modern deep learning architectures.

Usage
python
Copy code
# Safe concatenation of SMILES sequences
smiles_concat(smiles_list)

# Integer encoding with dynamic padding
int_vec_encode(tokenized_smiles_list, vocab)

# Monte Carlo uncertainty estimation for classification metrics
sigma_classification_metrics(true, pred, err_pred, n_mc=1000)
Notes
These updates were implemented as part of a research-oriented fork to improve
model reliability, interpretability, and extensibility in molecular machine
learning workflows.