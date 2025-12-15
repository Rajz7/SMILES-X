# SMILES Processing & Classification Utilities — Fork Updates

This fork provides major improvements to the **SMILES-X** utilities for molecular data processing and classification.  
The updates focus on increasing **robustness**, preserving **data integrity**, and enhancing **uncertainty estimation** for more reliable model evaluation.

---

## 🧩 Overview

This modified version of the SMILES-X suite refines data preprocessing, feature encoding, and classification evaluation steps commonly used in molecular machine learning workflows.  
It introduces better input handling, flexible sequence encoding, and uncertainty-aware classification metrics.

---

## 🚀 Key Changes & Enhancements

### 1. `smiles_concat` — Input Validation
- Enhanced validation to prevent misuse when a **single SMILES string** is passed instead of a list.
- Produces clear **logging errors** for incorrect input types.
- Ensures concatenation only occurs for **lists or tuples** of SMILES sequences using `'j'`.

#### Example:
if isinstance(smiles, str):
logging.error(
"smiles_concat expected a list of SMILES per entry but got a STRING."
)
logging.error("Wrap your SMILES into a list, e.g. ['CCO']")

text

✅ **Result:** Reduced runtime errors and improved robustness during SMILES batch processing.

---

### 2. `int_vec_encode` — Dynamic Padding (No Truncation)
- Replaced **fixed-length truncation** with **dynamic padding** to handle variable-length molecule sequences.
- Pads all SMILES to the **maximum sequence length** in the batch.
- Preserves critical information for longer SMILES strings.
- Supports `'unk'` for unknown tokens and `'pad'` for padding.

#### Example:
pad_len = max_length - len(ismiles)
ismiles_tmp = ismiles + ['pad'] * pad_len

text

✅ **Advantages:**
- Prevents data loss from truncation.
- Compatible with **transformer-based models** (e.g., SMILES-BERT).
- Future-ready for attention mask integration.

---

### 3. `sigma_classification_metrics` — Monte Carlo Uncertainty Estimation
- Added **Monte Carlo simulation** for improved robustness in classification metrics.
- Injects **Gaussian noise** into predictions using predicted error estimates.
- Computes standard deviation across multiple stochastic runs for:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
  - PR-AUC  

#### Example:
pred_mc = pred + np.random.normal(0, err_pred, size=len(pred))
metrics_mat[i] = [acc, prec, rec, f1, pr_auc]

text

✅ **Impact:**
- Produces **uncertainty-aware metrics**.
- Enables statistically meaningful interpretation of model performance.

---

### 4. Code Cleanup & Maintenance
- Pruned **unnecessary imports** from `main.py`.
- Improved **readability**, **modularity**, and general code hygiene.
- Reduced clutter, making it easier to extend or integrate new models.

---

## ⚙️ Example Usage

Safe concatenation of SMILES sequences
smiles_concat(smiles_list)

Integer encoding with dynamic padding
int_vec_encode(tokenized_smiles_list, vocab)

Monte Carlo uncertainty estimation for classification metrics
sigma_classification_metrics(true, pred, err_pred, n_mc=1000)

text

---

## 📈 Impact Summary

| Improvement                    | Benefit                                                     |
|--------------------------------|-------------------------------------------------------------|
| Input validation in `smiles_concat` | Prevents misuse and clarifies error handling              |
| Dynamic padding in `int_vec_encode` | Preserves complete molecule information                   |
| Monte Carlo uncertainty metrics     | Enables uncertainty quantification in evaluation          |
| Code cleanup and reorganization     | Enhances maintainability and readability                  |

---

## 🧠 Future Directions
- Integrate attention-mask generation for transformer compatibility.  
- Extend uncertainty estimation to regression tasks.  
- Add configuration utilities for dynamic hyperparameter control.

---

## 🧾 Notes
These updates were implemented as part of a **research-oriented fork** to improve reliability, interpretability, and extendability in molecular machine learning workflows.

---

**Author:** [Your Name or Research Group]  
**Base Repository:** [Original SMILES-X Repository URL]  
**License:** MIT (or indicate applicable license)