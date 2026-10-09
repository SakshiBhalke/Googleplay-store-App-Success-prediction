# Google Play Store — App Success / Popularity Classification

A complete ML classification project predicting whether a Play Store app is
**Popular** (1,000,000+ installs) based on its listing metadata.

## Files
| File | Description |
|---|---|
| `Google_Play_Store_Success_Classification.ipynb` | Full notebook: cleaning, EDA, feature engineering, model comparison, evaluation, export |
| `app.py` | Streamlit app — enter app details, get a live prediction |
| `model.joblib` | Trained classifier (best of Logistic Regression / Decision Tree / Random Forest / Gradient Boosting) |
| `scaler.joblib` | Fitted `StandardScaler` for numeric features |
| `encoders.joblib` | Fitted `LabelEncoder`s for categorical features |
| `meta.joblib` | Column lists + best model name |
| `model_comparison.csv` | Metrics table for all 4 models |
| `googleplaystore_csv.xlsx` | Source dataset |
| `requirements.txt` | Python dependencies |

## Run the notebook
Open `Google_Play_Store_Success_Classification.ipynb` in Jupyter — it's already fully executed with all outputs/plots, or re-run top to bottom to regenerate `model.joblib` etc.

## Run the app
```bash
pip install -r requirements.txt
streamlit run app.py
```
Make sure `model.joblib`, `scaler.joblib`, `encoders.joblib`, and `meta.joblib` are in the same folder as `app.py` (they already are).

## Approach summary
- **Target:** `Popular = 1` if `Installs >= 1,000,000`, else `0` (~36% positive class).
- **Cleaning:** dropped 1 corrupted row, parsed `Installs`/`Reviews`/`Size`/`Price` to numeric, imputed `Rating` by category median, engineered `Days_Since_Update`.
- **Features:** `Category`, `Type`, `Content Rating`, `Rating`, `Reviews`, `Size_MB`, `Price`, `Days_Since_Update`.
- **Models compared:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting — best selected by F1 score.
- **Best result:** ~96% accuracy / ~0.94 F1 / ~0.99 ROC-AUC (see notebook for the full table — `Reviews` is the dominant feature since it correlates with install volume).
