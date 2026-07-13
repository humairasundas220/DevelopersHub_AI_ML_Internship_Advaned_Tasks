# DevelopersHub_AI_ML_Internship_Advaned_Tasks
# Task 1: News Topic Classifier Using BERT

## Objective
Fine-tune BERT-base-uncased to classify news headlines into 4 categories: 
World, Sports, Business, and Science/Tech.

## Dataset
- **Source:** AG News Dataset
- **Size:** 120,000 training, 7,600 test samples
- **Task:** 4-class text classification

## Results

### Performance Metrics
- **Accuracy:** 94.08%
- **F1-Score (weighted):** 94.09%

### Per-Class Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| World | 97% | 94% | 95% |
| Sports | 98% | 99% | 99% |
| Business | 89% | 93% | 91% |
| Science/Tech | 93% | 91% | 92% |

## Methodology

### 1. Data Preprocessing
- Loaded AG News dataset from CSV
- Combined title + description into single text field
- Converted labels to 0-indexed format

### 2. Tokenization
- Used BERT-base-uncased tokenizer
- Max sequence length: 128 tokens
- Applied padding and truncation

### 3. Model Fine-tuning
- Base Model: BERT-base-uncased (110M parameters, 12 layers)
- Epochs: 3
- Batch size: 16
- Learning rate: 5e-5
- Optimizer: AdamW
- Training time: 2-3 hours on T4 GPU

### 4. Evaluation
- Metrics: Accuracy, F1-score, Confusion Matrix
- No severe overfitting detected
- Stable validation loss across epochs

## Key Findings

1. **Sports category** is easiest to classify (99% recall)
2. **Business-Science/Tech overlap** due to topic similarity
3. Model achieves **near-human level accuracy** (94%+)
4. **Production-ready** for deployment

## Files Included
- `Task1_BERT_News_Classifier.ipynb` — Complete training notebook
- `confusion_matrix.png` — Evaluation visualization
- `README.md` — This file

## How to Use

### Option 1: Run in Colab
1. Open `Task1_BERT_News_Classifier.ipynb` in Google Colab
2. Run all cells
3. Model trains and evaluates automatically

### Option 2: Deploy with Gradio
```python
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and create interface
# See notebook for complete deployment code
```

## Technologies Used
- **PyTorch** — Deep learning framework
- **Transformers** — Hugging Face BERT
- **Scikit-learn** — Evaluation metrics
- **Gradio** — Web deployment
- **Pandas, NumPy** — Data processing

## Skills Demonstrated
✅ NLP using Transformers  
✅ Transfer learning & fine-tuning  
✅ Text classification  
✅ Model evaluation (accuracy, F1-score, confusion matrix)  
✅ Deployment with Gradio  
✅ GPU training optimization  

## Future Improvements
- Deploy to Hugging Face Spaces for persistent public access
- Experiment with larger models (BERT-large, RoBERTa)
- Add multilingual support
- Fine-tune on domain-specific news corpus


# Task2 Telco Customer Churn — End-to-End ML Pipeline

## Objective
Build a reusable, production-ready machine learning pipeline that predicts whether a telecom customer will churn, using scikit-learn's `Pipeline` and `ColumnTransformer` API to bundle preprocessing (scaling + encoding) and modeling into a single deployable object. Two classifiers — Logistic Regression and Random Forest — are trained and tuned with `GridSearchCV`, and the best resulting pipeline is exported with `joblib` for reuse.

## Dataset
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample dataset) — 7,043 customers, 21 columns covering demographics, account details, subscribed services, and the `Churn` label. Class distribution: 73.5% No Churn / 26.5% Churn.

## Methodology / Approach
1. **Data loading & cleaning** — converted `TotalCharges` from text to numeric (11 blank values, from customers with 0 tenure, imputed with the median), dropped the `customerID` identifier column, and encoded the target as binary.
2. **Preprocessing pipeline** — a `ColumnTransformer` applying median imputation + `StandardScaler` to 4 numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`), and most-frequent imputation + `OneHotEncoder` to 15 categorical features.
3. **Modeling** — two full pipelines (`preprocessor` + classifier) built for Logistic Regression and Random Forest, on an 80/20 stratified train/test split (5,634 / 1,409 customers).
4. **Hyperparameter tuning** — 5-fold `GridSearchCV` optimizing ROC-AUC:
   - Logistic Regression: `C` ∈ {0.01, 0.1, 1, 10}, `solver` ∈ {lbfgs, liblinear} — best: `C=10`, `solver=lbfgs`
   - Random Forest: `n_estimators` ∈ {100, 200}, `max_depth` ∈ {None, 10, 20}, `min_samples_split` ∈ {2, 5} — best: `n_estimators=200`, `max_depth=10`, `min_samples_split=5`
5. **Evaluation** — accuracy, precision, recall, F1, and ROC-AUC on the held-out test set; confusion matrices, ROC curve comparison, and Random Forest feature importances plotted.
6. **Export** — the best-performing pipeline (Logistic Regression, preprocessing + model together) saved as `churn_pipeline.joblib`, then reloaded and verified to predict correctly on raw test rows.

## Key Results / Observations

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.806 | 0.657 | **0.559** | **0.604** | **0.841** |
| Random Forest | 0.800 | 0.656 | 0.516 | 0.578 | 0.838 |

- **Logistic Regression was selected as the best model** — it outperforms Random Forest on recall, F1, and ROC-AUC, despite both models achieving nearly identical ROC-AUC (0.84), indicating similar overall discriminative power but a better default decision threshold for Logistic Regression.
- On the test set, Logistic Regression correctly identifies 209 of 374 actual churners (56%), versus 193 of 374 (52%) for Random Forest.
- **Top churn drivers** (from Random Forest feature importance): `tenure`, `TotalCharges`, `Contract_Month-to-month`, and `MonthlyCharges` — churn is concentrated among newer customers on flexible, no-commitment contracts paying higher monthly bills, with lack of Online Security / Tech Support add-ons as secondary risk factors.
- **Business recommendation:** target retention offers (contract-upgrade incentives, security/support bundles) at customers matching this at-risk profile.
- **Limitation:** current recall (~56%) still misses a meaningful share of churners; future work could explore threshold tuning or class-weighting/SMOTE to catch more churners at the cost of some extra false positives.
- Because preprocessing is embedded in the exported pipeline object, the resulting `churn_pipeline.joblib` file can be loaded by any downstream service and called directly on new, raw customer records.

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib kagglehub
jupyter notebook Telco_Churn_Pipeline.ipynb
```

## How to Reuse the Exported Pipeline
```python
import joblib
pipeline = joblib.load("churn_pipeline.joblib")
predictions = pipeline.predict(new_raw_customer_dataframe)
```
# Task 3 Multimodal Housing Price Prediction — Images + Tabular Data

## Objective
Predict housing sale prices by combining structured tabular data (bedrooms, bathrooms, square footage, city) with visual features extracted from house photos using a pretrained CNN. This multimodal approach tests whether visual information (curb appeal, condition, architectural style) adds predictive power beyond what tabular attributes alone can capture, evaluated using MAE and RMSE.

## Dataset
[House Prices and Images - SoCal](https://www.kaggle.com/datasets/ted8080/house-prices-and-images-socal) (Kaggle) — 15,474 Southern California home listings, each with one photo and matching tabular data (price, bed, bath, sqft, city). After removing 3 rows with data-entry errors (implausible bathroom counts), 15,471 rows were used.

## Methodology / Approach
1. **Data loading & cleaning** — linked each tabular row to its image via `image_id` (all 15,474 images matched, 0 missing); removed 3 rows with physically implausible bathroom counts (e.g. 36 bathrooms in a 1,229 sqft home).
2. **Train/test split** — 80/20 split (12,376 / 3,095 houses), performed *before* any feature extraction or scaling to prevent data leakage.
3. **Image feature extraction (CNN)** — used ResNet50 pretrained on ImageNet as a fixed feature extractor (`include_top=False`, `pooling="avg"`), producing a 2,048-dimensional feature vector per house photo. No CNN training was needed since we reused ImageNet-learned visual features via transfer learning.
4. **Tabular preprocessing** — numeric features (`bed`, `bath`, `sqft`) scaled with `StandardScaler`; `citi` (411 unique cities) one-hot encoded. Both fit only on the training set.
5. **Feature fusion** — image features (2,048-dim), numeric tabular (3-dim), and city one-hot (411-dim) concatenated into a single 2,462-dimensional feature vector per house.
6. **Modeling** — Random Forest Regressor trained on the fused features to predict `log(1 + price)` (log-transformed to correct for the right-skewed price distribution), tuned via `GridSearchCV` over `n_estimators`, `max_depth`, and `min_samples_split`.
7. **Evaluation** — predictions converted back to dollar scale (`expm1`) and evaluated with MAE and RMSE, compared directly against a tabular-only baseline (same model, same hyperparameters, without image features) to isolate the actual contribution of the image modality.
8. **Export** — the trained Random Forest, numeric scaler, and city encoder were saved with `joblib` for reuse (the CNN itself doesn't need saving, since it's the standard pretrained ResNet50).

## Key Results / Observations

| Model | MAE | RMSE |
|---|---|---|
| Tabular-only baseline | 
| Multimodal (image + tabular) | 

- Best Random Forest hyperparameters: `[FILL IN from Step 12 best_params_]`
- Adding image features [**improved / did not meaningfully improve**] prediction accuracy relative to the tabular-only baseline — see the feature-group importance chart for how much predictive weight came from image features vs. numeric tabular vs. city.
- MAE represents `[FILL IN]`% of the average test-set house price (~$[FILL IN mean price]).
- *(Add 1-2 sentences here once you see the predicted-vs-actual and residual plots — e.g. whether errors are roughly balanced across price ranges, or whether the model under/over-predicts for expensive homes specifically.)*

## Repository Structure
