# Heart Disease Prediction – Streamlit

Educational Logistic Regression classification demo using the supplied CSV dataset.

## Files
- `heart_disease_prediction_2026.csv` – dataset
- `train_model.py` – trains and evaluates the preprocessing + Logistic Regression pipeline, then saves the model
- `app.py` – Streamlit prediction interface
- `requirements.txt` – deployment dependencies

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# Windows: .venv\Scripts\activate

pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The training script creates `heart_disease_model.joblib`. Commit that file along with the other project files before deploying.

## Deploy on Streamlit Community Cloud
1. Push the project files to a GitHub repository.
2. Open https://share.streamlit.io and sign in with GitHub.
3. Choose **Create app**, select your repository and branch.
4. Set the main file path to `app.py`.
5. Click **Deploy**.

## Important
This is an educational demonstration, not a medical diagnostic tool. Do not use predictions for healthcare decisions.
