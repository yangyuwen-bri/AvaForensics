# Deployment

## Recommended Path

The simplest hosting path for the current AvaForensics MVP is Streamlit Community Cloud.

Why:

- the product is already a Streamlit app
- no separate backend service is required
- the app can run from the GitHub repository directly

Official docs:

- [Streamlit Community Cloud overview](https://docs.streamlit.io/deploy/streamlit-community-cloud)
- [Deploy your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)
- [Secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)

## Before Deploying

Make sure these files are in the repository:

- `streamlit_app.py`
- `requirements.txt`
- `avaforensics/`
- `avax_data/early_features.csv`
- `avax_data/features_summary.csv`
- `avax_data/protocols_labeled.csv`
- `avax_data/combined_features.csv` if you want price enrichment shown
- `avax_data/onchain_features.csv` if you want on-chain enrichment shown

The app does **not** require local `tvl_*.csv` files to be committed. If they are missing in the deployed repo, the app will fall back to DeFiLlama for TVL history.

## Streamlit Community Cloud Settings

When creating the app:

- Repository: your GitHub repo
- Branch: `main`
- Main file path: `streamlit_app.py`
- Python version: `3.11`

Use Python `3.11` because the current dependency set is tested locally on the repo and is compatible with the MVP stack.

## Secrets

If you want Avalanche official live on-chain summaries in production, add this in Streamlit Cloud Secrets:

```toml
GLACIER_API_KEY = "your_real_key_here"
```

Without this secret:

- the app still runs
- baseline scoring still works
- live TVL refresh still works
- Avalanche Glacier live on-chain summary will be disabled

## Local Smoke Test Before Push

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Check:

1. app homepage loads
2. protocol selector works
3. protocol detail chart renders
4. leaderboard renders
5. live refresh works

## Optional Alternative Hosts

If you do not want to use Streamlit Community Cloud, the next-best option is Render.

That path is reasonable, but it adds more deployment configuration than this MVP currently needs.
