# Submission Checklist

## Repo

- `README.md` explains what the product is in the first screen
- Local run command works: `streamlit run streamlit_app.py`
- `requirements.txt` is present
- `LICENSE` is present
- `docs/TECHNICAL_IMPLEMENTATION.md` is present

## Product

- Streamlit app launches locally
- Protocol selector works
- Protocol detail page renders health score, chart, and risk signals
- Leaderboard renders
- Live refresh works for at least one protocol

## Avalanche Usage

- README clearly states Avalanche is the target ecosystem
- Glacier integration is visible in product and docs
- `.env.example` documents `GLACIER_API_KEY`

## Submission Assets Still Needed Outside Repo

- Final walkthrough video
- Submission form text
- Screenshots or short clips if the submission form asks for them

## Suggested Final Smoke Test

Run before submitting:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Check these flows manually:

1. Open the app and select a protocol.
2. Verify health score and TVL chart render.
3. Open leaderboard.
4. Click `Live Refresh Selected Protocol`.
5. Confirm the refreshed data section appears.
