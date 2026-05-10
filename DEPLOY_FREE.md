# Free Deployment Plan

This project is now prepared for a free live deployment path using a lightweight Streamlit bundle.

## Why this is needed
- Full training artifacts (`models/`) are very large, and some files exceed 100 MiB.
- Free Git hosting (e.g., GitHub normal push) blocks files larger than 100 MiB.
- Free app hosts work better with small dependency footprints.

## What to do
1. Regenerate deploy bundle:
   ```bash
   python prepare_deploy_bundle.py
   ```
2. Use only `deploy_streamlit_free/` for cloud deployment.
3. Create a new GitHub repository and upload the contents of `deploy_streamlit_free/` (not the whole project).
4. Deploy that repo on Streamlit Community Cloud (free).

## Local smoke test before cloud deploy
```bash
cd deploy_streamlit_free
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- This free bundle is dashboard-only (serves predictions and metrics already produced by the pipeline).
- Retraining is still done locally in the full project.
- Whenever you retrain, rerun `python prepare_deploy_bundle.py` and push updated bundle files.

