# ML Playground

Interactive visual revision tool for classic ML algorithms — see how decision
boundaries, fitted curves/surfaces, and clusters change live as you move
hyperparameter sliders.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501.

## Deploy live for free (GitHub → Streamlit Community Cloud)

1. Push this folder to a GitHub repository (keep the layout: `app.py`,
   `requirements.txt`, `algorithms/`, `data/` all at the repo root):

   ```bash
   git init
   git add app.py requirements.txt README.md algorithms/ data/
   git commit -m "ML Playground"
   git branch -M main
   git remote add origin https://github.com/<your-username>/ml-playground.git
   git push -u origin main
   ```

2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app → Deploy a public app from GitHub**, pick the repo,
   branch `main`, and main file path `app.py`.
4. Click **Deploy**. You get a public URL like
   `https://<your-username>-ml-playground.streamlit.app` — the app rebuilds
   automatically every time you push to GitHub.

Notes: the free tier sleeps after inactivity (first visit wakes it in ~30 s),
and `requirements.txt` is what the cloud installs — don't delete it.

## Layout

- `app.py` — page flow and Streamlit UI (all controls in the sidebar)
- `algorithms/notes.py` — revision notes, key equations, analogies
- `algorithms/models.py` — hyperparameter widgets + estimator builders
- `algorithms/datasets.py` — synthetic generators, bundled datasets, CSV cleaning
- `algorithms/plots.py` — matplotlib 2-D plots and plotly 3-D plots
- `data/` — bundled Kaggle-classic CSVs (Titanic, Penguins, Tips, Auto MPG,
  Pima Diabetes, Diamonds)
- `run_checks.py` — test harness (not needed to run the app)
