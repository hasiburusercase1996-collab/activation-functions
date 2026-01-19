## Motivation
Activation functions play a critical role in neural network training, influencing
non-linearity, gradient propagation, convergence, and stability.
This interactive tool was built to explore and compare activation functions
commonly used in deep learning and scientific machine learning (SciML),
including smooth, bounded, self-normalizing, and wavelet-based activations.

The app is intended for educational and research-oriented exploration and can be
extended for use in physics-informed neural networks (PINNs),
time-series forecasting models, and digital twin applications.

# Activation Function Explorer

Interactive Streamlit app to visualize classic, modern, and wavelet activation functions with their derivatives.

## Features
- Functions: Linear, Sigmoid, Tanh, ReLU/Leaky/ELU/SELU, Softplus/Softsign, Swish, Mish, GELU, Hard Sigmoid/Hard Swish, Morlet and Mexican Hat wavelets.
- Adjustable parameters (e.g., leak α, Swish β, SELU λ/α, wavelet ω/σ).
- Plotly charts for activation and derivative, LaTeX formulas, and sample values table.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to GitHub
1) Initialize repo (from this folder):
```bash
git init
git add app.py requirements.txt README.md
git commit -m "Add activation function explorer"
```
2) Create a public repo on GitHub (e.g., via the web UI under your account).  
3) Add remote and push:
```bash
git remote add origin https://github.com/hasiburusercase1996-collab/activation-functions.git
git branch -M main
git push -u origin main
```

## Deploy to Streamlit Community Cloud
1) Go to https://share.streamlit.io, sign in with GitHub, and connect the public repo.  
2) App settings:  
   - Repository: `hasiburusercase1996-collab/activation-functions`  
   - Branch: `main`  
   - Main file: `app.py`  
3) Click **Deploy**. Streamlit will install `requirements.txt` and build automatically.  
4) Share the generated URL; the app will be publicly accessible.

## File structure
- `app.py` — Streamlit UI, equations, Plotly charts.
- `requirements.txt` — Python dependencies (Streamlit, NumPy, Plotly).
- `README.md` — Usage and deployment steps.
