import math
from typing import Callable, Dict, List

import numpy as np
import plotly.graph_objects as go
import streamlit as st


def numeric_derivative(fn: Callable[[np.ndarray, Dict[str, float]], np.ndarray],
                       x: np.ndarray,
                       params: Dict[str, float],
                       eps: float = 1e-4) -> np.ndarray:
    """Central difference fallback."""
    return (fn(x + eps, params) - fn(x - eps, params)) / (2 * eps)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


SQRT_2_OVER_PI = math.sqrt(2 / math.pi)


ActivationConfig = Dict[str, object]

activations: Dict[str, ActivationConfig] = {
    "Linear": {
        "latex": r"f(x)=x",
        "dlatex": r"f'(x)=1",
        "params": [],
        "fn": lambda x, _: x,
        "dfn": lambda x, _: np.ones_like(x),
    },
    "Sigmoid": {
        "latex": r"\sigma(x)=\frac{1}{1+e^{-x}}",
        "dlatex": r"\sigma'(x)=\sigma(x)(1-\sigma(x))",
        "params": [],
        "fn": lambda x, _: sigmoid(x),
        "dfn": lambda x, _: sigmoid(x) * (1 - sigmoid(x)),
    },
    "Tanh": {
        "latex": r"\tanh(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}",
        "dlatex": r"\tanh'(x)=1-\tanh^2(x)",
        "params": [],
        "fn": lambda x, _: np.tanh(x),
        "dfn": lambda x, _: 1 - np.tanh(x) ** 2,
    },
    "ReLU": {
        "latex": r"f(x)=\max(0,x)",
        "dlatex": r"f'(x)=\begin{cases}0, & x\le 0\\1, & x>0\end{cases}",
        "params": [],
        "fn": lambda x, _: np.maximum(0, x),
        "dfn": lambda x, _: (x > 0).astype(float),
    },
    "Leaky ReLU": {
        "latex": r"f(x)=\begin{cases}\alpha x,&x<0\\x,&x\ge0\end{cases}",
        "dlatex": r"f'(x)=\begin{cases}\alpha,&x<0\\1,&x\ge0\end{cases}",
        "params": [
            {"name": "alpha", "label": "Leak α", "min": 0.0, "max": 1.0, "step": 0.01, "value": 0.01},
        ],
        "fn": lambda x, p: np.where(x >= 0, x, p["alpha"] * x),
        "dfn": lambda x, p: np.where(x >= 0, 1.0, p["alpha"]),
    },
    "ELU": {
        "latex": r"f(x)=\begin{cases}x,&x>0\\\alpha(e^{x}-1),&x\le0\end{cases}",
        "dlatex": r"f'(x)=\begin{cases}1,&x>0\\\alpha e^{x},&x\le0\end{cases}",
        "params": [
            {"name": "alpha", "label": "α", "min": 0.1, "max": 5.0, "step": 0.1, "value": 1.0},
        ],
        "fn": lambda x, p: np.where(x > 0, x, p["alpha"] * (np.exp(x) - 1)),
        "dfn": lambda x, p: np.where(x > 0, 1.0, p["alpha"] * np.exp(x)),
    },
    "SELU": {
        "latex": r"f(x)=\lambda\begin{cases}x,&x>0\\\alpha(e^{x}-1),&x\le0\end{cases}",
        "dlatex": r"f'(x)=\lambda\begin{cases}1,&x>0\\\alpha e^{x},&x\le0\end{cases}",
        "params": [
            {"name": "lambda_", "label": "λ", "min": 0.5, "max": 2.0, "step": 0.01, "value": 1.0507},
            {"name": "alpha", "label": "α", "min": 0.5, "max": 3.0, "step": 0.01, "value": 1.67326},
        ],
        "fn": lambda x, p: np.where(
            x > 0, p["lambda_"] * x, p["lambda_"] * p["alpha"] * (np.exp(x) - 1)
        ),
        "dfn": lambda x, p: np.where(x > 0, p["lambda_"], p["lambda_"] * p["alpha"] * np.exp(x)),
    },
    "Softplus": {
        "latex": r"f(x)=\ln(1+e^{x})",
        "dlatex": r"f'(x)=\sigma(x)",
        "params": [],
        "fn": lambda x, _: softplus(x),
        "dfn": lambda x, _: sigmoid(x),
    },
    "Softsign": {
        "latex": r"f(x)=\frac{x}{1+|x|}",
        "dlatex": r"f'(x)=\frac{1}{(1+|x|)^{2}}",
        "params": [],
        "fn": lambda x, _: x / (1 + np.abs(x)),
        "dfn": lambda x, _: 1 / (1 + np.abs(x)) ** 2,
    },
    "Swish": {
        "latex": r"f(x)=x\cdot\sigma(\beta x)",
        "dlatex": r"f'(x)=\sigma(\beta x)+x\,\sigma(\beta x)(1-\sigma(\beta x))\beta",
        "params": [
            {"name": "beta", "label": "β", "min": 0.0, "max": 5.0, "step": 0.05, "value": 1.0},
        ],
        "fn": lambda x, p: x * sigmoid(p["beta"] * x),
        "dfn": lambda x, p: (
            sigmoid(p["beta"] * x)
            + x * sigmoid(p["beta"] * x) * (1 - sigmoid(p["beta"] * x)) * p["beta"]
        ),
    },
    "Mish": {
        "latex": r"f(x)=x\tanh(\ln(1+e^{x}))",
        "dlatex": r"f'(x)=\tanh(\text{sp})+x\,\sigma(x)\big(1-\tanh^{2}(\text{sp})\big),~\text{sp}=\ln(1+e^{x})",
        "params": [],
        "fn": lambda x, _: x * np.tanh(softplus(x)),
        "dfn": lambda x, _: np.tanh(softplus(x)) + x * sigmoid(x) * (1 - np.tanh(softplus(x)) ** 2),
    },
    "GELU": {
        "latex": r"f(x)=0.5\,x\big(1+\tanh(\sqrt{2/\pi}(x+0.044715x^{3}))\big)",
        "dlatex": r"f'(x)=0.5\big(1+\tanh(u)\big)+0.5x(1-\tanh^{2}(u))\sqrt{2/\pi}(1+0.134145x^{2}),~u=\sqrt{2/\pi}(x+0.044715x^{3})",
        "params": [],
        "fn": lambda x, _: 0.5 * x * (1 + np.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x**3))),
        "dfn": lambda x, _: (
            0.5 * (1 + np.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x**3)))
            + 0.5
            * x
            * (1 - np.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x**3)) ** 2)
            * SQRT_2_OVER_PI
            * (1 + 0.134145 * x**2)
        ),
    },
    "Hard Sigmoid": {
        "latex": r"f(x)=\text{clip}(m x + b, 0, 1),~m=0.2,~b=0.5",
        "dlatex": r"f'(x)=\begin{cases}0,&x\le -2.5\\m,&-2.5<x<2.5\\0,&x\ge2.5\end{cases}",
        "params": [
            {"name": "slope", "label": "Slope m", "min": 0.05, "max": 1.0, "step": 0.05, "value": 0.2},
            {"name": "bias", "label": "Bias b", "min": 0.0, "max": 1.0, "step": 0.05, "value": 0.5},
        ],
        "fn": lambda x, p: np.clip(p["slope"] * x + p["bias"], 0, 1),
        "dfn": lambda x, p: np.where(
            (x > (-p["bias"] / p["slope"])) & (x < ((1 - p["bias"]) / p["slope"])),
            p["slope"],
            0.0,
        ),
    },
    "Hard Swish": {
        "latex": r"f(x)=x\cdot\text{clip}(m x + b,0,1),~m=0.2,~b=0.5",
        "dlatex": r"f'(x)=\text{clip}(m x+b,0,1)+x\,m\,[|m x+b|<1]",
        "params": [
            {"name": "slope", "label": "Slope m", "min": 0.05, "max": 1.0, "step": 0.05, "value": 0.2},
            {"name": "bias", "label": "Bias b", "min": 0.0, "max": 1.0, "step": 0.05, "value": 0.5},
        ],
        "fn": lambda x, p: x * np.clip(p["slope"] * x + p["bias"], 0, 1),
        "dfn": lambda x, p: (
            np.clip(p["slope"] * x + p["bias"], 0, 1)
            + x * p["slope"] * (((p["slope"] * x + p["bias"]) > 0) & ((p["slope"] * x + p["bias"]) < 1))
        ),
    },
    "Morlet Wavelet": {
        "latex": r"\psi(x)=e^{-\frac{x^{2}}{2\sigma^{2}}}\cos(\omega x)",
        "dlatex": r"\psi'(x)=e^{-\frac{x^{2}}{2\sigma^{2}}}\big(-\omega\sin(\omega x)-\frac{x}{\sigma^{2}}\cos(\omega x)\big)",
        "params": [
            {"name": "omega", "label": "Frequency ω", "min": 0.5, "max": 10.0, "step": 0.1, "value": 1.75},
            {"name": "sigma", "label": "Width σ", "min": 0.2, "max": 5.0, "step": 0.1, "value": 1.0},
        ],
        "fn": lambda x, p: np.exp(-(x**2) / (2 * p["sigma"] ** 2)) * np.cos(p["omega"] * x),
        "dfn": lambda x, p: np.exp(-(x**2) / (2 * p["sigma"] ** 2))
        * (-p["omega"] * np.sin(p["omega"] * x) - (x / (p["sigma"] ** 2)) * np.cos(p["omega"] * x)),
    },
    "Mexican Hat Wavelet": {
        "latex": r"\psi(x)=\big(1-\frac{x^{2}}{\sigma^{2}}\big)e^{-\frac{x^{2}}{2\sigma^{2}}}",
        "dlatex": r"\psi'(x)=-\frac{x}{\sigma^{2}}e^{-\frac{x^{2}}{2\sigma^{2}}}\Big(3-\frac{x^{2}}{\sigma^{2}}\Big)",
        "params": [
            {"name": "sigma", "label": "Width σ", "min": 0.2, "max": 5.0, "step": 0.1, "value": 1.0},
        ],
        "fn": lambda x, p: (1 - (x**2) / (p["sigma"] ** 2)) * np.exp(-(x**2) / (2 * p["sigma"] ** 2)),
        "dfn": lambda x, p: -(
            x / (p["sigma"] ** 2)
        ) * np.exp(-(x**2) / (2 * p["sigma"] ** 2)) * (3 - (x**2) / (p["sigma"] ** 2)),
    },
}


def get_params(conf: ActivationConfig) -> Dict[str, float]:
    params: Dict[str, float] = {}
    for spec in conf["params"]:  # type: ignore[index]
        params[spec["name"]] = st.sidebar.slider(
            spec["label"],
            min_value=float(spec["min"]),
            max_value=float(spec["max"]),
            value=float(spec["value"]),
            step=float(spec["step"]),
        )
    return params


st.set_page_config(page_title="Activation Function Explorer", layout="wide")
st.title("Activation Function Explorer")
st.caption(
    "Interactive visualizer for classic, modern, and wavelet activation functions, with derivatives."
)

with st.sidebar:
    st.header("Controls")
    selected = st.selectbox("Activation", list(activations.keys()))
    x_min, x_max = st.slider("x range", -10.0, 10.0, (-6.0, 6.0), step=0.1)
    num_points = st.slider("Resolution (samples)", 200, 2000, 600, step=50)
    show_values = st.checkbox("Show sample values", value=True)
    params = get_params(activations[selected])

conf = activations[selected]
x = np.linspace(x_min, x_max, num_points)
y = conf["fn"](x, params)  # type: ignore[arg-type]
dy = conf["dfn"](x, params) if conf.get("dfn") else numeric_derivative(conf["fn"], x, params)  # type: ignore[arg-type]

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, name="Activation", line=dict(width=3)))
fig.add_trace(
    go.Scatter(x=x, y=dy, name="Derivative", line=dict(width=2, dash="dash"))
)
fig.update_layout(
    title=f"{selected} function and derivative",
    xaxis_title="x",
    yaxis_title="f(x)",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

col1, col2 = st.columns([2, 1])
with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Equations")
    st.latex(conf["latex"])  # type: ignore[index]
    st.latex(conf["dlatex"])  # type: ignore[index]
    if conf["params"]:  # type: ignore[index]
        st.markdown("**Parameters**")
        for k, v in params.items():
            st.write(f"{k} = {v}")

if show_values:
    st.subheader("Sample values")
    sample_idx = np.linspace(0, num_points - 1, min(40, num_points), dtype=int)
    sample_x = x[sample_idx]
    sample_y = y[sample_idx]
    sample_dy = dy[sample_idx]
    st.dataframe(
        {
            "x": np.round(sample_x, 3),
            "f(x)": np.round(sample_y, 6),
            "f'(x)": np.round(sample_dy, 6),
        },
        use_container_width=True,
        hide_index=True,
    )

st.markdown(
    """
**How to run:** `streamlit run app.py`  
Drag the sliders to adjust the x-range, resolution, and any function parameters. Wavelet
options (Morlet and Mexican Hat) include width and frequency controls.
"""
)
