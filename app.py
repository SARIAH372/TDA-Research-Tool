

    
            
            
                
            

            
          
               
        
   
       
        

    
        
                
  # app.py
import time
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample

from topo import (
    SHAPES_2D, SHAPES_3D, make_dataset_mixed,
    persistence_diagrams, finite_bars,
    fit_imagers_multiscale, diagram_to_pis,
    tda_feature_block,
    prototype_diagrams, distances_to_prototypes,
    density_filtration_summaries,
)

st.set_page_config(page_title="TDA Tool", layout="wide")
st.title("TDA Tool")

plt.close("all")

if "training" not in st.session_state:
    st.session_state.training = False
if "trained" not in st.session_state:
    st.session_state.trained = False
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False
if "last_train" not in st.session_state:
    st.session_state.last_train = None

def _show(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

def plot_points(P, title):
    P = np.asarray(P)
    if P.shape[1] == 2:
        fig, ax = plt.subplots()
        ax.scatter(P[:, 0], P[:, 1], s=12)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        return fig
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=10)
    ax.set_title(title)
    return fig

def plot_diagram(dgm, title):
    bars = finite_bars(dgm)
    fig, ax = plt.subplots()
    if len(bars) == 0:
        ax.set_title(title)
        return fig
    b = bars[:, 0]
    d = bars[:, 1]
    lim = max(1.0, float(np.max(d)))
    ax.scatter(b, d, s=18)
    ax.plot([0, lim], [0, lim], linestyle="--")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title(title)
    return fig

def make_model(kind, seed):
    if kind == "LogReg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=800, multi_class="auto", random_state=int(seed)))
        ])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=600, random_state=int(seed)))
    ])

def fit_ensemble(kind, Xtr, ytr, n_models=5, seed=7):
    rng = np.random.default_rng(seed)
    models = []
    for _ in range(int(n_models)):
        sk = int(rng.integers(0, 10_000))
        Xb, yb = resample(Xtr, ytr, replace=True, random_state=sk)
        m = make_model(kind, sk)
        m.fit(Xb, yb)
        models.append(m)
    return models

def ensemble_predict(models, X):
    probs = np.stack([m.predict_proba(X) for m in models], axis=0)
    return probs.mean(axis=0), probs.std(axis=0)

def _maybe_stop():
    if st.session_state.stop_requested:
        st.session_state.training = False
        st.session_state.trained = False
        st.session_state.stop_requested = False
        st.stop()

with st.sidebar:
    n_samples = st.slider("n_samples", 50, 800, 200, step=50)
    n_points = st.slider("n_points", 40, 260, 120, step=10)
    p_3d = st.slider("p_3d", 0.10, 0.80, 0.45, step=0.05)
    noise_2d = st.slider("noise_2d", 0.0, 0.20, 0.03, step=0.01)
    noise_3d = st.slider("noise_3d", 0.0, 0.20, 0.02, step=0.01)
    seed = st.number_input("seed", value=7, step=1)

    maxdim = st.selectbox("maxdim", [1, 2], index=1)
    grid_len = st.select_slider("grid_len", options=[32, 64, 96], value=64)
    topk = st.select_slider("topk", options=[6, 8, 10, 12], value=8)
    k_levels = st.select_slider("k_levels", options=[2, 3, 4], value=3)

    pixel_sizes = st.multiselect("pi_channels", options=[0.03, 0.05, 0.08, 0.10], default=[0.05])
    if not pixel_sizes:
        pixel_sizes = [0.05]

    use_density = st.checkbox("use_density", value=False)
    dens_k = st.select_slider("dens_k", options=[6, 8, 10, 12], value=10)

    metric = st.selectbox("diagram_metric", ["wasserstein", "sliced"], index=1)
    proto_cap = st.select_slider("proto_cap", options=[10, 15, 20, 25], value=15)

    model_kind = st.selectbox("model_kind", ["LogReg", "MLP"], index=0)
    ens = st.select_slider("ensemble", options=[3, 5, 7], value=5)

tabs = st.tabs(["Explore", "Train", "Predict"])

with tabs[0]:
    keys2 = list(SHAPES_2D.keys())
    keys3 = list(SHAPES_3D.keys())
    pick = st.selectbox("shape", keys2 + keys3)

    if pick in SHAPES_2D:
        pts = SHAPES_2D[pick](n=int(n_points), noise=float(noise_2d), seed=int(seed) + 101)
    else:
        pts = SHAPES_3D[pick](n=int(n_points), noise=float(noise_3d), seed=int(seed) + 202)

    _show(plot_points(pts, f"{pick} dim={pts.shape[1]}"))
    dg = persistence_diagrams(pts, maxdim=int(maxdim))
    _show(plot_diagram(dg[0], "H0"))
    _show(plot_diagram(dg[1], "H1"))
    if int(maxdim) >= 2:
        _show(plot_diagram(dg[2], "H2"))

with tabs[1]:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        start = st.button("train", disabled=st.session_state.training)
    with col2:
        stop = st.button("stop", disabled=not st.session_state.training)
    with col3:
        reset = st.button("reset", disabled=st.session_state.training)

    if reset:
        for k in ["models", "class_names", "fit_pack", "last_train"]:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state.training = False
        st.session_state.trained = False
        st.session_state.stop_requested = False

    if stop:
        st.session_state.stop_requested = True

    if start and not st.session_state.training:
        st.session_state.training = True
        st.session_state.trained = False
        st.session_state.stop_requested = False

        t0 = time.time()
        pcs, y, class_names = make_dataset_mixed(
            n_samples=int(n_samples),
            n_points=int(n_points),
            noise_2d=float(noise_2d),
            noise_3d=float(noise_3d),
            p_3d=float(p_3d),
            seed=int(seed),
        )

        prog = st.progress(0.0)
        txt = st.empty()

        dgms = []
        for i, p in enumerate(pcs):
            _maybe_stop()
            dgms.append(persistence_diagrams(p, maxdim=int(maxdim)))
            prog.progress((i + 1) / max(1, len(pcs)))
            txt.write(f"diagrams: {i+1}/{len(pcs)}")

        pim_h1, pim_h2 = fit_imagers_multiscale(dgms, pixel_sizes=tuple(pixel_sizes))

        n_classes = int(np.max(y)) + 1
        protos_h1 = prototype_diagrams(dgms, y, n_classes=n_classes, dim=1, cap_per_class=int(proto_cap), seed=int(seed), metric=str(metric))
        protos_h2 = prototype_diagrams(dgms, y, n_classes=n_classes, dim=2, cap_per_class=int(proto_cap), seed=int(seed) + 1, metric=str(metric)) if int(maxdim) >= 2 else [np.zeros((0, 2), dtype=np.float32) for _ in range(n_classes)]

        Xfeat = []
        for i, (p, dg) in enumerate(zip(pcs, dgms)):
            _maybe_stop()
            pi1 = diagram_to_pis(pim_h1, dg[1])
            pi2 = diagram_to_pis(pim_h2, dg[2]) if int(maxdim) >= 2 else np.zeros_like(pi1)

            hard = tda_feature_block(dg, grid_len=int(grid_len), topk=int(topk), k_levels=int(k_levels))

            d1 = distances_to_prototypes(dg[1], protos_h1, metric=str(metric), seed=int(seed) + 100)
            d2 = distances_to_prototypes(dg[2], protos_h2, metric=str(metric), seed=int(seed) + 200) if int(maxdim) >= 2 else np.zeros_like(d1)

            dens = density_filtration_summaries(p, fracs=(0.5, 0.8, 1.0), k=int(dens_k), maxdim=int(maxdim)) if use_density else np.zeros((0,), dtype=np.float32)

            feat = np.concatenate([pi1, pi2, hard, d1, d2, dens], axis=0).astype(np.float32)
            Xfeat.append(feat)

        X = np.vstack(Xfeat).astype(np.float32)

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=int(seed), stratify=y)
        models = fit_ensemble(model_kind, Xtr, ytr, n_models=int(ens), seed=int(seed))
        p_mean, _ = ensemble_predict(models, Xte)
        pred = np.argmax(p_mean, axis=1)
        acc = accuracy_score(yte, pred)

        st.session_state.models = models
        st.session_state.class_names = class_names
        st.session_state.fit_pack = {
            "maxdim": int(maxdim),
            "pim_h1": pim_h1,
            "pim_h2": pim_h2,
            "protos_h1": protos_h1,
            "protos_h2": protos_h2,
            "grid_len": int(grid_len),
            "topk": int(topk),
            "k_levels": int(k_levels),
            "metric": str(metric),
            "use_density": bool(use_density),
            "dens_k": int(dens_k),
        }
        st.session_state.last_train = {"acc": float(acc), "n": int(len(y)), "d": int(X.shape[1]), "seconds": float(time.time() - t0)}
        st.session_state.training = False
        st.session_state.trained = True

    if st.session_state.last_train is not None:
        st.write(st.session_state.last_train)

with tabs[2]:
    mode = st.radio("input", ["generate", "upload"], horizontal=True)
    pts = None

    if mode == "generate":
        keys2 = list(SHAPES_2D.keys())
        keys3 = list(SHAPES_3D.keys())
        pick = st.selectbox("shape_pred", keys2 + keys3)
        sd = st.number_input("seed_pred", value=999, step=1)
        if pick in SHAPES_2D:
            pts = SHAPES_2D[pick](n=int(n_points), noise=float(noise_2d), seed=int(sd))
        else:
            pts = SHAPES_3D[pick](n=int(n_points), noise=float(noise_3d), seed=int(sd))
    else:
        up = st.file_uploader("csv", type=["csv"])
        if up is not None:
            data = np.loadtxt(up, delimiter=",")
            if data.ndim == 1:
                data = data.reshape(-1, data.shape[0])
            if data.shape[1] in (2, 3):
                pts = data.astype(np.float32)

    if pts is not None:
        _show(plot_points(pts, f"dim={pts.shape[1]}"))
        dg = persistence_diagrams(pts, maxdim=int(maxdim))
        _show(plot_diagram(dg[0], "H0"))
        _show(plot_diagram(dg[1], "H1"))
        if int(maxdim) >= 2:
            _show(plot_diagram(dg[2], "H2"))

        if "models" in st.session_state and "fit_pack" in st.session_state:
            fp = st.session_state.fit_pack
            pi1 = diagram_to_pis(fp["pim_h1"], dg[1])
            pi2 = diagram_to_pis(fp["pim_h2"], dg[2]) if fp["maxdim"] >= 2 else np.zeros_like(pi1)
            hard = tda_feature_block(dg, grid_len=fp["grid_len"], topk=fp["topk"], k_levels=fp["k_levels"])
            d1 = distances_to_prototypes(dg[1], fp["protos_h1"], metric=fp["metric"], seed=123)
            d2 = distances_to_prototypes(dg[2], fp["protos_h2"], metric=fp["metric"], seed=456) if fp["maxdim"] >= 2 else np.zeros_like(d1)
            dens = density_filtration_summaries(pts, fracs=(0.5, 0.8, 1.0), k=fp["dens_k"], maxdim=fp["maxdim"]) if fp["use_density"] else np.zeros((0,), dtype=np.float32)

            feat = np.concatenate([pi1, pi2, hard, d1, d2, dens], axis=0).astype(np.float32)

            models = st.session_state.models
            class_names = st.session_state.class_names
            probs = np.stack([m.predict_proba(feat.reshape(1, -1))[0] for m in models], axis=0)
            p_mean = probs.mean(axis=0)
            p_std = probs.std(axis=0)

            pred_idx = int(np.argmax(p_mean))
            conf = float(np.max(p_mean))
            st.write({"pred": class_names[pred_idx], "conf": conf})
            for k, name in enumerate(class_names):
                st.write(f"{name}: {p_mean[k]:.3f} ± {p_std[k]:.3f}")

