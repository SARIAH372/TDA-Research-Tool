

    
            
# app.py
import time
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample

from topo import (
    SHAPES_2D, SHAPES_3D, make_dataset_mixed,
    dgms_only, dgms_geodesic, finite_bars,
    fit_imagers_multiscale, diagram_to_pis,
    tda_feature_block,
    prototype_diagrams, distances_to_prototypes,
    density_filtration_summaries,
    circular_coordinates,
    lens_pca, mapper_graph, mapper_spectral_features,
)

st.set_page_config(page_title="TDA Research Tool", layout="wide")
st.title("TDA Research Tool")

plt.close("all")

if "models" not in st.session_state:
    st.session_state.models = None
if "class_names" not in st.session_state:
    st.session_state.class_names = None
if "fit_pack" not in st.session_state:
    st.session_state.fit_pack = None
if "last_train" not in st.session_state:
    st.session_state.last_train = None
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False


def _maybe_stop():
    if st.session_state.stop_requested:
        st.session_state.stop_requested = False
        st.stop()


def _show(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def plot_points(P, title, c=None):
    P = np.asarray(P)
    if P.shape[1] == 2:
        fig, ax = plt.subplots()
        if c is None:
            ax.scatter(P[:, 0], P[:, 1], s=12)
        else:
            ax.scatter(P[:, 0], P[:, 1], s=12, c=c)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        return fig
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if c is None:
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=10)
    else:
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=10, c=c)
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
    if kind == "HGB":
        return Pipeline([
            ("clf", HistGradientBoostingClassifier(
                random_state=int(seed)
            ))
        ])
    if kind == "LogReg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                random_state=int(seed)
            ))
        ])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=700,
            random_state=int(seed)
        ))
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


# -----------------------
# Sidebar params
# -----------------------
with st.sidebar:
    st.header("Data")
    n_samples = st.slider("n_samples", 50, 1200, 250, step=50)
    n_points = st.slider("n_points", 40, 320, 140, step=10)
    p_3d = st.slider("p_3d", 0.10, 0.80, 0.45, step=0.05)
    noise_2d = st.slider("noise_2d", 0.0, 0.20, 0.03, step=0.01)
    noise_3d = st.slider("noise_3d", 0.0, 0.20, 0.02, step=0.01)
    seed = st.number_input("seed", value=7, step=1)

    st.header("Topology")
    maxdim = st.selectbox("maxdim", [1, 2], index=1)
    grid_len = st.select_slider("grid_len", options=[32, 64, 96], value=64)
    topk = st.select_slider("topk", options=[6, 8, 10, 12], value=8)
    k_levels = st.select_slider("k_levels", options=[2, 3, 4], value=3)

    st.header("Persistence Images")
    pi_channels = st.multiselect("pi_channels", options=[0.05, 0.08, 0.10], default=[0.05, 0.08])
    if not pi_channels:
        pi_channels = [0.05]

    st.header("Geodesic")
    use_geodesic = st.checkbox("use_geodesic", value=False)
    geo_k = st.slider("geo_k", 4, 20, 10, step=1)

    st.header("Density filtration")
    use_density = st.checkbox("use_density", value=False)
    dens_k = st.slider("dens_k", 6, 16, 10, step=1)

    st.header("Prototype distances")
    use_proto = st.checkbox("use_prototype_distances", value=True)
    proto_cap = st.slider("proto_cap", 8, 30, 15, step=1)

    st.header("Cohomology")
    use_circ = st.checkbox("use_circular_coordinates", value=False)
    circ_coeff = st.select_slider("circ_coeff", options=[31, 47, 59, 83, 101], value=47)

    st.header("Mapper")
    use_mapper = st.checkbox("use_mapper", value=False)
    mapper_intervals = st.slider("mapper_intervals", 4, 24, 10, step=1)
    mapper_overlap = st.slider("mapper_overlap", 0.0, 0.8, 0.30, step=0.05)
    mapper_db_eps = st.slider("mapper_db_eps", 0.05, 2.0, 0.25, step=0.01)
    mapper_min_s = st.slider("mapper_min_samples", 2, 25, 5, step=1)

    st.header("Model")
    model_kind = st.selectbox("model_kind", ["HGB", "LogReg", "MLP"], index=0)

    ens = st.select_slider("ensemble", options=[3, 5, 7], value=5)


# -----------------------
# Tabs
# -----------------------
tabs = st.tabs(["PH", "Geodesic", "Density", "Cohomology", "Mapper", "Train", "Predict"])


# -----------------------
# Common sample generator for tabs
# -----------------------
def sample_points(pick, seed_offset):
    if pick in SHAPES_2D:
        return SHAPES_2D[pick](n=int(n_points), noise=float(noise_2d), seed=int(seed) + seed_offset)
    return SHAPES_3D[pick](n=int(n_points), noise=float(noise_3d), seed=int(seed) + seed_offset)


# ============================================================
# PH tab
# ============================================================
with tabs[0]:
    pick = st.selectbox("shape_ph", list(SHAPES_2D.keys()) + list(SHAPES_3D.keys()))
    pts = sample_points(pick, 101)
    _show(plot_points(pts, f"{pick} dim={pts.shape[1]}"))

    dg = dgms_only(pts, maxdim=int(maxdim))
    c1, c2, c3 = st.columns(3)
    with c1: _show(plot_diagram(dg[0], "H0"))
    with c2: _show(plot_diagram(dg[1], "H1"))
    with c3:
        if int(maxdim) >= 2: _show(plot_diagram(dg[2], "H2"))
        else: st.write("H2 disabled")


# ============================================================
# Geodesic tab
# ============================================================
with tabs[1]:
    pick = st.selectbox("shape_geo", list(SHAPES_2D.keys()) + list(SHAPES_3D.keys()))
    pts = sample_points(pick, 202)
    _show(plot_points(pts, f"{pick} dim={pts.shape[1]}"))

    dg_e = dgms_only(pts, maxdim=int(maxdim))
    dg_g, _ = dgms_geodesic(pts, k=int(geo_k), maxdim=int(maxdim))
    c1, c2 = st.columns(2)
    with c1: _show(plot_diagram(dg_e[1], "H1 euclidean"))
    with c2: _show(plot_diagram(dg_g[1], "H1 geodesic"))


# ============================================================
# Density tab
# ============================================================
with tabs[2]:
    pick = st.selectbox("shape_den", list(SHAPES_2D.keys()) + list(SHAPES_3D.keys()))
    pts = sample_points(pick, 303)
    _show(plot_points(pts, f"{pick} dim={pts.shape[1]}"))

    dens = density_filtration_summaries(pts, fracs=(0.5, 0.8, 1.0), k=int(dens_k), maxdim=int(maxdim))
    st.write({"density_feature_dim": int(dens.shape[0])})


# ============================================================
# Cohomology tab (2D only)
# ============================================================
with tabs[3]:
    pick = st.selectbox("shape_cc", list(SHAPES_2D.keys()))
    pts = SHAPES_2D[pick](n=int(n_points), noise=float(noise_2d), seed=int(seed) + 404)

    theta, birth, death = (None, None, None)
    if use_circ:
        theta, birth, death = circular_coordinates(pts, coeff=int(circ_coeff))
    _show(plot_points(pts, f"{pick} colored by theta if enabled", c=theta))
    st.write({"birth": birth, "death": death})


# ============================================================
# Mapper tab
# ============================================================
with tabs[4]:
    pick = st.selectbox("shape_map", list(SHAPES_2D.keys()) + list(SHAPES_3D.keys()))
    pts = sample_points(pick, 505)

    lens = lens_pca(pts, n_components=1)[:, 0]
    G = mapper_graph(
        pts, lens=lens,
        n_intervals=int(mapper_intervals),
        overlap=float(mapper_overlap),
        dbscan_eps=float(mapper_db_eps),
        min_samples=int(mapper_min_s),
    )
    spec = mapper_spectral_features(G, k_eigs=12)
    st.write({"nodes": int(len(G["nodes"])), "edges": int(len(G["edges"])), "spectral_dim": int(spec.shape[0])})


# ============================================================
# Train tab
# ============================================================
with tabs[5]:
    col1, col2 = st.columns([1, 1])
    with col1:
        start = st.button("train")
    with col2:
        stop = st.button("stop")

    if stop:
        st.session_state.stop_requested = True

    if start:
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
            dgms.append(dgms_only(p, maxdim=int(maxdim)))
            prog.progress((i + 1) / max(1, len(pcs)))
            txt.write(f"dgms: {i+1}/{len(pcs)}")

        pim_h1, pim_h2 = fit_imagers_multiscale(dgms, pixel_sizes=tuple(pi_channels))
        n_classes = int(np.max(y)) + 1

        if use_proto:
            protos_h1 = prototype_diagrams(dgms, y, n_classes=n_classes, dim=1, cap_per_class=int(proto_cap), seed=int(seed))
            protos_h2 = prototype_diagrams(dgms, y, n_classes=n_classes, dim=2, cap_per_class=int(proto_cap), seed=int(seed) + 1) if int(maxdim) >= 2 else [np.zeros((0, 2), dtype=np.float32) for _ in range(n_classes)]
        else:
            protos_h1 = [np.zeros((0, 2), dtype=np.float32) for _ in range(n_classes)]
            protos_h2 = [np.zeros((0, 2), dtype=np.float32) for _ in range(n_classes)]

        mapper_dim = 18
        dens_len = int(density_filtration_summaries(pcs[0], fracs=(0.5, 0.8, 1.0), k=int(dens_k), maxdim=int(maxdim)).shape[0])
        pi_len = len(tuple(pi_channels)) * 32 * 32
        tda_len = int(tda_feature_block(dgms[0], grid_len=int(grid_len), topk=int(topk), k_levels=int(k_levels)).shape[0])
        total_len = (pi_len * 2) + tda_len + (n_classes * 2) + dens_len + mapper_dim

        Xfeat = []
        for p, dg in zip(pcs, dgms):
            _maybe_stop()

            pi1 = diagram_to_pis(pim_h1, dg[1])
            pi2 = diagram_to_pis(pim_h2, dg[2]) if int(maxdim) >= 2 else np.zeros_like(pi1)

            hard = tda_feature_block(dg, grid_len=int(grid_len), topk=int(topk), k_levels=int(k_levels))

            if use_proto:
                d1 = distances_to_prototypes(dg[1], protos_h1, seed=int(seed) + 100)
                d2 = distances_to_prototypes(dg[2], protos_h2, seed=int(seed) + 200) if int(maxdim) >= 2 else np.zeros_like(d1)
            else:
                d1 = np.zeros((n_classes,), dtype=np.float32)
                d2 = np.zeros((n_classes,), dtype=np.float32)

            dens = density_filtration_summaries(p, fracs=(0.5, 0.8, 1.0), k=int(dens_k), maxdim=int(maxdim)) if use_density else np.zeros((dens_len,), dtype=np.float32)
            if dens.shape[0] != dens_len:
                tmp = np.zeros((dens_len,), dtype=np.float32)
                tmp[:min(dens_len, dens.shape[0])] = dens[:min(dens_len, dens.shape[0])]
                dens = tmp

            mapper_feat = np.zeros((mapper_dim,), dtype=np.float32)
            if use_mapper:
                lens = lens_pca(p, n_components=1)[:, 0]
                G = mapper_graph(p, lens=lens, n_intervals=int(mapper_intervals), overlap=float(mapper_overlap),
                                 dbscan_eps=float(mapper_db_eps), min_samples=int(mapper_min_s))
                mapper_feat = mapper_spectral_features(G, k_eigs=12)

            feat = np.concatenate([pi1, pi2, hard, d1, d2, dens, mapper_feat], axis=0).astype(np.float32)
            if feat.shape[0] != total_len:
                tmp = np.zeros((total_len,), dtype=np.float32)
                tmp[:min(total_len, feat.shape[0])] = feat[:min(total_len, feat.shape[0])]
                feat = tmp
            Xfeat.append(feat)

        X = np.vstack(Xfeat).astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=int(seed), stratify=y)
        Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
        Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)

                models = fit_ensemble(
            model_kind,
            Xtr,
            ytr,
            n_models=int(ens),
            seed=int(seed),
        )

        p_mean, _ = ensemble_predict(models, Xte)
        pred = np.argmax(p_mean, axis=1)
        acc = accuracy_score(yte, pred)

        # ===== per-class accuracy =====
        per_class_acc = []
        for c in range(len(class_names)):
            mask = (yte == c)
            if mask.sum() == 0:
                per_class_acc.append(np.nan)
            else:
                per_class_acc.append(float((pred[mask] == c).mean()))

        macro_acc = float(np.nanmean(per_class_acc))

        st.write({
            "acc": float(acc),
            "macro_acc": macro_acc,
        })

        # ===== per-class accuracy table =====
        rows = []
        for i, name in enumerate(class_names):
            rows.append({
                "class": name,
                "accuracy": per_class_acc[i],
            })

        st.dataframe(rows, use_container_width=True)

        # ===== confusion matrix =====
        cm = confusion_matrix(
            yte,
            pred,
            labels=np.arange(len(class_names)),
        )

        fig, ax = plt.subplots()
        im = ax.imshow(cm)

        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    int(cm[i, j]),
                    ha="center",
                    va="center",
                )

        fig.colorbar(im, ax=ax)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        # ===== save trained objects =====
        st.session_state.models = models
        st.session_state.class_names = class_names
        st.session_state.fit_pack = {
            "maxdim": int(maxdim),
            "pim_h1": pim_h1,
            "pim_h2": pim_h2,
            "grid_len": int(grid_len),
            "topk": int(topk),
            "k_levels": int(k_levels),
            "use_proto": bool(use_proto),
            "protos_h1": protos_h1,
            "protos_h2": protos_h2,
            "n_classes": int(n_classes),
            "use_density": bool(use_density),
            "dens_k": int(dens_k),
            "dens_len": int(dens_len),
            "use_mapper": bool(use_mapper),
            "mapper_params": (
                int(mapper_intervals),
                float(mapper_overlap),
                float(mapper_db_eps),
                int(mapper_min_s),
            ),
            "total_len": int(total_len),
        }

        # ===== save training summary =====
        st.session_state.last_train = {
            "acc": float(acc),
            "n": int(len(y)),
            "d": int(X.shape[1]),
            "seconds": float(time.time() - t0),
        }

        # ===== download summary =====
        summary_text = "\n".join([
            f"acc: {st.session_state.last_train['acc']}",
            f"n: {st.session_state.last_train['n']}",
            f"d: {st.session_state.last_train['d']}",
            f"seconds: {st.session_state.last_train['seconds']}",
            f"classes: {len(class_names)}",
        ])

        st.download_button(
            "Download training summary",
            data=summary_text,
            file_name="training_summary.txt",
            mime="text/plain",
        )

        st.success("Training complete. Model is ready in Predict tab.")





