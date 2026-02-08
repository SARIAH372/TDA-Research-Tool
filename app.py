

    
            
            
                
            

            
          
               
        
   
       
        

    
        
                
  # app.py  (full-feature toggles + fixed-length feature vector + no restart loops)
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
    from mpl_toolkits.mplot3d import Axes3D  # noqa
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

def _maybe_stop():
    if st.session_state.stop_requested:
        st.session_state.training = False
        st.session_state.trained = False
        st.session_state.stop_requested = False
        st.stop()

def make_model(kind, seed):
    if kind == "LogReg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=900, multi_class="auto", random_state=int(seed)))
        ])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=700, random_state=int(seed)))
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

with st.sidebar:
    n_samples = st.slider("n_samples", 50, 1200, 250, step=50)
    n_points  = st.slider("n_points", 40, 320, 140, step=10)
    p_3d      = st.slider("p_3d", 0.10, 0.80, 0.45, step=0.05)
    noise_2d  = st.slider("noise_2d", 0.0, 0.20, 0.03, step=0.01)
    noise_3d  = st.slider("noise_3d", 0.0, 0.20, 0.02, step=0.01)
    seed      = st.number_input("seed", value=7, step=1)

    maxdim = st.selectbox("maxdim", [1, 2], index=1)

    grid_len = st.select_slider("grid_len", options=[32, 64, 96], value=64)
    topk = st.select_slider("topk", options=[6, 8, 10, 12], value=8)
    k_levels = st.select_slider("k_levels", options=[2, 3, 4], value=3)

    pi_channels = st.multiselect("pi_channels", options=[0.03, 0.05, 0.08, 0.10], default=[0.05, 0.08])
    if not pi_channels:
        pi_channels = [0.05]

    use_geodesic = st.checkbox("use_geodesic", value=False)
    geo_k = st.slider("geo_k", 4, 20, 10, step=1)

    use_density = st.checkbox("use_density", value=False)
    dens_k = st.slider("dens_k", 6, 16, 10, step=1)

    use_proto = st.checkbox("use_prototype_distances", value=True)
    proto_cap = st.slider("proto_cap", 8, 30, 15, step=1)

    use_mapper = st.checkbox("use_mapper", value=False)
    mapper_intervals = st.slider("mapper_intervals", 4, 24, 10, step=1)
    mapper_overlap = st.slider("mapper_overlap", 0.0, 0.8, 0.30, step=0.05)
    mapper_db_eps = st.slider("mapper_db_eps", 0.05, 2.0, 0.25, step=0.01)
    mapper_min_s = st.slider("mapper_min_samples", 2, 25, 5, step=1)

    use_circ = st.checkbox("use_circular_coordinates", value=False)
    circ_coeff = st.select_slider("circ_coeff", options=[31, 47, 59, 83, 101], value=47)

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

    theta = None
    if use_circ:
        theta, birth, death = circular_coordinates(pts, coeff=int(circ_coeff))
        st.write({"birth": birth, "death": death})

    _show(plot_points(pts, f"{pick} dim={pts.shape[1]}", c=theta))
    dg = dgms_only(pts, maxdim=int(maxdim))
    _show(plot_diagram(dg[0], "H0"))
    _show(plot_diagram(dg[1], "H1"))
    if int(maxdim) >= 2:
        _show(plot_diagram(dg[2], "H2"))

    if use_geodesic:
        dg_g, _ = dgms_geodesic(pts, k=int(geo_k), maxdim=int(maxdim))
        _show(plot_diagram(dg_g[1], "H1 geodesic"))

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

        dgms_euc = []
        dgms_geo = []
        theta_list = []

        for i, p in enumerate(pcs):
            _maybe_stop()
            dgms_euc.append(dgms_only(p, maxdim=int(maxdim)))
            if use_geodesic:
                dg_g, _ = dgms_geodesic(p, k=int(geo_k), maxdim=int(maxdim))
                dgms_geo.append(dg_g)
            else:
                dgms_geo.append(None)

            if use_circ:
                th, _, _ = circular_coordinates(p, coeff=int(circ_coeff))
                theta_list.append(th)
            else:
                theta_list.append(None)

            prog.progress((i + 1) / max(1, len(pcs)))
            txt.write(f"diagrams: {i+1}/{len(pcs)}")

        pim_h1, pim_h2 = fit_imagers_multiscale(dgms_euc, pixel_sizes=tuple(pi_channels))

        n_classes = int(np.max(y)) + 1
        if use_proto:
            protos_h1 = prototype_diagrams(dgms_euc, y, n_classes=n_classes, dim=1, cap_per_class=int(proto_cap), seed=int(seed), metric="sliced")
            protos_h2 = prototype_diagrams(dgms_euc, y, n_classes=n_classes, dim=2, cap_per_class=int(proto_cap), seed=int(seed)+1, metric="sliced") if int(maxdim) >= 2 else [np.zeros((0, 2), dtype=np.float32) for _ in range(n_classes)]
        else:
            protos_h1 = [np.zeros((0, 2), dtype=np.float32) for _ in range(n_classes)]
            protos_h2 = [np.zeros((0, 2), dtype=np.float32) for _ in range(n_classes)]

        mapper_dim = 12 + 6
        dens_len = (3 * (3 + 6)) * 3  # fracs=3, dims=3, (entropy,tp1,tp2,topk6)=9 => 3*3*9=81
        # fixed slots
        pi_len = len(tuple(pi_channels)) * 32 * 32
        pi_block_len = pi_len + pi_len  # H1 + H2
        tda_len = tda_feature_block(dgms_euc[0], grid_len=int(grid_len), topk=int(topk), k_levels=int(k_levels)).shape[0]
        proto_len = n_classes * 2  # H1 + H2
        total_len = pi_block_len + tda_len + proto_len + dens_len + mapper_dim

        Xfeat = []
        for i, (p, dg, dg_g, th) in enumerate(zip(pcs, dgms_euc, dgms_geo, theta_list)):
            _maybe_stop()

            pi1 = diagram_to_pis(pim_h1, dg[1])
            pi2 = diagram_to_pis(pim_h2, dg[2]) if int(maxdim) >= 2 else np.zeros_like(pi1)

            hard = tda_feature_block(dg, grid_len=int(grid_len), topk=int(topk), k_levels=int(k_levels))

            if use_proto:
                d1 = distances_to_prototypes(dg[1], protos_h1, metric="sliced", seed=int(seed) + 100)
                d2 = distances_to_prototypes(dg[2], protos_h2, metric="sliced", seed=int(seed) + 200) if int(maxdim) >= 2 else np.zeros_like(d1)
            else:
                d1 = np.zeros((n_classes,), dtype=np.float32)
                d2 = np.zeros((n_classes,), dtype=np.float32)

            dens = density_filtration_summaries(p, fracs=(0.5, 0.8, 1.0), k=int(dens_k), maxdim=int(maxdim)) if use_density else np.zeros((dens_len,), dtype=np.float32)
            if dens.shape[0] != dens_len:
                dens2 = np.zeros((dens_len,), dtype=np.float32)
                dens2[: min(dens_len, dens.shape[0])] = dens[: min(dens_len, dens.shape[0])]
                dens = dens2

            mapper_feat = np.zeros((mapper_dim,), dtype=np.float32)
            if use_mapper:
                lens = th if (th is not None and th.shape[0] == p.shape[0]) else lens_pca(p, n_components=1)[:, 0]
                G = mapper_graph(p, lens=lens, n_intervals=int(mapper_intervals), overlap=float(mapper_overlap), dbscan_eps=float(mapper_db_eps), min_samples=int(mapper_min_s))
                mapper_feat = mapper_spectral_features(G, k_eigs=12)

            feat = np.concatenate([pi1, pi2, hard, d1, d2, dens, mapper_feat], axis=0).astype(np.float32)
            if feat.shape[0] != total_len:
                f2 = np.zeros((total_len,), dtype=np.float32)
                f2[: min(total_len, feat.shape[0])] = feat[: min(total_len, feat.shape[0])]
                feat = f2
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
            "pi_channels": tuple(pi_channels),
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
            "mapper_params": (int(mapper_intervals), float(mapper_overlap), float(mapper_db_eps), int(mapper_min_s)),
            "use_circ": bool(use_circ),
            "circ_coeff": int(circ_coeff),
            "total_len": int(total_len),
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
        dg = dgms_only(pts, maxdim=int(maxdim))
        _show(plot_diagram(dg[0], "H0"))
        _show(plot_diagram(dg[1], "H1"))
        if int(maxdim) >= 2:
            _show(plot_diagram(dg[2], "H2"))

        if "models" in st.session_state and "fit_pack" in st.session_state:
            fp = st.session_state.fit_pack
            theta = None
            if fp["use_circ"]:
                theta, _, _ = circular_coordinates(pts, coeff=int(fp["circ_coeff"]))

            pi1 = diagram_to_pis(fp["pim_h1"], dg[1])
            pi2 = diagram_to_pis(fp["pim_h2"], dg[2]) if fp["maxdim"] >= 2 else np.zeros_like(pi1)
            hard = tda_feature_block(dg, grid_len=fp["grid_len"], topk=fp["topk"], k_levels=fp["k_levels"])

            if fp["use_proto"]:
                d1 = distances_to_prototypes(dg[1], fp["protos_h1"], metric="sliced", seed=123)
                d2 = distances_to_prototypes(dg[2], fp["protos_h2"], metric="sliced", seed=456) if fp["maxdim"] >= 2 else np.zeros_like(d1)
            else:
                d1 = np.zeros((fp["n_classes"],), dtype=np.float32)
                d2 = np.zeros((fp["n_classes"],), dtype=np.float32)

            dens = density_filtration_summaries(pts, fracs=(0.5, 0.8, 1.0), k=fp["dens_k"], maxdim=fp["maxdim"]) if fp["use_density"] else np.zeros((fp["dens_len"],), dtype=np.float32)
            if dens.shape[0] != fp["dens_len"]:
                dens2 = np.zeros((fp["dens_len"],), dtype=np.float32)
                dens2[: min(fp["dens_len"], dens.shape[0])] = dens[: min(fp["dens_len"], dens.shape[0])]
                dens = dens2

            mapper_dim = 12 + 6
            mapper_feat = np.zeros((mapper_dim,), dtype=np.float32)
            if fp["use_mapper"]:
                (mi, mo, meps, mmin) = fp["mapper_params"]
                lens = theta if (theta is not None and theta.shape[0] == pts.shape[0]) else lens_pca(pts, n_components=1)[:, 0]
                G = mapper_graph(pts, lens=lens, n_intervals=mi, overlap=mo, dbscan_eps=meps, min_samples=mmin)
                mapper_feat = mapper_spectral_features(G, k_eigs=12)

            feat = np.concatenate([pi1, pi2, hard, d1, d2, dens, mapper_feat], axis=0).astype(np.float32)
            if feat.shape[0] != fp["total_len"]:
                f2 = np.zeros((fp["total_len"],), dtype=np.float32)
                f2[: min(fp["total_len"], feat.shape[0])] = feat[: min(fp["total_len"], feat.shape[0])]
                feat = f2

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

            fig, ax = plt.subplots()
            ax.bar(np.arange(len(class_names)), p_mean)
            ax.set_title("probabilities")
            _show(fig)
