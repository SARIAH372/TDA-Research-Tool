

    
            
            
                
            

            
          # app.py (corrected for Streamlit reruns + matplotlib figure leaks)
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
    persistence_diagrams, persistence_diagrams_geodesic, finite_bars,
    fit_imagers_multiscale, diagram_to_pis,
    tda_feature_block,
    prototype_diagrams, distances_to_prototypes,
    density_filtration_summaries,
    pss_kernel_features,
    circular_coordinates,
    lens_pca, mapper_graph, mapper_spectral_features,
)

st.set_page_config(page_title="TDA Research Tool", layout="wide")
st.title("TDA Research Tool")

# ---- session state guards ----
if "training" not in st.session_state:
    st.session_state.training = False
if "trained" not in st.session_state:
    st.session_state.trained = False

# ---- prevent matplotlib figure accumulation across reruns ----
plt.close("all")


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


def plot_barcode(dgm, title, max_bars=60):
    bars = finite_bars(dgm)
    fig, ax = plt.subplots()
    if len(bars) == 0:
        ax.set_title(title)
        return fig
    pers = bars[:, 1] - bars[:, 0]
    order = np.argsort(pers)[::-1]
    bars = bars[order][:max_bars]
    for i, (bi, di) in enumerate(bars):
        ax.plot([bi, di], [i, i])
    ax.set_xlabel("eps")
    ax.set_ylabel("bar")
    ax.set_title(title)
    return fig


def make_model(kind, seed):
    if kind == "LogReg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1600, multi_class="auto", random_state=int(seed)))
        ])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 160, 96, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=1100,
            random_state=int(seed)
        ))
    ])


def fit_ensemble(kind, Xtr, ytr, n_models=7, seed=7):
    rng = np.random.default_rng(seed)
    models = []
    for _ in range(n_models):
        sk = int(rng.integers(0, 10_000))
        Xb, yb = resample(Xtr, ytr, replace=True, random_state=sk)
        m = make_model(kind, sk)
        m.fit(Xb, yb)
        models.append(m)
    return models


def ensemble_predict(models, X):
    probs = np.stack([m.predict_proba(X) for m in models], axis=0)
    return probs.mean(axis=0), probs.std(axis=0)


def build_features(
    point_clouds, y,
    pixel_sizes, grid_len, topk, k_levels,
    proto_cap, seed, metric,
    use_geodesic, geo_k,
    use_density, dens_k,
    use_pss, pss_nb, pss_np, pss_sigma,
    use_mapper, mapper_intervals, mapper_overlap, mapper_db_eps, mapper_min_s,
    use_circ, circ_coeff,
):
    dgms_euc = []
    dgms_geo = []
    circ_cache = []

    for p in point_clouds:
        dg_e = persistence_diagrams(p, maxdim=2)
        dgms_euc.append(dg_e)

        if use_geodesic:
            dg_g, _ = persistence_diagrams_geodesic(p, k=int(geo_k), maxdim=2)
        else:
            dg_g = None
        dgms_geo.append(dg_g)

        if use_circ:
            theta, _, _, _ = circular_coordinates(p, coeff=int(circ_coeff), eps=None)
            if theta is None:
                circ_cache.append(np.zeros((p.shape[0],), dtype=np.float32))
            else:
                circ_cache.append(theta.astype(np.float32))
        else:
            circ_cache.append(None)

    if not pixel_sizes:
        pixel_sizes = (0.05,)

    pim_h1_list, pim_h2_list = fit_imagers_multiscale(dgms_euc, pixel_sizes=tuple(pixel_sizes))
    n_classes = int(np.max(y)) + 1

    protos_h1 = prototype_diagrams(
        dgms_euc, y, n_classes=n_classes, dim=1,
        cap_per_class=int(proto_cap), seed=int(seed), metric=str(metric)
    )
    protos_h2 = prototype_diagrams(
        dgms_euc, y, n_classes=n_classes, dim=2,
        cap_per_class=int(proto_cap), seed=int(seed), metric=str(metric)
    )

    Xfeat = []
    for p, dg_e, dg_g, theta in zip(point_clouds, dgms_euc, dgms_geo, circ_cache):
        pi1 = diagram_to_pis(pim_h1_list, dg_e[1])
        pi2 = diagram_to_pis(pim_h2_list, dg_e[2])

        hard = tda_feature_block(dg_e, grid_len=int(grid_len), topk=int(topk), k_levels=int(k_levels))

        d1 = distances_to_prototypes(dg_e[1], protos_h1, metric=str(metric), seed=int(seed) + 100)
        d2 = distances_to_prototypes(dg_e[2], protos_h2, metric=str(metric), seed=int(seed) + 200)

        geo = np.zeros((0,), dtype=np.float32)
        if use_geodesic and dg_g is not None:
            geo = tda_feature_block(
                dg_g, grid_len=int(grid_len),
                topk=min(10, int(topk)),
                k_levels=min(3, int(k_levels))
            ).astype(np.float32)

        dens = np.zeros((0,), dtype=np.float32)
        if use_density:
            dens = density_filtration_summaries(
                p, fracs=(0.35, 0.55, 0.75, 1.0),
                k=int(dens_k), maxdim=2
            )

        pss = np.zeros((0,), dtype=np.float32)
        if use_pss:
            pss_h1 = pss_kernel_features(dg_e[1], nb=int(pss_nb), npers=int(pss_np), sigma=float(pss_sigma))
            pss_h2 = pss_kernel_features(dg_e[2], nb=int(pss_nb), npers=int(pss_np), sigma=float(pss_sigma))
            pss = np.concatenate([pss_h1, pss_h2], axis=0).astype(np.float32)

        mapper_feat = np.zeros((0,), dtype=np.float32)
        if use_mapper:
            if theta is not None and use_circ and p.shape[0] == theta.shape[0]:
                lens = theta
            else:
                lens = lens_pca(p, n_components=1)[:, 0]
            G = mapper_graph(
                p, lens=lens,
                n_intervals=int(mapper_intervals),
                overlap=float(mapper_overlap),
                dbscan_eps=float(mapper_db_eps),
                min_samples=int(mapper_min_s),
            )
            mapper_feat = mapper_spectral_features(G, k_eigs=12).astype(np.float32)

        feat = np.concatenate([pi1, pi2, hard, d1, d2, geo, dens, pss, mapper_feat], axis=0).astype(np.float32)
        Xfeat.append(feat)

    fit_pack = {
        "pixel_sizes": tuple(pixel_sizes),
        "pim_h1_list": pim_h1_list,
        "pim_h2_list": pim_h2_list,
        "protos_h1": protos_h1,
        "protos_h2": protos_h2,
        "grid_len": int(grid_len),
        "topk": int(topk),
        "k_levels": int(k_levels),
        "metric": str(metric),
        "use_geodesic": bool(use_geodesic),
        "geo_k": int(geo_k),
        "use_density": bool(use_density),
        "dens_k": int(dens_k),
        "use_pss": bool(use_pss),
        "pss_nb": int(pss_nb),
        "pss_np": int(pss_np),
        "pss_sigma": float(pss_sigma),
        "use_mapper": bool(use_mapper),
        "mapper_intervals": int(mapper_intervals),
        "mapper_overlap": float(mapper_overlap),
        "mapper_db_eps": float(mapper_db_eps),
        "mapper_min_s": int(mapper_min_s),
        "use_circ": bool(use_circ),
        "circ_coeff": int(circ_coeff),
    }
    return np.vstack(Xfeat), dgms_euc, dgms_geo, fit_pack


def featurize_one(points, fit_pack):
    dg_e = persistence_diagrams(points, maxdim=2)

    pi1 = diagram_to_pis(fit_pack["pim_h1_list"], dg_e[1])
    pi2 = diagram_to_pis(fit_pack["pim_h2_list"], dg_e[2])

    hard = tda_feature_block(
        dg_e,
        grid_len=int(fit_pack["grid_len"]),
        topk=int(fit_pack["topk"]),
        k_levels=int(fit_pack["k_levels"])
    )

    d1 = distances_to_prototypes(dg_e[1], fit_pack["protos_h1"], metric=str(fit_pack["metric"]), seed=123)
    d2 = distances_to_prototypes(dg_e[2], fit_pack["protos_h2"], metric=str(fit_pack["metric"]), seed=456)

    geo = np.zeros((0,), dtype=np.float32)
    if fit_pack["use_geodesic"]:
        dg_g, _ = persistence_diagrams_geodesic(points, k=int(fit_pack["geo_k"]), maxdim=2)
        geo = tda_feature_block(
            dg_g,
            grid_len=int(fit_pack["grid_len"]),
            topk=min(10, int(fit_pack["topk"])),
            k_levels=min(3, int(fit_pack["k_levels"]))
        ).astype(np.float32)
    else:
        dg_g = None

    dens = np.zeros((0,), dtype=np.float32)
    if fit_pack["use_density"]:
        dens = density_filtration_summaries(
            points, fracs=(0.35, 0.55, 0.75, 1.0),
            k=int(fit_pack["dens_k"]), maxdim=2
        )

    pss = np.zeros((0,), dtype=np.float32)
    if fit_pack["use_pss"]:
        pss_h1 = pss_kernel_features(dg_e[1], nb=int(fit_pack["pss_nb"]), npers=int(fit_pack["pss_np"]), sigma=float(fit_pack["pss_sigma"]))
        pss_h2 = pss_kernel_features(dg_e[2], nb=int(fit_pack["pss_nb"]), npers=int(fit_pack["pss_np"]), sigma=float(fit_pack["pss_sigma"]))
        pss = np.concatenate([pss_h1, pss_h2], axis=0).astype(np.float32)

    mapper_feat = np.zeros((0,), dtype=np.float32)
    theta = None
    if fit_pack["use_circ"]:
        theta, _, _, _ = circular_coordinates(points, coeff=int(fit_pack["circ_coeff"]), eps=None)

    if fit_pack["use_mapper"]:
        if theta is not None and points.shape[0] == theta.shape[0]:
            lens = theta
        else:
            lens = lens_pca(points, n_components=1)[:, 0]
        G = mapper_graph(
            points, lens=lens,
            n_intervals=int(fit_pack["mapper_intervals"]),
            overlap=float(fit_pack["mapper_overlap"]),
            dbscan_eps=float(fit_pack["mapper_db_eps"]),
            min_samples=int(fit_pack["mapper_min_s"]),
        )
        mapper_feat = mapper_spectral_features(G, k_eigs=12).astype(np.float32)

    feat = np.concatenate([pi1, pi2, hard, d1, d2, geo, dens, pss, mapper_feat], axis=0).astype(np.float32)
    return feat, dg_e, dg_g


with st.sidebar:
    n_samples = st.slider("n_samples", 200, 1600, 650, step=50)
    n_points = st.slider("n_points", 80, 320, 160, step=10)
    p_3d = st.slider("p_3d", 0.10, 0.80, 0.45, step=0.05)
    noise_2d = st.slider("noise_2d", 0.0, 0.20, 0.03, step=0.01)
    noise_3d = st.slider("noise_3d", 0.0, 0.20, 0.02, step=0.01)
    seed = st.number_input("seed", value=7, step=1)

    pixel_sizes = st.multiselect(
        "pixel_sizes",
        options=[0.02, 0.03, 0.05, 0.08, 0.10],
        default=[0.03, 0.05, 0.08]
    )
    if not pixel_sizes:
        pixel_sizes = [0.05]

    grid_len = st.slider("grid_len", 32, 128, 64, step=16)
    topk = st.slider("topk", 4, 24, 10, step=2)
    k_levels = st.slider("k_levels", 1, 6, 3, step=1)

    metric = st.selectbox("diagram_metric", ["wasserstein", "bottleneck", "sliced"])
    proto_cap = st.slider("proto_cap", 10, 70, 25, step=5)

    use_geodesic = st.checkbox("use_geodesic", value=True)
    geo_k = st.slider("geo_k", 4, 20, 10, step=1)

    use_density = st.checkbox("use_density", value=True)
    dens_k = st.slider("dens_k", 4, 30, 10, step=1)

    use_pss = st.checkbox("use_pss_kernel", value=True)
    pss_nb = st.select_slider("pss_nb", options=[16, 24, 32, 40], value=32)
    pss_np = st.select_slider("pss_np", options=[16, 24, 32, 40], value=32)
    pss_sigma = st.select_slider("pss_sigma", options=[0.08, 0.12, 0.15, 0.20, 0.25], value=0.15)

    use_circ = st.checkbox("use_circular_coords", value=True)
    circ_coeff = st.select_slider("circ_coeff", options=[31, 47, 59, 83, 101], value=47)

    use_mapper = st.checkbox("use_mapper_spectral", value=True)
    mapper_intervals = st.slider("mapper_intervals", 4, 24, 10, step=1)
    mapper_overlap = st.slider("mapper_overlap", 0.0, 0.8, 0.30, step=0.05)
    mapper_db_eps = st.slider("mapper_db_eps", 0.05, 2.0, 0.25, step=0.01)
    mapper_min_s = st.slider("mapper_min_samples", 2, 25, 5, step=1)

    model_kind = st.selectbox("model_kind", ["LogReg", "MLP"])
    ens = st.slider("ens", 3, 11, 7, step=2)

tabs = st.tabs(["PH", "Cohomology", "Mapper", "Train", "Predict"])

with tabs[0]:
    keys2 = list(SHAPES_2D.keys())
    keys3 = list(SHAPES_3D.keys())
    all_names = keys2 + keys3
    pick = st.selectbox("shape", all_names)

    if pick in SHAPES_2D:
        pts = SHAPES_2D[pick](n=int(n_points), noise=float(noise_2d), seed=int(seed) + 101)
    else:
        pts = SHAPES_3D[pick](n=int(n_points), noise=float(noise_3d), seed=int(seed) + 202)

    _show(plot_points(pts, f"{pick} dim={pts.shape[1]}"))

    dg_e = persistence_diagrams(pts, maxdim=2)
    c1, c2, c3 = st.columns(3)
    with c1:
        _show(plot_diagram(dg_e[0], "H0 euclidean"))
        _show(plot_barcode(dg_e[0], "H0 barcode"))
    with c2:
        _show(plot_diagram(dg_e[1], "H1 euclidean"))
        _show(plot_barcode(dg_e[1], "H1 barcode"))
    with c3:
        _show(plot_diagram(dg_e[2], "H2 euclidean"))
        _show(plot_barcode(dg_e[2], "H2 barcode"))

    if use_geodesic:
        dg_g, _ = persistence_diagrams_geodesic(pts, k=int(geo_k), maxdim=2)
        g1, g2, g3 = st.columns(3)
        with g1:
            _show(plot_diagram(dg_g[0], "H0 geodesic"))
        with g2:
            _show(plot_diagram(dg_g[1], "H1 geodesic"))
        with g3:
            _show(plot_diagram(dg_g[2], "H2 geodesic"))

with tabs[1]:
    keys2 = list(SHAPES_2D.keys())
    keys3 = list(SHAPES_3D.keys())
    all_names = keys2 + keys3
    pick = st.selectbox("shape_cc", all_names)

    if pick in SHAPES_2D:
        pts = SHAPES_2D[pick](n=int(n_points), noise=float(noise_2d), seed=int(seed) + 303)
    else:
        pts = SHAPES_3D[pick](n=int(n_points), noise=float(noise_3d), seed=int(seed) + 404)

    theta, birth, death, _ = circular_coordinates(pts, coeff=int(circ_coeff), eps=None) if use_circ else (None, None, None, None)
    _show(plot_points(pts, f"{pick} dim={pts.shape[1]}", c=theta if theta is not None else None))
    if theta is not None:
        st.write({"birth": float(birth), "death": float(death)})

with tabs[2]:
    keys2 = list(SHAPES_2D.keys())
    keys3 = list(SHAPES_3D.keys())
    all_names = keys2 + keys3
    pick = st.selectbox("shape_mapper", all_names)

    if pick in SHAPES_2D:
        pts = SHAPES_2D[pick](n=int(n_points), noise=float(noise_2d), seed=int(seed) + 505)
    else:
        pts = SHAPES_3D[pick](n=int(n_points), noise=float(noise_3d), seed=int(seed) + 606)

    if use_circ:
        theta, _, _, _ = circular_coordinates(pts, coeff=int(circ_coeff), eps=None)
        lens = theta if theta is not None else lens_pca(pts, n_components=1)[:, 0]
    else:
        lens = lens_pca(pts, n_components=1)[:, 0]

    G = mapper_graph(
        pts, lens=lens,
        n_intervals=int(mapper_intervals),
        overlap=float(mapper_overlap),
        dbscan_eps=float(mapper_db_eps),
        min_samples=int(mapper_min_s),
    )
    spec = mapper_spectral_features(G, k_eigs=12)
    st.write({"n_nodes": int(len(G["nodes"])), "n_edges": int(len(G["edges"]))})
    st.write({"spectral": spec.tolist()})

with tabs[3]:
    @st.cache_data
    def cached_data(n_samples, n_points, noise_2d, noise_3d, p_3d, seed):
        return make_dataset_mixed(
            n_samples=int(n_samples),
            n_points=int(n_points),
            noise_2d=float(noise_2d),
            noise_3d=float(noise_3d),
            p_3d=float(p_3d),
            seed=int(seed),
        )

    colA, colB = st.columns([1, 1])
    with colA:
        start = st.button("train", disabled=st.session_state.training)
    with colB:
        reset = st.button("reset", disabled=st.session_state.training)

    if reset:
        for k in ["class_names", "models", "fit_pack"]:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state.training = False
        st.session_state.trained = False
        st.rerun()

    if start and not st.session_state.training:
        st.session_state.training = True
        st.session_state.trained = False
        st.rerun()

    if st.session_state.training and not st.session_state.trained:
        with st.spinner("training..."):
            pcs, y, class_names = cached_data(n_samples, n_points, noise_2d, noise_3d, p_3d, seed)

            X, dgms_euc, dgms_geo, fit_pack = build_features(
                pcs, y,
                pixel_sizes=tuple(pixel_sizes),
                grid_len=int(grid_len),
                topk=int(topk),
                k_levels=int(k_levels),
                proto_cap=int(proto_cap),
                seed=int(seed),
                metric=str(metric),
                use_geodesic=bool(use_geodesic),
                geo_k=int(geo_k),
                use_density=bool(use_density),
                dens_k=int(dens_k),
                use_pss=bool(use_pss),
                pss_nb=int(pss_nb),
                pss_np=int(pss_np),
                pss_sigma=float(pss_sigma),
                use_mapper=bool(use_mapper),
                mapper_intervals=int(mapper_intervals),
                mapper_overlap=float(mapper_overlap),
                mapper_db_eps=float(mapper_db_eps),
                mapper_min_s=int(mapper_min_s),
                use_circ=bool(use_circ),
                circ_coeff=int(circ_coeff),
            )

            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=int(seed), stratify=y)
            models = fit_ensemble(model_kind, Xtr, ytr, n_models=int(ens), seed=int(seed))
            p_mean, p_std = ensemble_predict(models, Xte)
            pred = np.argmax(p_mean, axis=1)
            acc = accuracy_score(yte, pred)

            st.session_state["class_names"] = class_names
            st.session_state["models"] = models
            st.session_state["fit_pack"] = fit_pack

            st.session_state.trained = True
            st.session_state.training = False

        st.rerun()

    if st.session_state.trained and ("models" in st.session_state):
        st.write({
            "trained": True,
            "classes": int(len(st.session_state["class_names"])),
        })

        # quick evaluation on a tiny cached batch (no recomputation)
        # (accuracy already computed during training step; show only confusion matrix from last train run if desired)
        # Keep this section minimal to avoid extra heavy computation.

with tabs[4]:
    mode = st.radio("input", ["generate", "upload"], horizontal=True)
    pts = None

    if mode == "generate":
        keys2 = list(SHAPES_2D.keys())
        keys3 = list(SHAPES_3D.keys())
        all_names = keys2 + keys3
        pick = st.selectbox("shape_pred", all_names)
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
        dg_e = persistence_diagrams(pts, maxdim=2)
        c1, c2, c3 = st.columns(3)
        with c1:
            _show(plot_diagram(dg_e[0], "H0"))
        with c2:
            _show(plot_diagram(dg_e[1], "H1"))
        with c3:
            _show(plot_diagram(dg_e[2], "H2"))

        if "models" in st.session_state and "fit_pack" in st.session_state:
            feat, dg_e2, dg_g2 = featurize_one(pts, st.session_state["fit_pack"])
            models = st.session_state["models"]
            class_names = st.session_state["class_names"]

            probs = np.stack([m.predict_proba(feat.reshape(1, -1))[0] for m in models], axis=0)
            p_mean = probs.mean(axis=0)
            p_std = probs.std(axis=0)

            pred_idx = int(np.argmax(p_mean))
            conf = float(np.max(p_mean))

            st.write({"pred": class_names[pred_idx], "conf": conf})
            for k, name in enumerate(class_names):
                st.write(f"{name}: {p_mean[k]:.3f} ± {p_std[k]:.3f}")
