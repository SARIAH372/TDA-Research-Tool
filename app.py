

    
            

    
    
       
       # app.py
import time
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier

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
st.write("BUILD_TAG: 7tabs_v1")

plt.close("all")

# -----------------------
# session state (persistent display)
# -----------------------
defaults = {
    "stop_requested": False,
    "trained": False,
    "class_names": None,
    "model": None,
    "fit_pack": None,
    "train_summary": None,
    "train_rows": None,
    "train_cm": None,
    "summary_text": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def maybe_stop():
    if st.session_state.stop_requested:
        st.session_state.stop_requested = False
        st.stop()


def sanitize_array(x):
    x = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def show_fig(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def plot_points(points, title, c=None):
    P = np.asarray(points)
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


def plot_confusion_matrix(cm, class_names):
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
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    return fig


def sample_points(pick, offset, n_points, noise_2d, noise_3d, seed):
    if pick in SHAPES_2D:
        return SHAPES_2D[pick](n=int(n_points), noise=float(noise_2d), seed=int(seed) + int(offset))
    return SHAPES_3D[pick](n=int(n_points), noise=float(noise_3d), seed=int(seed) + int(offset))


# -----------------------
# Sidebar
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

    st.header("PI channels")
    pi_channels = st.multiselect("pi_channels", options=[0.05, 0.08, 0.10], default=[0.05, 0.08])
    if not pi_channels:
        pi_channels = [0.05]

    st.header("Geodesic")
    geo_k = st.slider("geo_k", 4, 20, 10, step=1)

    st.header("Density")
    use_density = st.checkbox("use_density", value=True)
    dens_k = st.slider("dens_k", 6, 16, 10, step=1)

    st.header("Prototype distances")
    use_proto = st.checkbox("use_proto_distances", value=True)
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

    st.header("Train")
    test_size = st.slider("test_size", 0.10, 0.40, 0.25, step=0.05)


tabs = st.tabs(["PH", "Geodesic", "Density", "Cohomology", "Mapper", "Train", "Predict"])


# ============================================================
# Tab 0: PH
# ============================================================
with tabs[0]:
    pick = st.selectbox("shape_ph", list(SHAPES_2D.keys()) + list(SHAPES_3D.keys()))
    pts = sample_points(pick, 101, n_points, noise_2d, noise_3d, seed)
    show_fig(plot_points(pts, f"{pick} dim={pts.shape[1]}"))
    dg = dgms_only(pts, maxdim=int(maxdim))
    c1, c2, c3 = st.columns(3)
    with c1: show_fig(plot_diagram(dg[0], "H0"))
    with c2: show_fig(plot_diagram(dg[1], "H1"))
    with c3:
        if int(maxdim) >= 2: show_fig(plot_diagram(dg[2], "H2"))
        else: st.write("H2 disabled (maxdim=1)")


# ============================================================
# Tab 1: Geodesic
# ============================================================
with tabs[1]:
    pick = st.selectbox("shape_geo", list(SHAPES_2D.keys()) + list(SHAPES_3D.keys()))
    pts = sample_points(pick, 202, n_points, noise_2d, noise_3d, seed)
    show_fig(plot_points(pts, f"{pick} dim={pts.shape[1]}"))
    dg_e = dgms_only(pts, maxdim=int(maxdim))
    dg_g, _ = dgms_geodesic(pts, k=int(geo_k), maxdim=int(maxdim))
    c1, c2 = st.columns(2)
    with c1: show_fig(plot_diagram(dg_e[1], "H1 euclidean"))
    with c2: show_fig(plot_diagram(dg_g[1], "H1 geodesic"))


# ============================================================
# Tab 2: Density
# ============================================================
with tabs[2]:
    pick = st.selectbox("shape_den", list(SHAPES_2D.keys()) + list(SHAPES_3D.keys()))
    pts = sample_points(pick, 303, n_points, noise_2d, noise_3d, seed)
    show_fig(plot_points(pts, f"{pick} dim={pts.shape[1]}"))
    dens = density_filtration_summaries(pts, fracs=(0.5, 0.8, 1.0), k=int(dens_k), maxdim=int(maxdim))
    st.write({"density_feature_dim": int(dens.shape[0])})


# ============================================================
# Tab 3: Cohomology
# ============================================================
with tabs[3]:
    pick = st.selectbox("shape_cc", list(SHAPES_2D.keys()))
    pts = SHAPES_2D[pick](n=int(n_points), noise=float(noise_2d), seed=int(seed) + 404)
    theta, birth, death = (None, None, None)
    if use_circ:
        theta, birth, death = circular_coordinates(pts, coeff=int(circ_coeff))
    show_fig(plot_points(pts, f"{pick} colored by theta (if enabled)", c=theta))
    st.write({"birth": birth, "death": death})


# ============================================================
# Tab 4: Mapper
# ============================================================
with tabs[4]:
    pick = st.selectbox("shape_map", list(SHAPES_2D.keys()) + list(SHAPES_3D.keys()))
    pts = sample_points(pick, 505, n_points, noise_2d, noise_3d, seed)
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
# Tab 5: Train (always shows last results)
# ============================================================
with tabs[5]:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1: start = st.button("train")
    with col2: stop = st.button("stop")
    with col3: reset = st.button("reset_results")

    if reset:
        for k in ["trained","class_names","model","fit_pack","train_summary","train_rows","train_cm","summary_text"]:
            st.session_state[k] = None if k not in ["trained"] else False
        st.session_state.stop_requested = False
        st.success("Results cleared.")

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
        status = st.empty()

        dgms = []
        for i, p in enumerate(pcs):
            maybe_stop()
            dgms.append(dgms_only(p, maxdim=int(maxdim)))
            prog.progress((i + 1) / max(1, len(pcs)))
            status.write(f"diagrams: {i+1}/{len(pcs)}")
            time.sleep(0.003)

        pim_h1, pim_h2 = fit_imagers_multiscale(dgms, pixel_sizes=tuple(pi_channels))
        n_classes = int(np.max(y)) + 1

        protos_h1 = prototype_diagrams(dgms, y, n_classes=n_classes, dim=1, cap_per_class=int(proto_cap), seed=int(seed)) if use_proto else None
        protos_h2 = (
            prototype_diagrams(dgms, y, n_classes=n_classes, dim=2, cap_per_class=int(proto_cap), seed=int(seed) + 1)
            if (use_proto and int(maxdim) >= 2) else None
        )

        dens_len = int(density_filtration_summaries(pcs[0], fracs=(0.5, 0.8, 1.0), k=int(dens_k), maxdim=int(maxdim)).shape[0]) if use_density else 0
        pi_len = len(tuple(pi_channels)) * 32 * 32
        tda_len = int(tda_feature_block(dgms[0], grid_len=int(grid_len), topk=int(topk), k_levels=int(k_levels)).shape[0])
        proto_len = (n_classes * 2)
        total_len = (pi_len * 2) + tda_len + proto_len + dens_len

        Xfeat = []
        for p, dg in zip(pcs, dgms):
            maybe_stop()
            pi1 = diagram_to_pis(pim_h1, dg[1])
            pi2 = diagram_to_pis(pim_h2, dg[2]) if int(maxdim) >= 2 else np.zeros_like(pi1)
            hard = tda_feature_block(dg, grid_len=int(grid_len), topk=int(topk), k_levels=int(k_levels))

            if use_proto and protos_h1 is not None:
                d1 = distances_to_prototypes(dg[1], protos_h1, seed=int(seed) + 100)
                d2 = distances_to_prototypes(dg[2], protos_h2, seed=int(seed) + 200) if (int(maxdim) >= 2 and protos_h2 is not None) else np.zeros_like(d1)
            else:
                d1 = np.zeros((n_classes,), dtype=np.float32)
                d2 = np.zeros((n_classes,), dtype=np.float32)

            dens = density_filtration_summaries(p, fracs=(0.5, 0.8, 1.0), k=int(dens_k), maxdim=int(maxdim)) if use_density else np.zeros((0,), dtype=np.float32)
            feat = np.concatenate([pi1, pi2, hard, d1, d2, dens], axis=0).astype(np.float32)

            if feat.shape[0] != total_len:
                tmp = np.zeros((total_len,), dtype=np.float32)
                tmp[:min(total_len, feat.shape[0])] = feat[:min(total_len, feat.shape[0])]
                feat = tmp

            Xfeat.append(feat)

        X = sanitize_array(np.vstack(Xfeat))
        y = np.asarray(y, dtype=np.int64)

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=float(test_size), random_state=int(seed), stratify=y)

        model = HistGradientBoostingClassifier(random_state=int(seed))
        model.fit(Xtr, ytr)

        proba = model.predict_proba(Xte)
        pred = np.argmax(proba, axis=1)
        acc = accuracy_score(yte, pred)

        per_class_acc = []
        for c in range(len(class_names)):
            mask = (yte == c)
            per_class_acc.append(float((pred[mask] == c).mean()) if mask.sum() else np.nan)
        macro_acc = float(np.nanmean(per_class_acc))

        rows = [{"class": class_names[i], "accuracy": per_class_acc[i]} for i in range(len(class_names))]
        cm = confusion_matrix(yte, pred, labels=np.arange(len(class_names)))

        st.session_state.trained = True
        st.session_state.class_names = class_names
        st.session_state.model = model
        st.session_state.fit_pack = {"total_len": int(total_len), "maxdim": int(maxdim)}

        st.session_state.train_summary = {
            "acc": float(acc),
            "macro_acc": float(macro_acc),
            "n": int(len(y)),
            "d": int(X.shape[1]),
            "seconds": float(time.time() - t0),
        }
        st.session_state.train_rows = rows
        st.session_state.train_cm = cm

        st.session_state.summary_text = "\n".join([
            f"acc: {st.session_state.train_summary['acc']}",
            f"macro_acc: {st.session_state.train_summary['macro_acc']}",
            f"n: {st.session_state.train_summary['n']}",
            f"d: {st.session_state.train_summary['d']}",
            f"seconds: {st.session_state.train_summary['seconds']}",
            f"classes: {len(class_names)}",
        ])

        st.success("Training complete. Results are shown below.")

    st.divider()
    st.subheader("Last Training Results")

    if st.session_state.train_summary is None:
        st.info("No training results yet.")
    else:
        st.write(st.session_state.train_summary)

        if st.session_state.train_rows is not None:
            st.dataframe(st.session_state.train_rows, width="stretch")

        if st.session_state.train_cm is not None and st.session_state.class_names is not None:
            fig = plot_confusion_matrix(st.session_state.train_cm, st.session_state.class_names)
            show_fig(fig)

        if st.session_state.summary_text is not None:
            st.download_button(
                "Download training summary",
                data=st.session_state.summary_text,
                file_name="training_summary.txt",
                mime="text/plain",
            )


# ============================================================
# Tab 6: Predict (shows model ready)
# ============================================================
with tabs[6]:
    st.subheader("Predict")
    if not st.session_state.trained or st.session_state.model is None:
        st.info("Train first in the Train tab.")
    else:
        st.success("Model ready.")
        st.write({"classes": len(st.session_state.class_names)})
        st.write("Prediction feature rebuild can be added next (kept stable here).")

