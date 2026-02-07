


    # topo.py
import numpy as np
from ripser import ripser

try:
    from persim import wasserstein as _persim_wasserstein
except Exception:
    _persim_wasserstein = None


# ============================
# Generators (2D)
# ============================
def make_circle(n=200, r=1.0, noise=0.03, seed=7):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, size=n)
    x = r * np.cos(t) + rng.normal(scale=noise, size=n)
    y = r * np.sin(t) + rng.normal(scale=noise, size=n)
    return np.c_[x, y].astype(np.float32)

def make_two_circles(n=220, r=1.0, sep=2.6, noise=0.03, seed=7):
    n1 = n // 2
    p1 = make_circle(n=n1, r=r, noise=noise, seed=seed)
    p2 = make_circle(n=n - n1, r=r, noise=noise, seed=seed + 1)
    p1[:, 0] -= sep / 2
    p2[:, 0] += sep / 2
    return np.vstack([p1, p2]).astype(np.float32)

def make_blob_2d(n=200, scale=0.6, noise=0.0, seed=7):
    rng = np.random.default_rng(seed)
    p = rng.normal(scale=scale, size=(n, 2))
    if noise > 0:
        p += rng.normal(scale=noise, size=p.shape)
    return p.astype(np.float32)

def make_figure8(n=240, r=1.0, sep=1.3, noise=0.03, seed=7):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    t1 = rng.uniform(0, 2 * np.pi, size=n1)
    t2 = rng.uniform(0, 2 * np.pi, size=n - n1)
    c1 = np.c_[r * np.cos(t1) - sep / 2, r * np.sin(t1)]
    c2 = np.c_[r * np.cos(t2) + sep / 2, r * np.sin(t2)]
    p = np.vstack([c1, c2]) + rng.normal(scale=noise, size=(n, 2))
    return p.astype(np.float32)


# ============================
# Generators (3D)
# ============================
def make_3d_blob(n=260, scale=0.7, noise=0.0, seed=7):
    rng = np.random.default_rng(seed)
    p = rng.normal(scale=scale, size=(n, 3))
    if noise > 0:
        p += rng.normal(scale=noise, size=p.shape)
    return p.astype(np.float32)

def make_sphere(n=260, r=1.0, noise=0.02, seed=7):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, size=n)
    v = rng.uniform(0, 1, size=n)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    p = np.c_[x, y, z] + rng.normal(scale=noise, size=(n, 3))
    return p.astype(np.float32)

def make_torus(n=320, R=1.3, r=0.45, noise=0.02, seed=7):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 2 * np.pi, size=n)
    v = rng.uniform(0, 2 * np.pi, size=n)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    p = np.c_[x, y, z] + rng.normal(scale=noise, size=(n, 3))
    return p.astype(np.float32)

SHAPES_2D = {
    "2D blob": make_blob_2d,
    "2D circle": make_circle,
    "2D two circles": make_two_circles,
    "2D figure-8": make_figure8,
}
SHAPES_3D = {
    "3D blob": make_3d_blob,
    "3D sphere": make_sphere,
    "3D torus": make_torus,
}


# ============================
# Mixed dataset
# ============================
def make_dataset_mixed(n_samples=400, n_points=140, noise_2d=0.03, noise_3d=0.02, p_3d=0.45, seed=7):
    rng = np.random.default_rng(seed)
    keys2 = list(SHAPES_2D.keys())
    keys3 = list(SHAPES_3D.keys())
    class_names = keys2 + keys3

    X, y = [], []
    for _ in range(int(n_samples)):
        is3 = (rng.uniform() < float(p_3d))
        if is3:
            cls = int(rng.integers(0, len(keys3)))
            name = keys3[cls]
            pts = SHAPES_3D[name](n=int(n_points), noise=float(noise_3d), seed=int(rng.integers(0, 10_000)))
            label = len(keys2) + cls
        else:
            cls = int(rng.integers(0, len(keys2)))
            name = keys2[cls]
            pts = SHAPES_2D[name](n=int(n_points), noise=float(noise_2d), seed=int(rng.integers(0, 10_000)))
            label = cls
        X.append(pts)
        y.append(label)

    return X, np.array(y, dtype=np.int64), class_names


# ============================
# Persistent homology
# ============================
def persistence_diagrams(points, maxdim=2):
    out = ripser(points, maxdim=int(maxdim))
    return out["dgms"]


# ============================
# Diagram utilities
# ============================
def finite_bars(dgm):
    if dgm is None or len(dgm) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    finite = np.isfinite(dgm[:, 1])
    return dgm[finite].astype(np.float32)

def lifetimes(dgm):
    bars = finite_bars(dgm)
    if len(bars) == 0:
        return np.zeros((0,), dtype=np.float32)
    lt = (bars[:, 1] - bars[:, 0]).astype(np.float32)
    lt = lt[lt > 0]
    return np.sort(lt)[::-1]

def persistence_entropy(dgm, eps=1e-12):
    lt = lifetimes(dgm)
    if lt.size == 0:
        return 0.0
    p = lt / (lt.sum() + eps)
    return float(-(p * np.log(p + eps)).sum())

def total_persistence(dgm, power=1):
    lt = lifetimes(dgm)
    if lt.size == 0:
        return 0.0
    return float((lt ** int(power)).sum())

def topk_lifetimes(dgm, k=8):
    lt = lifetimes(dgm)
    out = np.zeros((int(k),), dtype=np.float32)
    m = min(int(k), int(lt.size))
    if m > 0:
        out[:m] = lt[:m]
    return out

def silhouette_curve(dgm, grid, p=1.0, eps=1e-12):
    bars = finite_bars(dgm)
    if len(bars) == 0:
        return np.zeros_like(grid, dtype=np.float32)
    b = bars[:, 0]
    d = bars[:, 1]
    lt = (d - b).astype(np.float32)
    w = (lt ** float(p))
    wsum = float(w.sum() + eps)

    s = np.zeros_like(grid, dtype=np.float32)
    for bi, di, wi in zip(b, d, w):
        left = grid - bi
        right = di - grid
        tri = np.minimum(left, right)
        tri = np.maximum(tri, 0.0).astype(np.float32)
        s += (wi / wsum) * tri
    return s

def landscape_samples(dgm, grid, k_levels=3):
    bars = finite_bars(dgm)
    if len(bars) == 0:
        return np.zeros((int(k_levels), len(grid)), dtype=np.float32)
    b = bars[:, 0]
    d = bars[:, 1]
    tents = []
    for bi, di in zip(b, d):
        left = grid - bi
        right = di - grid
        tri = np.minimum(left, right)
        tri = np.maximum(tri, 0.0).astype(np.float32)
        tents.append(tri)
    T = np.stack(tents, axis=0)  # [bars, G]
    Tsort = np.sort(T, axis=0)[::-1]
    return Tsort[: int(k_levels), :]

def tda_feature_block(dgms, grid_len=64, topk=8, k_levels=3):
    all_bd = []
    for dgm in dgms[:3]:
        bars = finite_bars(dgm)
        if len(bars):
            all_bd.append(bars)

    if len(all_bd) == 0:
        grid = np.linspace(0.0, 1.0, int(grid_len), dtype=np.float32)
    else:
        bd = np.vstack(all_bd)
        lo = float(np.min(bd[:, 0]))
        hi = float(np.max(bd[:, 1]))
        hi = max(hi, lo + 1e-3)
        grid = np.linspace(max(0.0, lo), hi, int(grid_len), dtype=np.float32)

    blocks = []
    for dim in range(3):
        dgm = dgms[dim] if dim < len(dgms) else np.zeros((0, 2), dtype=np.float32)
        ent = np.array([persistence_entropy(dgm)], dtype=np.float32)
        tp1 = np.array([total_persistence(dgm, power=1)], dtype=np.float32)
        tp2 = np.array([total_persistence(dgm, power=2)], dtype=np.float32)
        tkl = topk_lifetimes(dgm, k=int(topk))
        sil = silhouette_curve(dgm, grid, p=1.0).astype(np.float32)
        land = landscape_samples(dgm, grid, k_levels=int(k_levels)).astype(np.float32).ravel()
        blocks.append(np.concatenate([ent, tp1, tp2, tkl, sil, land], axis=0))

    return np.concatenate(blocks, axis=0).astype(np.float32)


# ============================
# Fixed-length persistence images (always 32x32 per channel)
# ============================
class PIModel:
    def __init__(self, birth_max=1.0, pers_max=1.0, sigma=0.15, nb=32, npers=32):
        self.birth_max = float(birth_max)
        self.pers_max = float(pers_max)
        self.sigma = float(sigma)
        self.nb = int(nb)
        self.npers = int(npers)

def _infer_ranges(dgms_dim):
    births = []
    pers = []
    for dgm in dgms_dim:
        bars = finite_bars(dgm)
        if len(bars) == 0:
            continue
        births.append(bars[:, 0])
        pers.append(bars[:, 1] - bars[:, 0])
    if not births:
        return 1.0, 1.0
    b = np.concatenate(births).astype(np.float64)
    p = np.concatenate(pers).astype(np.float64)
    bmax = float(np.quantile(b, 0.98))
    pmax = float(np.quantile(p, 0.98))
    return max(bmax, 1.0), max(pmax, 1.0)

def fit_imagers_multiscale(dgms, pixel_sizes=(0.03, 0.05, 0.08)):
    # pixel_sizes only determines how many channels you use; each channel is fixed 32x32
    pixel_sizes = tuple(pixel_sizes) if len(tuple(pixel_sizes)) else (0.05,)

    h1 = [d[1] for d in dgms]
    h2 = [d[2] for d in dgms]
    bmax1, pmax1 = _infer_ranges(h1)
    bmax2, pmax2 = _infer_ranges(h2)

    pims_h1 = [PIModel(birth_max=bmax1, pers_max=pmax1, sigma=0.15, nb=32, npers=32) for _ in pixel_sizes]
    pims_h2 = [PIModel(birth_max=bmax2, pers_max=pmax2, sigma=0.15, nb=32, npers=32) for _ in pixel_sizes]
    return pims_h1, pims_h2

def diagram_to_pi(pim: PIModel, dgm):
    bars = finite_bars(dgm)
    if len(bars) == 0:
        return np.zeros((pim.npers * pim.nb,), dtype=np.float32)

    bp = np.c_[bars[:, 0], (bars[:, 1] - bars[:, 0])].astype(np.float32)

    bg = np.linspace(0.0, pim.birth_max, pim.nb, dtype=np.float32)
    pg = np.linspace(0.0, pim.pers_max, pim.npers, dtype=np.float32)
    B, P = np.meshgrid(bg, pg, indexing="xy")

    Z = np.zeros_like(B, dtype=np.float32)
    s2 = float(pim.sigma) ** 2
    for (b, pers) in bp:
        Z += np.exp(-((B - b) ** 2 + (P - pers) ** 2) / (2.0 * s2)).astype(np.float32)

    Z = Z / (float(len(bp)) + 1e-12)
    return Z.ravel().astype(np.float32)

def diagram_to_pis(pims, dgm):
    vecs = [diagram_to_pi(pim, dgm) for pim in pims]
    return np.concatenate(vecs, axis=0).astype(np.float32)


# ============================
# Diagram distances (robust)
# ============================
def _sw_1d(u, v):
    u = np.sort(u.astype(np.float32))
    v = np.sort(v.astype(np.float32))
    m = max(len(u), len(v))
    if m == 0:
        return 0.0
    if len(u) != m:
        u = np.interp(np.linspace(0, 1, m), np.linspace(0, 1, max(1, len(u))), u if len(u) else np.array([0.0], dtype=np.float32)).astype(np.float32)
    if len(v) != m:
        v = np.interp(np.linspace(0, 1, m), np.linspace(0, 1, max(1, len(v))), v if len(v) else np.array([0.0], dtype=np.float32)).astype(np.float32)
    return float(np.mean(np.abs(u - v)))

def sliced_wasserstein(dgm_a, dgm_b, n_dirs=30, seed=0):
    rng = np.random.default_rng(seed)
    A = finite_bars(dgm_a)
    B = finite_bars(dgm_b)
    if len(A):
        A = np.c_[A[:, 0], (A[:, 1] - A[:, 0])]
    if len(B):
        B = np.c_[B[:, 0], (B[:, 1] - B[:, 0])]
    if len(A) == 0 and len(B) == 0:
        return 0.0
    ds = []
    for _ in range(int(n_dirs)):
        v = rng.normal(size=(2,))
        v = v / (np.linalg.norm(v) + 1e-12)
        a1 = (A @ v).astype(np.float32) if len(A) else np.zeros((0,), dtype=np.float32)
        b1 = (B @ v).astype(np.float32) if len(B) else np.zeros((0,), dtype=np.float32)
        ds.append(_sw_1d(a1, b1))
    return float(np.mean(ds))

def safe_wasserstein(dgm_a, dgm_b, order=1, internal_p=2):
    if _persim_wasserstein is None:
        return sliced_wasserstein(dgm_a, dgm_b, n_dirs=30, seed=0)
    a = finite_bars(dgm_a)
    b = finite_bars(dgm_b)
    try:
        return float(_persim_wasserstein(a, b, matching=False, order=order, internal_p=internal_p))
    except TypeError:
        try:
            return float(_persim_wasserstein(a, b, order=order))
        except TypeError:
            return float(_persim_wasserstein(a, b))

def prototype_diagrams(dgms, y, n_classes, dim=1, cap_per_class=20, seed=0, metric="wasserstein"):
    rng = np.random.default_rng(seed)
    protos = []
    for c in range(int(n_classes)):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            protos.append(np.zeros((0, 2), dtype=np.float32))
            continue
        if len(idx) > int(cap_per_class):
            idx = rng.choice(idx, size=int(cap_per_class), replace=False)
        H = [dgms[i][int(dim)] for i in idx]
        m = len(H)
        if m == 1:
            protos.append(H[0])
            continue
        D = np.zeros((m, m), dtype=np.float32)
        for i in range(m):
            for j in range(i + 1, m):
                if metric == "sliced":
                    d = sliced_wasserstein(H[i], H[j], n_dirs=30, seed=seed + i * 1000 + j)
                else:
                    d = safe_wasserstein(H[i], H[j])
                D[i, j] = d
                D[j, i] = d
        medoid = int(np.argmin(D.mean(axis=1)))
        protos.append(H[medoid])
    return protos

def distances_to_prototypes(dgm, protos, metric="wasserstein", seed=0):
    out = []
    for i, p in enumerate(protos):
        if metric == "sliced":
            out.append(sliced_wasserstein(dgm, p, n_dirs=30, seed=seed + 777 * i))
        else:
            out.append(safe_wasserstein(dgm, p))
    return np.array(out, dtype=np.float32)


# ============================
# Density summaries (fixed length)
# ============================
def knn_radius(points, k=10):
    X = np.asarray(points, dtype=np.float32)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    nn = np.partition(D, kth=int(k) - 1, axis=1)[:, int(k) - 1]
    return nn.astype(np.float32)

def density_filtration_summaries(points, fracs=(0.5, 0.8, 1.0), k=10, maxdim=2):
    X = np.asarray(points, dtype=np.float32)
    r = knn_radius(X, k=int(k))
    order = np.argsort(r)
    out = []
    for f in fracs:
        m = max(12, int(np.floor(float(f) * X.shape[0])))
        idx = order[:m]
        dg = persistence_diagrams(X[idx], maxdim=int(maxdim))
        for dim in range(3):
            dgm = dg[dim] if dim < len(dg) else np.zeros((0, 2), dtype=np.float32)
            out.append(float(persistence_entropy(dgm)))
            out.append(float(total_persistence(dgm, power=1)))
            out.extend(topk_lifetimes(dgm, k=6).tolist())
    return np.array(out, dtype=np.float32)

