
       # topo.py
import numpy as np
from ripser import ripser
from persim import PersistenceImager, wasserstein

try:
    from persim import bottleneck
except Exception:
    bottleneck = None

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

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
def make_dataset_mixed(n_samples=600, n_points=160, noise_2d=0.03, noise_3d=0.02, p_3d=0.45, seed=7):
    rng = np.random.default_rng(seed)
    keys2 = list(SHAPES_2D.keys())
    keys3 = list(SHAPES_3D.keys())
    class_names = keys2 + keys3
    X, y = [], []
    for _ in range(n_samples):
        is3 = (rng.uniform() < p_3d)
        if is3:
            cls = int(rng.integers(0, len(keys3)))
            name = keys3[cls]
            gen = SHAPES_3D[name]
            pts = gen(n=n_points, noise=noise_3d, seed=int(rng.integers(0, 10_000)))
            label = len(keys2) + cls
        else:
            cls = int(rng.integers(0, len(keys2)))
            name = keys2[cls]
            gen = SHAPES_2D[name]
            pts = gen(n=n_points, noise=noise_2d, seed=int(rng.integers(0, 10_000)))
            label = cls
        X.append(pts)
        y.append(label)
    return X, np.array(y, dtype=np.int64), class_names

# ============================
# PH (Euclidean) + PH (Geodesic kNN metric)
# ============================
def persistence(points, maxdim=2, do_cocycles=False, coeff=47):
    return ripser(points, maxdim=maxdim, do_cocycles=do_cocycles, coeff=int(coeff))

def persistence_diagrams(points, maxdim=2):
    out = persistence(points, maxdim=maxdim, do_cocycles=False)
    return out["dgms"]

def _knn_adjacency(points, k=10):
    X = np.asarray(points, dtype=np.float32)
    n = X.shape[0]
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, 0.0)
    adj = [[] for _ in range(n)]
    for i in range(n):
        nn = np.argsort(D[i])[1 : k + 1]
        for j in nn:
            w = float(D[i, j])
            adj[i].append((int(j), w))
            adj[int(j)].append((i, w))
    return adj

def _all_pairs_dijkstra(adj):
    import heapq
    n = len(adj)
    distmat = np.full((n, n), np.inf, dtype=np.float32)
    for s in range(n):
        dist = np.full((n,), np.inf, dtype=np.float32)
        dist[s] = 0.0
        visited = np.zeros((n,), dtype=np.uint8)
        heap = [(0.0, s)]
        while heap:
            du, u = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = 1
            for v, w in adj[u]:
                nd = du + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (float(nd), int(v)))
        distmat[s] = dist
    max_finite = float(np.nanmax(distmat[np.isfinite(distmat)])) if np.any(np.isfinite(distmat)) else 1.0
    distmat[~np.isfinite(distmat)] = max_finite * 2.0
    np.fill_diagonal(distmat, 0.0)
    return distmat

def persistence_diagrams_geodesic(points, k=10, maxdim=2):
    adj = _knn_adjacency(points, k=int(k))
    G = _all_pairs_dijkstra(adj)
    out = ripser(G, distance_matrix=True, maxdim=maxdim)
    return out["dgms"], G

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
    return float((lt ** power).sum())

def topk_lifetimes(dgm, k=8):
    lt = lifetimes(dgm)
    out = np.zeros((k,), dtype=np.float32)
    m = min(k, lt.size)
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
    w = lt ** p
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
        return np.zeros((k_levels, len(grid)), dtype=np.float32)
    b = bars[:, 0]
    d = bars[:, 1]
    tents = []
    for bi, di in zip(b, d):
        left = grid - bi
        right = di - grid
        tri = np.minimum(left, right)
        tri = np.maximum(tri, 0.0).astype(np.float32)
        tents.append(tri)
    T = np.stack(tents, axis=0)
    Tsort = np.sort(T, axis=0)[::-1]
    return Tsort[:k_levels, :]

def tda_feature_block(dgms, grid_len=64, topk=8, k_levels=3):
    all_bd = []
    for dgm in dgms[:3]:
        bars = finite_bars(dgm)
        if len(bars):
            all_bd.append(bars)
    if len(all_bd) == 0:
        grid = np.linspace(0.0, 1.0, grid_len).astype(np.float32)
    else:
        bd = np.vstack(all_bd)
        lo = float(np.min(bd[:, 0]))
        hi = float(np.max(bd[:, 1]))
        hi = max(hi, lo + 1e-3)
        grid = np.linspace(max(0.0, lo), hi, grid_len).astype(np.float32)

    blocks = []
    for dim in range(3):
        dgm = dgms[dim] if dim < len(dgms) else np.zeros((0, 2), dtype=np.float32)
        ent = np.array([persistence_entropy(dgm)], dtype=np.float32)
        tp1 = np.array([total_persistence(dgm, power=1)], dtype=np.float32)
        tp2 = np.array([total_persistence(dgm, power=2)], dtype=np.float32)
        tkl = topk_lifetimes(dgm, k=topk)
        sil = silhouette_curve(dgm, grid, p=1.0).astype(np.float32)
        land = landscape_samples(dgm, grid, k_levels=k_levels).astype(np.float32).ravel()
        blocks.append(np.concatenate([ent, tp1, tp2, tkl, sil, land], axis=0))
    return np.concatenate(blocks, axis=0)

# ============================
# Persistence images (multi-scale, H1/H2)
# ============================
def _fit_imager(train_dgms, pixel_size=0.05):
    births, deaths = [], []
    for dgm in train_dgms:
        bars = finite_bars(dgm)
        if len(bars) == 0:
            continue
        births.append(bars[:, 0])
        deaths.append(bars[:, 1])

    if len(births) == 0:
        birth_range, pers_range = (0.0, 1.0), (0.0, 1.0)
    else:
        b = np.concatenate(births)
        d = np.concatenate(deaths)
        p = d - b
        b0, b1 = np.quantile(b, [0.02, 0.98])
        p0, p1 = np.quantile(p, [0.02, 0.98])
        birth_range = (float(max(0.0, b0)), float(b1))
        pers_range = (float(max(0.0, p0)), float(p1))

    pim = PersistenceImager(pixel_size=float(pixel_size))

    try:
        pim.fit(birth_range=birth_range, pers_range=pers_range)
    except TypeError:
        bmin, bmax = birth_range
        pmin, pmax = pers_range
        fake = np.array([[bmin, bmin + pmin], [bmax, bmax + pmax]], dtype=np.float32)
        try:
            pim.fit(fake)
        except Exception:
            pim.fit([fake])

    return pim

def fit_imagers_multiscale(dgms, pixel_sizes=(0.03, 0.05, 0.08)):
    h1 = [d[1] for d in dgms]
    h2 = [d[2] for d in dgms]
    pim_h1 = [_fit_imager(h1, ps) for ps in pixel_sizes]
    pim_h2 = [_fit_imager(h2, ps) for ps in pixel_sizes]
    return pim_h1, pim_h2

def diagram_to_pi(pim, dgm):
    bars = finite_bars(dgm)
    if len(bars) == 0:
        img = np.zeros(pim.resolution)
        return img.ravel()
    return pim.transform(bars).ravel()

def diagram_to_pis(pims, dgm):
    out = []
    for pim in pims:
        out.append(diagram_to_pi(pim, dgm))
    return np.concatenate(out, axis=0).astype(np.float32)

# ============================
# Diagram distances (Wasserstein/Bottleneck/Sliced-Wasserstein)
# ============================
def safe_wasserstein(dgm_a, dgm_b, order=1, internal_p=2):
    a = finite_bars(dgm_a)
    b = finite_bars(dgm_b)
    return float(wasserstein(a, b, matching=False, order=order, internal_p=internal_p))

def safe_bottleneck(dgm_a, dgm_b):
    if bottleneck is None:
        return 0.0
    a = finite_bars(dgm_a)
    b = finite_bars(dgm_b)
    return float(bottleneck(a, b))

def _sw_1d(u, v):
    u = np.sort(u.astype(np.float32))
    v = np.sort(v.astype(np.float32))
    m = max(len(u), len(v))
    if m == 0:
        return 0.0
    if len(u) != m:
        u = np.interp(
            np.linspace(0, 1, m),
            np.linspace(0, 1, len(u) if len(u) else 1),
            u if len(u) else np.array([0.0], dtype=np.float32),
        ).astype(np.float32)
    if len(v) != m:
        v = np.interp(
            np.linspace(0, 1, m),
            np.linspace(0, 1, len(v) if len(v) else 1),
            v if len(v) else np.array([0.0], dtype=np.float32),
        ).astype(np.float32)
    return float(np.mean(np.abs(u - v)))

def sliced_wasserstein(dgm_a, dgm_b, n_dirs=50, seed=0, use_birth_persist=True):
    rng = np.random.default_rng(seed)
    A = finite_bars(dgm_a)
    B = finite_bars(dgm_b)
    if use_birth_persist:
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

def prototype_diagrams(dgms, y, n_classes, dim=1, cap_per_class=30, seed=0, metric="wasserstein"):
    rng = np.random.default_rng(seed)
    protos = []
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            protos.append(np.zeros((0, 2), dtype=np.float32))
            continue
        if len(idx) > cap_per_class:
            idx = rng.choice(idx, size=cap_per_class, replace=False)
        H = [dgms[i][dim] for i in idx]
        m = len(H)
        if m == 1:
            protos.append(H[0])
            continue
        D = np.zeros((m, m), dtype=np.float32)
        for i in range(m):
            for j in range(i + 1, m):
                if metric == "bottleneck" and bottleneck is not None:
                    d = safe_bottleneck(H[i], H[j])
                elif metric == "sliced":
                    d = sliced_wasserstein(H[i], H[j], n_dirs=40, seed=seed + i * 1000 + j)
                else:
                    d = safe_wasserstein(H[i], H[j], order=1, internal_p=2)
                D[i, j] = d
                D[j, i] = d
        medoid = int(np.argmin(D.mean(axis=1)))
        protos.append(H[medoid])
    return protos

def distances_to_prototypes(dgm, protos, metric="wasserstein", seed=0):
    if metric == "bottleneck" and bottleneck is not None:
        return np.array([safe_bottleneck(dgm, p) for p in protos], dtype=np.float32)
    if metric == "sliced":
        return np.array(
            [sliced_wasserstein(dgm, p, n_dirs=40, seed=seed + 777 * i) for i, p in enumerate(protos)],
            dtype=np.float32,
        )
    return np.array([safe_wasserstein(dgm, p, order=1, internal_p=2) for p in protos], dtype=np.float32)

# ============================
# Persistence Scale-Space Kernel features
# ============================
def pss_kernel_features(dgm, grid_birth=(0.0, 2.5), grid_pers=(0.0, 2.5), nb=32, npers=32, sigma=0.15):
    bars = finite_bars(dgm)
    if len(bars) == 0:
        return np.zeros((nb * npers,), dtype=np.float32)
    bp = np.c_[bars[:, 0], (bars[:, 1] - bars[:, 0])].astype(np.float32)
    bmin, bmax = float(grid_birth[0]), float(grid_birth[1])
    pmin, pmax = float(grid_pers[0]), float(grid_pers[1])
    bg = np.linspace(bmin, bmax, int(nb), dtype=np.float32)
    pg = np.linspace(pmin, pmax, int(npers), dtype=np.float32)
    B, P = np.meshgrid(bg, pg, indexing="xy")
    Z = np.zeros_like(B, dtype=np.float32)
    s2 = float(sigma) ** 2
    for (b, pers) in bp:
        Z += np.exp(-((B - b) ** 2 + (P - pers) ** 2) / (2.0 * s2)).astype(np.float32)
    Z = Z / (float(len(bp)) + 1e-12)
    return Z.ravel().astype(np.float32)

# ============================
# Density-filtration proxy
# ============================
def knn_radius(points, k=10):
    X = np.asarray(points, dtype=np.float32)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    nn = np.partition(D, kth=int(k) - 1, axis=1)[:, int(k) - 1]
    return nn.astype(np.float32)

def density_filtration_summaries(points, fracs=(0.4, 0.6, 0.8, 1.0), k=10, maxdim=2):
    X = np.asarray(points, dtype=np.float32)
    r = knn_radius(X, k=int(k))
    order = np.argsort(r)
    out = []
    for f in fracs:
        m = max(8, int(np.floor(float(f) * X.shape[0])))
        idx = order[:m]
        dg = persistence_diagrams(X[idx], maxdim=maxdim)
        for dim in range(3):
            dgm = dg[dim] if dim < len(dg) else np.zeros((0, 2), dtype=np.float32)
            out.append(float(persistence_entropy(dgm)))
            out.append(float(total_persistence(dgm, power=1)))
            out.append(float(total_persistence(dgm, power=2)))
            out.extend(topk_lifetimes(dgm, k=6).tolist())
    return np.array(out, dtype=np.float32)

# ============================
# Circular coordinates (H1 cohomology)
# ============================
def _extract_most_persistent_h1(out):
    dgms = out["dgms"]
    if len(dgms) < 2:
        return None, None
    dgm1 = dgms[1]
    bars = finite_bars(dgm1)
    if len(bars) == 0:
        return None, None
    pers = bars[:, 1] - bars[:, 0]
    mx = int(np.argmax(pers))
    target = bars[mx]
    idx = None
    for i in range(len(dgm1)):
        if np.isfinite(dgm1[i, 1]) and float(dgm1[i, 0]) == float(target[0]) and float(dgm1[i, 1]) == float(target[1]):
            idx = i
            break
    return idx, target

def _cocycle_edges(cocycle, p):
    cocycle = np.asarray(cocycle, dtype=np.int64)
    if cocycle.ndim != 2 or cocycle.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.int64)
    cocycle[:, 2] = cocycle[:, 2] % int(p)
    return cocycle

def _solve_theta(n, edges_ijw, p, anchor=0):
    if len(edges_ijw) == 0:
        return np.zeros((n,), dtype=np.float32)
    m = len(edges_ijw)
    A = np.zeros((m + 1, n), dtype=np.float32)
    b = np.zeros((m + 1,), dtype=np.float32)
    for k, (i, j, w) in enumerate(edges_ijw):
        i = int(i); j = int(j); w = int(w)
        A[k, j] = 1.0
        A[k, i] = -1.0
        b[k] = (2.0 * np.pi) * (float(w) / float(p))
    A[m, int(anchor)] = 1.0
    b[m] = 0.0
    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    theta = np.asarray(theta, dtype=np.float32)
    theta = np.mod(theta, 2.0 * np.pi)
    return theta

def circular_coordinates(points, coeff=47, eps=None):
    X = np.asarray(points, dtype=np.float32)
    out = persistence(X, maxdim=1, do_cocycles=True, coeff=int(coeff))
    idx, bar = _extract_most_persistent_h1(out)
    if idx is None:
        return None, None, None, out
    birth = float(bar[0])
    death = float(bar[1])
    if eps is None:
        eps = float(0.5 * (birth + death))
    cocycles = out.get("cocycles", None)
    if cocycles is None or len(cocycles) < 2 or len(cocycles[1]) == 0:
        return None, birth, death, out
    cocycle = cocycles[1][idx]
    E = _cocycle_edges(cocycle, int(coeff))
    theta = _solve_theta(n=X.shape[0], edges_ijw=E, p=int(coeff), anchor=0)
    return theta, birth, death, out

# ============================
# Mapper + spectral invariants
# ============================
def lens_pca(points, n_components=2):
    X = np.asarray(points, dtype=np.float32)
    k = min(int(n_components), X.shape[1], X.shape[0])
    pca = PCA(n_components=k, random_state=0)
    Z = pca.fit_transform(X)
    if Z.shape[1] < int(n_components):
        pad = np.zeros((Z.shape[0], int(n_components) - Z.shape[1]), dtype=np.float32)
        Z = np.hstack([Z, pad])
    return Z.astype(np.float32)

def cover_intervals(vals, n_intervals=10, overlap=0.3):
    vals = np.asarray(vals, dtype=np.float32).ravel()
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmax <= vmin + 1e-12:
        return [(vmin, vmax)]
    length = (vmax - vmin) / float(n_intervals)
    step = length * (1.0 - float(overlap))
    if step <= 1e-12:
        step = length * 0.5
    intervals = []
    start = vmin
    for _ in range(int(n_intervals)):
        end = start + length
        intervals.append((start, end))
        start += step
        if start >= vmax:
            break
    return intervals

def mapper_graph(points, lens, n_intervals=10, overlap=0.3, dbscan_eps=0.25, min_samples=5):
    X = np.asarray(points, dtype=np.float32)
    lens = np.asarray(lens, dtype=np.float32)
    if lens.ndim == 1:
        lens = lens.reshape(-1, 1)
    n = X.shape[0]
    d_lens = lens.shape[1]

    covers = [cover_intervals(lens[:, j], n_intervals=n_intervals, overlap=overlap) for j in range(d_lens)]
    boxes = []

    def rec_build(j, cur):
        if j == d_lens:
            boxes.append(list(cur))
            return
        for itv in covers[j]:
            cur.append(itv)
            rec_build(j + 1, cur)
            cur.pop()

    rec_build(0, [])

    nodes = []
    node_points = []
    node_lens_mean = []
    node_center = []
    node_cover_id = []

    def in_box(i, box):
        for j, (a, b) in enumerate(box):
            v = float(lens[i, j])
            if not (v >= a and v <= b):
                return False
        return True

    for b_id, box in enumerate(boxes):
        idx = [i for i in range(n) if in_box(i, box)]
        if len(idx) == 0:
            continue
        Xi = X[idx]
        cl = DBSCAN(eps=float(dbscan_eps), min_samples=int(min_samples)).fit(Xi).labels_
        labs = sorted(set(int(t) for t in cl if int(t) != -1))
        for lab in labs:
            pts_idx_local = np.where(cl == lab)[0]
            pts_idx = [idx[int(k)] for k in pts_idx_local]
            nodes.append(len(nodes))
            node_points.append(np.array(pts_idx, dtype=np.int64))
            node_lens_mean.append(np.mean(lens[pts_idx], axis=0).astype(np.float32))
            node_center.append(np.mean(X[pts_idx], axis=0).astype(np.float32))
            node_cover_id.append(int(b_id))

    edges = set()
    for i in range(len(node_points)):
        set_i = set(int(x) for x in node_points[i].tolist())
        for j in range(i + 1, len(node_points)):
            if len(set_i.intersection(int(x) for x in node_points[j].tolist())) > 0:
                edges.add((i, j))

    return {
        "nodes": list(range(len(node_points))),
        "node_points": node_points,
        "node_lens_mean": node_lens_mean,
        "node_center": node_center,
        "node_cover_id": node_cover_id,
        "edges": sorted(list(edges)),
    }

def mapper_spectral_features(G, k_eigs=12):
    n = len(G["nodes"])
    if n == 0:
        return np.zeros((k_eigs + 6,), dtype=np.float32)
    A = np.zeros((n, n), dtype=np.float32)
    for (i, j) in G["edges"]:
        A[int(i), int(j)] = 1.0
        A[int(j), int(i)] = 1.0
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    try:
        w = np.linalg.eigvalsh(L).astype(np.float32)
        w = np.sort(w)
    except Exception:
        w = np.zeros((n,), dtype=np.float32)
    out = np.zeros((k_eigs,), dtype=np.float32)
    m = min(int(k_eigs), len(w))
    if m > 0:
        out[:m] = w[:m]
    n_edges = float(len(G["edges"]))
    n_nodes = float(n)
    avg_deg = float(deg.mean()) if n else 0.0
    max_deg = float(deg.max()) if n else 0.0
    conn = float((w[1] if len(w) > 1 else 0.0))
    tri = float(np.trace(A @ A @ A) / 6.0) if n else 0.0
    stats = np.array([n_nodes, n_edges, avg_deg, max_deg, conn, tri], dtype=np.float32)
    return np.concatenate([out, stats], axis=0).astype(np.float32)
