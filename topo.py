


  
    
       
  
   
    # topo.py
import numpy as np
from ripser import ripser
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


# ============================
# Generators (2D)
# ============================
def make_circle(n=200, r=1.0, noise=0.03, seed=7):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2*np.pi, size=n)
    x = r*np.cos(t) + rng.normal(scale=noise, size=n)
    y = r*np.sin(t) + rng.normal(scale=noise, size=n)
    return np.c_[x, y].astype(np.float32)

def make_two_circles(n=220, r=1.0, sep=2.6, noise=0.03, seed=7):
    n1 = n // 2
    p1 = make_circle(n=n1, r=r, noise=noise, seed=seed)
    p2 = make_circle(n=n-n1, r=r, noise=noise, seed=seed+1)
    p1[:, 0] -= sep/2
    p2[:, 0] += sep/2
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
    t1 = rng.uniform(0, 2*np.pi, size=n1)
    t2 = rng.uniform(0, 2*np.pi, size=n-n1)
    c1 = np.c_[r*np.cos(t1) - sep/2, r*np.sin(t1)]
    c2 = np.c_[r*np.cos(t2) + sep/2, r*np.sin(t2)]
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
    theta = 2*np.pi*u
    phi = np.arccos(2*v - 1)
    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)
    p = np.c_[x, y, z] + rng.normal(scale=noise, size=(n, 3))
    return p.astype(np.float32)

def make_torus(n=320, R=1.3, r=0.45, noise=0.02, seed=7):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 2*np.pi, size=n)
    v = rng.uniform(0, 2*np.pi, size=n)
    x = (R + r*np.cos(v)) * np.cos(u)
    y = (R + r*np.cos(v)) * np.sin(u)
    z = r*np.sin(v)
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
def make_dataset_mixed(n_samples=250, n_points=140, noise_2d=0.03, noise_3d=0.02, p_3d=0.45, seed=7):
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
def dgms_only(points, maxdim=2):
    out = ripser(np.asarray(points, dtype=np.float32), maxdim=int(maxdim))
    return out["dgms"]


# ============================
# Geodesic PH
# ============================
def _knn_adjacency(points, k=10):
    X = np.asarray(points, dtype=np.float32)
    n = X.shape[0]
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, 0.0)
    adj = [[] for _ in range(n)]
    for i in range(n):
        nn = np.argsort(D[i])[1:k+1]
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
        vis = np.zeros((n,), dtype=np.uint8)
        heap = [(0.0, s)]
        while heap:
            du, u = heapq.heappop(heap)
            if vis[u]:
                continue
            vis[u] = 1
            for v, w in adj[u]:
                nd = du + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (float(nd), int(v)))
        distmat[s] = dist
    mf = float(np.nanmax(distmat[np.isfinite(distmat)])) if np.any(np.isfinite(distmat)) else 1.0
    distmat[~np.isfinite(distmat)] = mf * 2.0
    np.fill_diagonal(distmat, 0.0)
    return distmat

def dgms_geodesic(points, k=10, maxdim=2):
    adj = _knn_adjacency(points, k=int(k))
    G = _all_pairs_dijkstra(adj)
    out = ripser(G, distance_matrix=True, maxdim=int(maxdim))
    return out["dgms"], G


# ============================
# Diagram utilities / robust stats
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
        tri = np.minimum(grid - bi, di - grid)
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
        tri = np.minimum(grid - bi, di - grid)
        tri = np.maximum(tri, 0.0).astype(np.float32)
        tents.append(tri)
    T = np.stack(tents, axis=0)
    Tsort = np.sort(T, axis=0)[::-1]
    return Tsort[: int(k_levels), :]

def tda_feature_block(dgms, grid_len=64, topk=8, k_levels=3):
    grid = np.linspace(0.0, 3.0, int(grid_len), dtype=np.float32)
    blocks = []
    for dim in range(3):
        dgm = dgms[dim] if dim < len(dgms) else np.zeros((0, 2), dtype=np.float32)
        ent = np.array([persistence_entropy(dgm)], dtype=np.float32)
        tp1 = np.array([total_persistence(dgm, 1)], dtype=np.float32)
        tp2 = np.array([total_persistence(dgm, 2)], dtype=np.float32)
        tkl = topk_lifetimes(dgm, k=int(topk))
        sil = silhouette_curve(dgm, grid, p=1.0).astype(np.float32)
        land = landscape_samples(dgm, grid, k_levels=int(k_levels)).astype(np.float32).ravel()
        blocks.append(np.concatenate([ent, tp1, tp2, tkl, sil, land], axis=0))
    f = np.concatenate(blocks, axis=0).astype(np.float32)
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)


# ============================
# Fixed-length persistence images (32x32 per channel)
# ============================
class PIModel:
    def __init__(self, birth_max=3.0, pers_max=3.0, sigma=0.15, nb=32, npers=32):
        self.birth_max = float(birth_max)
        self.pers_max = float(pers_max)
        self.sigma = float(sigma)
        self.nb = int(nb)
        self.npers = int(npers)

def fit_imagers_multiscale(dgms, pixel_sizes=(0.05,)):
    pixel_sizes = tuple(pixel_sizes) if len(tuple(pixel_sizes)) else (0.05,)
    pims_h1 = [PIModel() for _ in pixel_sizes]
    pims_h2 = [PIModel() for _ in pixel_sizes]
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
    z = Z.ravel().astype(np.float32)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

def diagram_to_pis(pims, dgm):
    v = np.concatenate([diagram_to_pi(pim, dgm) for pim in pims], axis=0).astype(np.float32)
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


# ============================
# Sliced Wasserstein distances (pure numpy)
# ============================
def _sw_1d(u, v):
    u = np.sort(u.astype(np.float32))
    v = np.sort(v.astype(np.float32))
    m = max(len(u), len(v))
    if m == 0:
        return 0.0
    if len(u) != m:
        u = np.interp(np.linspace(0, 1, m), np.linspace(0, 1, max(1, len(u))),
                      u if len(u) else np.array([0.0], dtype=np.float32)).astype(np.float32)
    if len(v) != m:
        v = np.interp(np.linspace(0, 1, m), np.linspace(0, 1, max(1, len(v))),
                      v if len(v) else np.array([0.0], dtype=np.float32)).astype(np.float32)
    return float(np.mean(np.abs(u - v)))

def sliced_wasserstein(dgm_a, dgm_b, n_dirs=25, seed=0):
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

def prototype_diagrams(dgms, y, n_classes, dim=1, cap_per_class=20, seed=0):
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
            for j in range(i+1, m):
                d = sliced_wasserstein(H[i], H[j], n_dirs=25, seed=seed + i*1000 + j)
                D[i, j] = d
                D[j, i] = d
        medoid = int(np.argmin(D.mean(axis=1)))
        protos.append(H[medoid])
    return protos

def distances_to_prototypes(dgm, protos, seed=0):
    out = []
    for i, p in enumerate(protos):
        out.append(sliced_wasserstein(dgm, p, n_dirs=25, seed=seed + 777*i))
    v = np.array(out, dtype=np.float32)
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


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
        dg = dgms_only(X[idx], maxdim=int(maxdim))
        for dim in range(3):
            dgm = dg[dim] if dim < len(dg) else np.zeros((0, 2), dtype=np.float32)
            out.append(float(persistence_entropy(dgm)))
            out.append(float(total_persistence(dgm, 1)))
            out.append(float(total_persistence(dgm, 2)))
            out.extend(topk_lifetimes(dgm, k=6).tolist())
    v = np.array(out, dtype=np.float32)
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


# ============================
# Cohomology (safe)
# ============================
def circular_coordinates(points, coeff=47):
    X = np.asarray(points, dtype=np.float32)
    out = ripser(X, maxdim=1, do_cocycles=True, coeff=int(coeff))
    dgm1 = out["dgms"][1]
    bars = finite_bars(dgm1)
    if len(bars) == 0:
        return None, None, None
    pers = bars[:, 1] - bars[:, 0]
    mx = int(np.argmax(pers))
    birth = float(bars[mx, 0])
    death = float(bars[mx, 1])

    cocycles = out.get("cocycles", None)
    if cocycles is None or len(cocycles) < 2 or len(cocycles[1]) == 0:
        return None, birth, death

    try:
        cocycle = np.asarray(cocycles[1][mx], dtype=np.int64)
    except Exception:
        return None, birth, death

    if cocycle.ndim != 2 or cocycle.shape[1] != 3:
        return None, birth, death

    p = int(coeff)
    cocycle[:, 2] = cocycle[:, 2] % p

    n = X.shape[0]
    m = cocycle.shape[0]
    A = np.zeros((m + 1, n), dtype=np.float32)
    bvec = np.zeros((m + 1,), dtype=np.float32)
    for k, (i, j, w) in enumerate(cocycle):
        i = int(i); j = int(j); w = int(w)
        A[k, j] = 1.0
        A[k, i] = -1.0
        bvec[k] = (2.0 * np.pi) * (float(w) / float(p))
    A[m, 0] = 1.0
    bvec[m] = 0.0
    theta, *_ = np.linalg.lstsq(A, bvec, rcond=None)
    theta = np.mod(theta.astype(np.float32), 2.0*np.pi)
    return theta, birth, death


# ============================
# Mapper + spectral features
# ============================
def lens_pca(points, n_components=1):
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
    lens = np.asarray(lens, dtype=np.float32).reshape(-1)
    intervals = cover_intervals(lens, n_intervals=int(n_intervals), overlap=float(overlap))

    node_points = []
    for (a, b) in intervals:
        idx = np.where((lens >= a) & (lens <= b))[0]
        if idx.size == 0:
            continue
        Xi = X[idx]
        labels = DBSCAN(eps=float(dbscan_eps), min_samples=int(min_samples)).fit(Xi).labels_
        for lab in sorted(set(int(t) for t in labels if int(t) != -1)):
            pts_idx_local = np.where(labels == lab)[0]
            pts_idx = idx[pts_idx_local]
            node_points.append(np.array(pts_idx, dtype=np.int64))

    edges = set()
    for i in range(len(node_points)):
        si = set(node_points[i].tolist())
        for j in range(i+1, len(node_points)):
            if len(si.intersection(node_points[j].tolist())) > 0:
                edges.add((i, j))

    return {"nodes": list(range(len(node_points))), "node_points": node_points, "edges": sorted(list(edges))}

def mapper_spectral_features(G, k_eigs=12):
    n = len(G["nodes"])
    if n == 0:
        return np.zeros((int(k_eigs) + 6,), dtype=np.float32)
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
    out = np.zeros((int(k_eigs),), dtype=np.float32)
    m = min(int(k_eigs), int(len(w)))
    if m > 0:
        out[:m] = w[:m]
    stats = np.array(
        [float(n), float(len(G["edges"])), float(deg.mean()) if n else 0.0, float(deg.max()) if n else 0.0,
         float(w[1]) if len(w) > 1 else 0.0, float(np.trace(A @ A @ A) / 6.0) if n else 0.0],
        dtype=np.float32,
    )
    v = np.concatenate([out, stats], axis=0).astype(np.float32)
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)



      
    
    
