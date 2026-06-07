"""
Texturizar los 10 modelos nuevos (10modelos3d/N.ply) con sus try-on (10imagenes2d/<metodo>/N.png).

PROBLEMA: las 10 nubes vienen RE-NORMALIZADAS (todas con caja XY identica ±0.42/±0.97),
asi que la proyeccion exacta de M3D-VTON (w=(X+1)*256-95) ya no aplica.

SOLUCION: anclamos la nube al BOUNDING-BOX de la persona en su imagen (cabeza de la nube
-> cabeza de la foto, pies -> pies, lados -> lados). Asi la persona 3D queda COMPLETA y con
las proporciones correctas (cara, torso y pies en su sitio). Es aproximado -la normalizacion
deformo la nube, que ademas es mas ancha que la silueta-, pero respeta el cuerpo entero.
(Un intento previo maximizaba el % de puntos sobre la persona: daba 94% pero hacia trampa
encogiendo la nube al torso y se perdian cara y pies.)
Lo demas (frente/espalda, pelo, limpieza) sigue a texture_pcd.py.

Uso:
    python texture_10.py --method ssim_200
    python texture_10.py --method baseline --out resultados/10_baseline
"""
import argparse, os
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import distance_transform_edt

IMG_W, IMG_H = 320, 512


def img_mask(arr):
    return arr.astype(int).sum(2) > 30


def person_bbox(mask):
    """Bounding-box robusto de la persona (ignora motas sueltas)."""
    cols = mask.sum(0)
    rows = mask.sum(1)
    xs = np.where(cols > 2)[0]
    ys = np.where(rows > 2)[0]
    return xs.min(), xs.max(), ys.min(), ys.max()


def texture_one(n, method):
    pcd = o3d.io.read_point_cloud(f"10modelos3d/{n}.ply")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    V = np.asarray(pcd.points)
    X, Y, Z = V[:, 0], V[:, 1], V[:, 2]

    img = np.asarray(Image.open(f"10imagenes2d/{method}/{n}.png").convert("RGB"))
    mask = img_mask(img)

    # anclar extent de la nube (percentiles robustos) -> bbox de la persona
    wmin, wmax, hmin, hmax = person_bbox(mask)
    x0, x1 = np.percentile(X, [0.3, 99.7])
    y0, y1 = np.percentile(Y, [0.3, 99.7])
    w = wmin + (X - x0) / (x1 - x0) * (wmax - wmin)
    h = hmin + (y1 - Y) / (y1 - y0) * (hmax - hmin)  # Y grande = arriba = hmin
    wi = np.clip(np.round(w).astype(int), 0, IMG_W - 1)
    hi = np.clip(np.round(h).astype(int), 0, IMG_H - 1)
    cov = mask[hi, wi].mean()

    # puntos que caen FUERA de la persona (sobresalen al fondo negro): en vez de pintarlos
    # de negro, los llevamos al pixel-persona mas cercano -> quita el halo oscuro del borde.
    off = ~mask[hi, wi]
    if off.any():
        _, (ny, nx) = distance_transform_edt(~mask, return_indices=True)
        hi2 = np.where(off, ny[hi, wi], hi)
        wi2 = np.where(off, nx[hi, wi], wi)
        hi, wi = hi2, wi2
    col = img[hi, wi].astype(np.float64) / 255.0

    # frente (menor Z por pixel) vs espalda
    key = hi.astype(np.int64) * IMG_W + wi
    zmin = np.full(IMG_W * IMG_H, np.inf)
    zmax = np.full(IMG_W * IMG_H, -np.inf)
    np.minimum.at(zmin, key, Z)
    np.maximum.at(zmax, key, Z)
    mid = (zmin[key] + zmax[key]) / 2.0
    front = Z <= mid

    # espalda = frente oscurecido; cabeza trasera = pelo sintetico
    gray = col.mean(axis=1, keepdims=True)
    back = (col * 0.5 + gray * 0.5) * 0.45
    hmin, hmax = hi.min(), hi.max()
    head = hi < hmin + 0.16 * (hmax - hmin)
    top = hi < hmin + 0.06 * (hmax - hmin)
    if top.sum() > 0:
        hair = np.clip(np.median(col[top], axis=0), 0.04, 1.0)
        back[head] = hair
    col = np.where(front[:, None], col, back)

    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(V)
    p.colors = o3d.utility.Vector3dVector(np.clip(col, 0, 1))

    # limpieza: cluster mayor
    labels = np.array(p.cluster_dbscan(eps=0.04, min_points=10))
    if labels.max() >= 0:
        big = np.bincount(labels[labels >= 0]).argmax()
        p = p.select_by_index(np.where(labels == big)[0])
    return p, cov


def render(p, path, size=(300, 540), point_size=6.0):
    # la camara por defecto de open3d mira el lado +Z (la espalda sintetica);
    # giramos 180 sobre Y para mostrar el FRENTE (el try-on real).
    p = o3d.geometry.PointCloud(p)
    p.rotate(p.get_rotation_matrix_from_axis_angle(np.array([0, np.pi, 0])),
             center=p.get_center())
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=size[0], height=size[1])
    vis.add_geometry(p)
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array([1, 1, 1])
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(path, do_render=True)
    vis.destroy_window()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="ssim_200")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"resultados/10_{args.method}"
    os.makedirs(out, exist_ok=True)

    covs = []
    renders = []
    for n in range(1, 11):
        p, cov = texture_one(n, args.method)
        covs.append(cov)
        ply = f"{out}/recon_{n}_{args.method}_pcd.ply"
        o3d.io.write_point_cloud(ply, p)
        rp = f"{out}/_render_{n}.png"
        render(p, rp)
        renders.append(rp)
        print(f"[{n:2d}] cover={cov*100:4.0f}%  -> {ply}")

    # hoja comparativa: fila originales / fila renders
    cw, ch = 110, 200
    sheet = Image.new("RGB", (cw * 10, ch * 2), (255, 255, 255))
    for i, n in enumerate(range(1, 11)):
        o = Image.open(f"10imagenes2d/{args.method}/{n}.png").convert("RGB").resize((cw, ch))
        r = Image.open(renders[i]).convert("RGB").resize((cw, ch))
        sheet.paste(o, (i * cw, 0))
        sheet.paste(r, (i * cw, ch))
    sheet.save(f"{out}/_comparativa.png")
    print(f"\nmedia cover = {np.mean(covs)*100:.1f}%")
    print(f"comparativa -> {out}/_comparativa.png")


if __name__ == "__main__":
    main()
