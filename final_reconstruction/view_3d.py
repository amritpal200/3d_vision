"""
Visor 3D interactivo para los modelos texturizados.

Uso:
    python view_3d.py                      # lista los modelos y te deja elegir
    python view_3d.py recon_0_textured.ply # abre ese fichero directamente
    python view_3d.py --list               # solo lista los disponibles

Controles en la ventana de open3d:
    raton izq  -> rotar
    raton der / scroll -> zoom
    raton medio -> desplazar
    R -> resetea la vista
    +/- -> tamano de los puntos (en nubes)
    Q o Esc -> cerrar

Requisitos: pip install open3d
"""

import argparse
import glob
import os
import sys

import open3d as o3d


def find_models():
    """Devuelve los modelos 3D del proyecto, ordenados (buenos primero)."""
    files = []
    for ext in ("ply", "obj", "glb"):
        files += glob.glob(os.path.join("resultados/buenos_pointcloud", f"*.{ext}"))
        files += glob.glob(os.path.join("originales", f"*.{ext}"))
        # mesh_obsoletos ahora tiene subcarpetas (normal/clean/tex) -> recursivo
        files += glob.glob(os.path.join("resultados/mesh_obsoletos", "**", f"*.{ext}"),
                           recursive=True)
    # quitar duplicados conservando orden; los _pcd (buenos) primero
    seen, ordered = set(), []
    for f in files:
        if f not in seen:
            seen.add(f); ordered.append(f)
    ordered.sort(key=lambda f: (0 if "buenos_pointcloud" in f else 1, f))
    return ordered


def load_geometry(path):
    """Carga como malla si tiene caras; si no, como nube de puntos."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".obj", ".glb", ".gltf"):
        mesh = o3d.io.read_triangle_mesh(path, True)  # True = con texturas
        mesh.compute_vertex_normals()
        return mesh
    # .ply puede ser malla o nube
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
        if not mesh.has_vertex_colors():
            mesh.paint_uniform_color([0.8, 0.8, 0.8])
        return mesh
    pcd = o3d.io.read_point_cloud(path)
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.5, 0.5, 0.7])
    return pcd


def view(path, point_size=5.0):
    if not os.path.isfile(path):
        print(f"[ERROR] no existe: {path}")
        return
    geo = load_geometry(path)
    # abrir mostrando el FRENTE (giro 180 sobre Y); el lado +Z es la espalda sintetica
    geo.rotate(geo.get_rotation_matrix_from_axis_angle([0, 3.14159, 0]),
               center=geo.get_center())
    n = (len(geo.vertices) if hasattr(geo, "vertices") and len(geo.vertices)
         else len(geo.points))
    print(f"[abriendo] {path}  ({n} vertices/puntos)")
    print("  raton: rotar/zoom/pan | +/-: tamano punto | R: reset | Q/Esc: cerrar")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=os.path.basename(path), width=720, height=900)
    vis.get_render_option().point_size = point_size
    vis.add_geometry(geo)
    vis.run()
    vis.destroy_window()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", help="modelo a abrir (.ply/.obj/.glb)")
    ap.add_argument("--list", action="store_true", help="solo listar modelos")
    args = ap.parse_args()

    if args.file:
        view(args.file)
        return

    models = find_models()
    if not models:
        print("No encontre modelos (.ply/.obj/.glb) en esta carpeta.")
        return

    print("Modelos disponibles:")
    for i, f in enumerate(models):
        size = os.path.getsize(f) / 1e6
        print(f"  [{i}] {f}  ({size:.1f} MB)")

    if args.list:
        return

    try:
        sel = input("\nNumero a abrir (Enter = 0): ").strip()
    except EOFError:
        sel = ""
    idx = int(sel) if sel.isdigit() else 0
    if 0 <= idx < len(models):
        view(models[idx])
    else:
        print("Indice fuera de rango.")


if __name__ == "__main__":
    main()
