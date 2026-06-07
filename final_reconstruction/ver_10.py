"""
Visualizar los 10 modelos texturizados.

Uso:
    python ver_10.py                 # los 10 en fila en una sola ventana
    python ver_10.py --uno           # uno por uno (cierra cada ventana para pasar al siguiente)
    python ver_10.py --metodo baseline   # otra carpeta (resultados/10_baseline)

Controles: raton izq = girar | rueda = zoom | raton medio = mover | +/- = tamano punto | Q/Esc = cerrar
"""
import argparse, glob, os
import numpy as np
import open3d as o3d


def cara_frente(p):
    """Gira 180 sobre Y para que el visor se abra mostrando el FRENTE (try-on),
    no la espalda sintetica (que es el lado +Z que ve la camara por defecto)."""
    p.rotate(p.get_rotation_matrix_from_axis_angle(np.array([0, np.pi, 0])),
             center=p.get_center())
    return p


def carpeta(metodo):
    return f"resultados/10_{metodo}"


def ficheros(metodo):
    d = carpeta(metodo)
    fs = glob.glob(os.path.join(d, "*_pcd.ply"))
    # ordena por el numero del modelo: recon_<N>_...
    def num(f):
        b = os.path.basename(f)
        try:
            return int(b.split("_")[1])
        except Exception:
            return 999
    return sorted(fs, key=num)


def ver_todos(metodo, point_size=6.0):
    fs = ficheros(metodo)
    if not fs:
        print(f"No hay ficheros en {carpeta(metodo)}")
        return
    geos = []
    for i, f in enumerate(fs):
        p = cara_frente(o3d.io.read_point_cloud(f))
        # desplaza cada uno a la derecha para verlos en fila (ancho ~0.84 -> separa 1.0)
        p.translate((i * 1.0, 0, 0))
        geos.append(p)
        print(f"  [{i+1}] {os.path.basename(f)}  ({len(p.points)} pts)")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"10 modelos - {metodo}", width=1400, height=700)
    vis.get_render_option().point_size = point_size
    vis.get_render_option().background_color = np.array([1, 1, 1])
    for g in geos:
        vis.add_geometry(g)
    vis.run()
    vis.destroy_window()


def ver_uno_a_uno(metodo, point_size=6.0):
    for f in ficheros(metodo):
        p = cara_frente(o3d.io.read_point_cloud(f))
        print(f"[abriendo] {os.path.basename(f)}  ({len(p.points)} pts) -- cierra la ventana para el siguiente")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=os.path.basename(f), width=720, height=900)
        vis.get_render_option().point_size = point_size
        vis.get_render_option().background_color = np.array([1, 1, 1])
        vis.add_geometry(p)
        vis.run()
        vis.destroy_window()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metodo", default="ssim_200")
    ap.add_argument("--uno", action="store_true", help="uno por uno en vez de todos juntos")
    ap.add_argument("--punto", type=float, default=6.0, help="tamano del punto")
    args = ap.parse_args()
    if args.uno:
        ver_uno_a_uno(args.metodo, args.punto)
    else:
        ver_todos(args.metodo, args.punto)


if __name__ == "__main__":
    main()
