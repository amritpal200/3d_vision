"""
Comprueba si un .ply esta en formato CRUDO de M3D-VTON (el que necesitamos) o NORMALIZADO.

Uso:
    python check_formato.py ruta/al/modelo.ply [ruta/a/su_imagen.png]

CRUDO  (bueno): caja XY asimetrica; al proyectar, ~100% de puntos caen sobre la persona.
NORMALIZADO (malo): caja XY simetrica ±0.42/±0.97; al proyectar, ~30% sobre la persona.
"""
import sys
import numpy as np
import open3d as o3d
from PIL import Image

IMG_W, IMG_H = 320, 512


def main():
    if len(sys.argv) < 2:
        print("uso: python check_formato.py modelo.ply [imagen.png]")
        return
    f = sys.argv[1]
    V = np.asarray(o3d.io.read_point_cloud(f).points)
    X, Y = V[:, 0], V[:, 1]
    xc = (X.min() + X.max()) / 2
    yc = (Y.min() + Y.max()) / 2
    print(f"puntos: {len(V)}")
    print(f"caja X[{X.min():.4f}, {X.max():.4f}]  centro={xc:+.4f}")
    print(f"caja Y[{Y.min():.4f}, {Y.max():.4f}]  centro={yc:+.4f}")
    simetrico = abs(xc) < 0.01 and abs(X.max() - 0.42) < 0.01
    print(f"-> caja {'SIMETRICA (normalizado, MALO)' if simetrico else 'asimetrica (posible CRUDO, bueno)'}")

    if len(sys.argv) >= 3:
        img = np.asarray(Image.open(sys.argv[2]).convert("RGB")).astype(int)
        mask = img.sum(2) > 30
        w = (X + 1.0) * 256 - 95
        h = (IMG_H - 1) - (Y + 1.0) * 256
        wi = np.clip(np.round(w).astype(int), 0, IMG_W - 1)
        hi = np.clip(np.round(h).astype(int), 0, IMG_H - 1)
        sobre = mask[hi, wi].mean()
        print(f"proyeccion exacta -> {sobre*100:.0f}% de puntos sobre la persona "
              f"({'CRUDO/correcto' if sobre > 0.85 else 'NORMALIZADO/incorrecto'})")


if __name__ == "__main__":
    main()
