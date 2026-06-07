# 3D texturing of M3D-VTON reconstructions with 2D try-on

Dress the 3D bodies reconstructed by **M3D-VTON** with the **2D try-on** images,
producing **colored point clouds** (digital mannequins that can be rotated).

Covers **10 subjects**, each with its own 3D geometry and three try-on variants
(`baseline`, `ssim_100`, `ssim_200`) → **30 textured point clouds**.

---

## Structure

```
.
├── README.md
├── requirements.txt
│
├── texture_10.py              ← ★ generates the textured clouds (10 per method)
├── ver_10.py                  ← viewer for the 10 results of a method
├── view_3d.py                 ← viewer for a single file
├── check_formato.py           ← checks whether a .ply is raw or normalized
│
├── 10modelos3d/               ← INPUT: 10 point clouds (1.ply … 10.ply)
├── 10imagenes2d/              ← INPUT: 2D try-on (320×512)
│   ├── baseline/  (1.png … 10.png)
│   ├── ssim_100/  (1.png … 10.png)
│   └── ssim_200/  (1.png … 10.png)
│
└── resultados/                ← OUTPUT: textured clouds + comparison sheet
    ├── 10_baseline/   (recon_1_baseline_pcd.ply … + _comparativa.png)
    ├── 10_ssim_100/
    └── 10_ssim_200/
```

---

## Installation

```bash
pip install -r requirements.txt
```
Python 3.9+ (tested on Windows). Dependencies: `open3d`, `numpy`, `pillow`, `scipy`.

---

## Data correspondence (important)

- The pairing is **one-to-one per subject**: model `N.ply` goes with image `N.png`.
- The three folders (`baseline`, `ssim_100`, `ssim_200`) are **three try-on methods for the
  same 10 subjects**, not different subjects. They share the same geometry per subject.
- It is not "all-with-all": you cannot dress one subject's body with another subject's clothes.

---

## Usage

### 1. Generate / regenerate the results

```bash
python texture_10.py --method ssim_200   # creates resultados/10_ssim_200/recon_1_…_pcd.ply …
python texture_10.py --method baseline
python texture_10.py --method ssim_100
```
Each run saves the 10 `.ply` clouds and a `_comparativa.png` sheet (photo vs 3D).
The results are already included under `resultados/`.

### 2. Visualize

```bash
python ver_10.py                      # the 10 (ssim_200) in a row, in one window
python ver_10.py --metodo baseline    # another method
python ver_10.py --uno                # one by one
python ver_10.py --punto 9            # larger points (fills gaps)

python view_3d.py resultados/10_ssim_200/recon_1_ssim_200_pcd.ply   # a single file
```
Controls: left mouse = rotate · wheel = zoom · middle mouse = pan · `+`/`-` = point size ·
`Q`/`Esc` = close.

### 3. Check the format of a cloud

```bash
python check_formato.py 10modelos3d/7.ply 10imagenes2d/ssim_200/7.png
```
Reports whether the cloud is **RAW** (exact projection) or **NORMALIZED** (what we have;
forces an approximate projection).

---

## Method (summary)

1. **Inverse projection**: M3D-VTON places each pixel of the photo at a 3D point following a
   fixed rule; we invert it to find, for a given point, which pixel (and color) it maps to.
2. **Front/back by depth**: the nearer layer takes the try-on color; the back (not
   photographed) is synthesized (darkened front + hair on the nape).
3. **Bounding-box alignment**: the clouds come **re-normalized** (all in the box
   `[-0.42,0.42]×[-0.97,0.97]`), which breaks the exact projection. We anchor each cloud to
   the silhouette of its photo (head→head, feet→feet) to recover the alignment.
4. **Cleaning**: DBSCAN (largest cluster) + outlier removal.

> **Main limitation:** because the clouds are normalized, the texture is approximate
> (somewhat "blocky"). With the **raw** clouds the projection would be exact
> (`check_formato.py` tells them apart).

---

## Credits

3D reconstruction and 2D try-on: **M3D-VTON** (Zhao et al., ICCV 2021). This work adds the
3D texturing of the reconstructions from the try-on outputs.
