#!/usr/bin/env python3
"""
Cross-Modality Co-registration (SAR ↔ Optical) — NGF + affine refinement

Supports RAW inputs:
- Sentinel-1: .SAFE OR measurement/*vv*.tiff -> (optional) pyroSAR+SNAP geocode to sigma0, terrain-corrected GeoTIFF
- Sentinel-2: B08_10m.jp2 -> (optional) GDAL convert to GeoTIFF

Pipeline:
1) (Optional) S1 preproc with pyroSAR (Calibration -> Terrain Correction) to UTM.
2) (Optional) S2 JP2 -> GeoTIFF via gdal_translate.
3) Coarse alignment: reproject SAR & MSI to a *shared* UTM grid.
4) Fine alignment: NGF-based affine refinement (Huber loss) in a pyramid.
5) QC: RMSE_x, RMSE_y, radial RMSE, median error, CE90; checkerboard overlays.
6) Outputs: co-registered chip(s), transform params, QC summary.

Dependencies:
  pip install numpy rasterio shapely scikit-image scipy matplotlib tqdm spatialist pyroSAR
  (Install GDAL system libs; ensure SNAP 'gpt' is on PATH for pyroSAR)
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from shapely.geometry import shape, box
from shapely.ops import unary_union, transform as shp_transform
from skimage.transform import AffineTransform, warp, pyramid_gaussian
from skimage.feature import ORB, match_descriptors
from skimage.filters import sobel_h, sobel_v, gaussian
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# ----------------------------
# S1 preprocessing via pyroSAR (SNAP gpt backend)
# ----------------------------
def s1_preprocess_with_pyrosar(safe_or_meas: str,
                               out_dir: Path,
                               target_epsg: str,
                               target_res_m: float = 10.0,
                               pol: str = "VV",
                               dem_name: str = "Copernicus 30m Global DEM") -> Path:
    """
    Use pyroSAR to run SNAP 'geocode' (Calibration -> Terrain Correction).
    Returns the path to the produced GeoTIFF (sigma0, terrain-corrected).
    """
    try:
        from pyroSAR.snap import geocode
    except Exception as e:
        raise RuntimeError(
            "pyroSAR not importable. Ensure 'pyroSAR' & 'spatialist' are installed and GDAL Python works."
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)

    # pyroSAR 0.30.1 signature: spacing, t_srs, polarizations, refarea, scaling, demName, export_extra
    geocode(
        infile=str(safe_or_meas),
        outdir=str(out_dir),
        spacing=target_res_m,          # pixel spacing [m]
        t_srs=target_epsg,             # e.g., "EPSG:32651"
        polarizations=[pol],           # e.g., ["VV"] or ["VH"]
        refarea="sigma0",              # output reference area
        scaling="linear",              # "linear" or "dB"
        demName=dem_name,              # SNAP DEM name, e.g., "Copernicus 30m Global DEM"
        export_extra=["localIncidenceAngle"],
    )

    # Find newest GeoTIFF written by geocode()
    cands = sorted(out_dir.glob("*.tif")) + sorted(out_dir.glob("*.tiff"))
    if not cands:
        raise RuntimeError("pyroSAR geocode finished but no GeoTIFF found in output directory.")
    return max(cands, key=lambda p: p.stat().st_mtime)


def gdal_translate_jp2_to_tif(jp2_path: str, out_tif: Path) -> Path:
    """Convert Sentinel-2 JP2 (B08_10m.jp2) to GeoTIFF using gdal_translate."""
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "gdal_translate",
        "-of", "GTiff",
        "-co", "TILED=YES",
        "-co", "BIGTIFF=YES",
        "-co", "COMPRESS=DEFLATE",
        "-co", "PREDICTOR=2",
        jp2_path,
        str(out_tif)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_tif


# ----------------------------
# Utilities
# ----------------------------
def read_geojson_polygon(path: str):
    with open(path, "r") as f:
        gj = json.load(f)
    if gj["type"] == "FeatureCollection":
        geoms = [shape(feat["geometry"]) for feat in gj["features"]]
        return unary_union(geoms)
    elif gj["type"] == "Feature":
        return shape(gj["geometry"])
    else:
        return shape(gj)


def dataset_bounds_in_wgs84(ds: rio.io.DatasetReader):
    from pyproj import Transformer
    bounds = ds.bounds
    transformer = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
    xs = [bounds.left, bounds.right]
    ys = [bounds.bottom, bounds.top]
    corners = [(x, y) for x in xs for y in ys]
    lons, lats = zip(*[transformer.transform(x, y) for x, y in corners])
    return min(lons), min(lats), max(lons), max(lats)


def utm_epsg_from_lonlat(lon, lat):
    zone = int(np.floor((lon + 180) / 6) + 1)
    return f"EPSG:{32600 + zone}" if lat >= 0 else f"EPSG:{32700 + zone}"


def to_float32_minmax(img, clip_p=(1, 99)):
    finite = np.isfinite(img)
    if not np.any(finite):
        return np.zeros_like(img, dtype=np.float32)
    vmin, vmax = np.percentile(img[finite], clip_p)
    return np.clip((img - vmin) / (vmax - vmin + 1e-12), 0, 1).astype(np.float32)


def snap_bounds(bounds: Tuple[float, float, float, float], res: float) -> Tuple[float, float, float, float]:
    left, bottom, right, top = bounds
    left   = np.floor(left   / res) * res
    bottom = np.floor(bottom / res) * res
    right  = np.ceil(right   / res) * res
    top    = np.ceil(top     / res) * res
    return (left, bottom, right, top)


# ----------------------------
# IO & Reprojection (Coarse)
# ----------------------------
@dataclass
class GeoImage:
    data: np.ndarray
    transform: rio.Affine
    crs: rio.crs.CRS
    nodata: Optional[float]
    profile: dict


def load_single_band(path: str, band: int = 1) -> GeoImage:
    with rio.open(path) as ds:
        img = ds.read(band, masked=False).astype(np.float32)
        return GeoImage(img, ds.transform, ds.crs, ds.nodata, ds.profile.copy())


def reproject_to_common_grid(src: GeoImage, dst_crs: rio.crs.CRS, res: float,
                             dst_bounds: Optional[Tuple[float, float, float, float]] = None) -> GeoImage:
    if dst_bounds is None:
        raise ValueError("dst_bounds must be provided to ensure a shared grid.")
    left, bottom, right, top = dst_bounds
    dst_width = int(np.ceil((right - left) / res))
    dst_height = int(np.ceil((top - bottom) / res))
    new_transform = rio.transform.from_bounds(left, bottom, right, top, dst_width, dst_height)

    out = np.zeros((dst_height, dst_width), dtype=np.float32)
    out_nodata = 0.0
    reproject(
        source=src.data,
        destination=out,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=new_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        dst_nodata=out_nodata,
    )
    profile = src.profile.copy()
    profile.update({
        "crs": dst_crs,
        "transform": new_transform,
        "width": dst_width,
        "height": dst_height,
        "dtype": "float32",
        "count": 1,
        "nodata": out_nodata
    })
    return GeoImage(out, new_transform, dst_crs, out_nodata, profile)


def crop_to_bounds(img: GeoImage, bounds: Tuple[float, float, float, float]) -> GeoImage:
    """
    Crop using world bounds. Note: inverse affine returns (col, row).
    """
    left, bottom, right, top = bounds
    col_min, row_min = (~img.transform) * (left,  top)
    col_max, row_max = (~img.transform) * (right, bottom)
    r0, r1 = int(np.floor(row_min)), int(np.ceil(row_max))
    c0, c1 = int(np.floor(col_min)), int(np.ceil(col_max))
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(img.data.shape[0], r1), min(img.data.shape[1], c1)
    data = img.data[r0:r1, c0:c1]
    transform = img.transform * rio.Affine.translation(c0, r0)
    profile = img.profile.copy()
    profile.update({"height": data.shape[0], "width": data.shape[1], "transform": transform})
    return GeoImage(data, transform, img.crs, img.nodata, profile)


# ----------------------------
# NGF + Affine refinement
# ----------------------------
def compute_ngf(image: np.ndarray, eps: float = 1e-3, sigma: float = 1.0) -> np.ndarray:
    img = gaussian(image, sigma=sigma, preserve_range=True) if sigma > 0 else image
    gx = sobel_h(img).astype(np.float32)
    gy = sobel_v(img).astype(np.float32)
    denom = np.sqrt(gx * gx + gy * gy + eps * eps)
    return np.stack([gx / denom, gy / denom], axis=-1)


def affine_from_params(p):
    tx, ty, rot_deg, sx, sy, shear_rad = p
    return AffineTransform(scale=(sx, sy), rotation=np.deg2rad(rot_deg), shear=shear_rad, translation=(tx, ty))


def warp_image(img: np.ndarray, tform: AffineTransform, output_shape, order=1):
    return warp(img, inverse_map=tform.inverse, output_shape=output_shape, order=order, mode="edge", preserve_range=True)


def ngf_cost(p, fix_ngf: np.ndarray, mov_img: np.ndarray, output_shape, eps=1e-3, sigma=1.0, huber_delta=0.05):
    tform = affine_from_params(p)
    mov_warp = warp_image(mov_img, tform, output_shape=output_shape, order=1)
    mov_ngf = compute_ngf(mov_warp, eps=eps, sigma=sigma)
    dot = np.clip(fix_ngf[..., 0] * mov_ngf[..., 0] + fix_ngf[..., 1] * mov_ngf[..., 1], -1.0, 1.0)
    resid = 1.0 - dot
    abs_r = np.abs(resid)
    huber = np.where(abs_r <= huber_delta, 0.5 * abs_r**2, huber_delta * (abs_r - 0.5 * huber_delta))
    return float(np.mean(huber))


def refine_affine_ngf(fix_img: np.ndarray, mov_img: np.ndarray, levels: int = 3,
                      init_params=(0, 0, 0, 1, 1, 0)):
    """
    Multi-scale NGF with Huber loss.
    Params: (tx_px, ty_px, rot_deg, sx, sy, shear_rad)
    """
    fix = to_float32_minmax(fix_img)
    mov = to_float32_minmax(mov_img)

    def build_pyr(im):
        pyr = list(pyramid_gaussian(im, max_layer=levels-1, downscale=2, channel_axis=None))
        return pyr[::-1]

    fix_pyr, mov_pyr = build_pyr(fix), build_pyr(mov)

    p = np.array(init_params, dtype=np.float64)
    for lvl, (f, m) in enumerate(zip(fix_pyr, mov_pyr), 1):
        fix_ngf = compute_ngf(f, eps=1e-3, sigma=1.0)
        out_shape = f.shape
        huber_delta = 0.08 / (2 ** (levels - lvl))
        res = minimize(ngf_cost, p, args=(fix_ngf, m, out_shape, 1e-3, 1.0, huber_delta),
                       method="Powell", options={"maxiter": 80, "xtol": 1e-4, "ftol": 1e-4})
        p = res.x
    return p, affine_from_params(p)


# ----------------------------
# QC helpers
# ----------------------------
def orb_matches_xy(imgA: np.ndarray, imgB: np.ndarray, n_keypoints=2000, seed: int = 0):
    """
    Feature matching for QC (on gradient magnitudes). Returns (xy_ref, xy_mov).
    Deterministic via random_state.
    """
    def gradmag(im):
        gx = sobel_h(im); gy = sobel_v(im); return np.hypot(gx, gy)

    A = to_float32_minmax(imgA); B = to_float32_minmax(imgB)
    Ag = to_float32_minmax(gradmag(A)); Bg = to_float32_minmax(gradmag(B))

    orbA = ORB(n_keypoints=n_keypoints, fast_threshold=0.05, random_state=seed)
    orbA.detect_and_extract(Ag)
    kA, dA = orbA.keypoints, orbA.descriptors

    orbB = ORB(n_keypoints=n_keypoints, fast_threshold=0.05, random_state=seed)
    orbB.detect_and_extract(Bg)
    kB, dB = orbB.keypoints, orbB.descriptors

    if dA is None or dB is None or len(dA) == 0 or len(dB) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    matches = match_descriptors(dA, dB, cross_check=True)
    ptsA = kA[matches[:, 0]][:, ::-1]  # (row,col)->(x,y)
    ptsB = kB[matches[:, 1]][:, ::-1]
    return ptsA.astype(np.float32), ptsB.astype(np.float32)


def residual_metrics(pts_ref_xy, pts_mov_xy, tform: AffineTransform):
    """
    Errors in pixel units on the reference grid.
    CE90 ≈ 2.146 * (RMSE / sqrt(2))
    """
    if len(pts_ref_xy) == 0:
        return {"num_matches": 0, "rmse_x": None, "rmse_y": None, "rmse": None, "median_err": None, "ce90": None}
    pts_mov_warp = tform(pts_mov_xy)
    dxdy = pts_ref_xy - pts_mov_warp
    rmse_x = float(np.sqrt(np.mean(dxdy[:, 0] ** 2)))
    rmse_y = float(np.sqrt(np.mean(dxdy[:, 1] ** 2)))
    rmse = float(np.sqrt(np.mean(np.sum(dxdy ** 2, axis=1))))
    d = np.linalg.norm(dxdy, axis=1)
    sigma_R = rmse / np.sqrt(2.0 + 1e-12)
    ce90 = float(2.146 * sigma_R)
    return {"num_matches": int(len(d)), "rmse_x": rmse_x, "rmse_y": rmse_y, "rmse": rmse,
            "median_err": float(np.median(d)), "ce90": ce90}


# ----------------------------
# Visualization
# ----------------------------
def save_checkerboard(path, imgA, imgB, tile=64):
    A = to_float32_minmax(imgA); B = to_float32_minmax(imgB)
    H, W = A.shape; out = np.zeros((H, W), dtype=np.float32)
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            out[y:y+tile, x:x+tile] = A[y:y+tile, x:x+tile] if ((x//tile + y//tile) % 2 == 0) else B[y:y+tile, x:x+tile]
    plt.figure(figsize=(8, 8)); plt.imshow(out, cmap="gray", vmin=0, vmax=1); plt.axis("off"); plt.tight_layout()
    plt.savefig(path, dpi=200); plt.close()


def save_overlay(path, imgA, imgB):
    A = to_float32_minmax(imgA); B = to_float32_minmax(imgB)
    rgb = np.zeros((A.shape[0], A.shape[1], 3), dtype=np.float32); rgb[..., 0] = A; rgb[..., 1] = B
    plt.figure(figsize=(8, 8)); plt.imshow(rgb, vmin=0, vmax=1); plt.axis("off"); plt.tight_layout()
    plt.savefig(path, dpi=200); plt.close()


# ----------------------------
# Main pipeline
# ----------------------------
def main(sar_path, msi_path, out_dir,
         aoi_geojson: Optional[str] = None,
         target_res: float = 10.0,
         target_crs: Optional[str] = None,
         levels: int = 3,
         tile: int = 64,
         s1_preproc: bool = False,
         s1_pol: str = "VV",
         s1_dem: str = "Copernicus 30m Global DEM",
         s2_convert: bool = False):

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Optional S1 preprocess (pyroSAR + SNAP)
    sar_input = Path(sar_path)
    if s1_preproc:
        if target_crs is None:
            target_crs = "EPSG:32651"
        sar_geo = s1_preprocess_with_pyrosar(
            safe_or_meas=str(sar_input),
            out_dir=out_dir / "s1_preproc",
            target_epsg=target_crs,
            target_res_m=target_res,
            pol=s1_pol,
            dem_name=s1_dem
        )
        sar_path = str(sar_geo)

    # ---- Optional convert: Sentinel-2 JP2 -> GeoTIFF
    if s2_convert and msi_path.lower().endswith(".jp2"):
        msi_tif = out_dir / "s2_B08_10m.tif"
        msi_geo = gdal_translate_jp2_to_tif(msi_path, msi_tif)
        msi_path = str(msi_geo)

    # Load rasters (both should be GeoTIFFs now)
    sar = load_single_band(sar_path)
    msi = load_single_band(msi_path)

    # ---- Determine target CRS (UTM)
    if target_crs is None:
        with rio.open(sar_path) as _ds: s1_bounds = dataset_bounds_in_wgs84(_ds)
        with rio.open(msi_path) as _ds: s2_bounds = dataset_bounds_in_wgs84(_ds)
        inter = (max(s1_bounds[0], s2_bounds[0]), max(s1_bounds[1], s2_bounds[1]),
                 min(s1_bounds[2], s2_bounds[2]), min(s1_bounds[3], s2_bounds[3]))
        lon_c = 0.5 * (inter[0] + inter[2]); lat_c = 0.5 * (inter[1] + inter[3])
        target_crs = utm_epsg_from_lonlat(lon_c, lat_c)
    dst_crs = rio.crs.CRS.from_string(target_crs)

    # ---- Optional AOI polygon (assumed WGS84); transform to dst_crs if provided
    aoi_poly = None
    if aoi_geojson:
        from pyproj import Transformer
        aoi_ll = read_geojson_polygon(aoi_geojson)
        tr = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True).transform
        aoi_poly = shp_transform(tr, aoi_ll)

    # ---- Compute shared, snapped grid bounds (in dst_crs) BEFORE raster reprojection
    from pyproj import Transformer
    # SAR bounds -> dst_crs
    with rio.open(sar_path) as ds:
        sleft, sbot, sright, stop = rio.transform.array_bounds(ds.height, ds.width, ds.transform)
        tr_sar = Transformer.from_crs(ds.crs, dst_crs, always_xy=True).transform
        x1, y1 = tr_sar(sleft, sbot); x2, y2 = tr_sar(sright, stop)
        sar_bounds_dst = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    # MSI bounds -> dst_crs
    with rio.open(msi_path) as ds:
        mleft, mbot, mright, mtop = rio.transform.array_bounds(ds.height, ds.width, ds.transform)
        tr_msi = Transformer.from_crs(ds.crs, dst_crs, always_xy=True).transform
        u1, v1 = tr_msi(mleft, mbot); u2, v2 = tr_msi(mright, mtop)
        msi_bounds_dst = (min(u1, u2), min(v1, v2), max(u1, u2), max(v1, v2))

    # Intersection (and apply AOI if given), then snap to pixel grid
    inter_bounds = (
        max(sar_bounds_dst[0], msi_bounds_dst[0]),
        max(sar_bounds_dst[1], msi_bounds_dst[1]),
        min(sar_bounds_dst[2], msi_bounds_dst[2]),
        min(sar_bounds_dst[3], msi_bounds_dst[3]),
    )
    if inter_bounds[2] <= inter_bounds[0] or inter_bounds[3] <= inter_bounds[1]:
        raise RuntimeError("No spatial overlap between SAR and MSI in target CRS.")

    if aoi_poly is not None:
        inter_poly = box(*inter_bounds).intersection(aoi_poly)
        if inter_poly.is_empty:
            raise RuntimeError("AOI does not intersect data.")
        inter_bounds = inter_poly.bounds

    grid_bounds = snap_bounds(inter_bounds, target_res)
    if (grid_bounds[2] - grid_bounds[0]) <= 0 or (grid_bounds[3] - grid_bounds[1]) <= 0:
        raise RuntimeError("Snapped grid bounds collapsed (check AOI size vs. resolution).")

    # ---- Coarse alignment: reproject both to the SAME grid
    sar_chip = reproject_to_common_grid(sar, dst_crs=dst_crs, res=target_res, dst_bounds=grid_bounds)
    msi_chip = reproject_to_common_grid(msi, dst_crs=dst_crs, res=target_res, dst_bounds=grid_bounds)

    # ---- Save coarse overlays
    save_overlay(out_dir / "coarse_overlay.png", sar_chip.data, msi_chip.data)
    save_checkerboard(out_dir / "coarse_checker.png", sar_chip.data, msi_chip.data, tile=tile)

    # ---- Fine: NGF affine refinement (warp MSI -> SAR grid)
    init = (0, 0, 0, 1, 1, 0)  # tx, ty, rot_deg, sx, sy, shear_rad
    params, tform = refine_affine_ngf(fix_img=sar_chip.data, mov_img=msi_chip.data, levels=levels, init_params=init)
    msi_refined = warp_image(msi_chip.data, tform, output_shape=sar_chip.data.shape, order=1)

    # ---- QC: residuals (simple ORB-on-gradient surrogate)
    refA, movA = orb_matches_xy(sar_chip.data, msi_chip.data, seed=0)
    qc_before = residual_metrics(refA, movA, AffineTransform())
    refB, movB = orb_matches_xy(sar_chip.data, msi_refined, seed=0)
    qc_after = residual_metrics(refB, movB, AffineTransform())

    # ---- Save fine overlays
    save_overlay(out_dir / "fine_overlay.png", sar_chip.data, msi_refined)
    save_checkerboard(out_dir / "fine_checker.png", sar_chip.data, msi_refined, tile=tile)

    # ---- Export GeoTIFFs (float32 with float predictor)
    out_profile = sar_chip.profile.copy()
    out_profile.update({"compress": "deflate", "predictor": 3, "tiled": True})
    with rio.open(out_dir / "msi_coreg_to_sar.tif", "w", **out_profile) as dst:
        dst.write(msi_refined.astype(np.float32), 1)
    with rio.open(out_dir / "sar_chip.tif", "w", **out_profile) as dst:
        dst.write(sar_chip.data.astype(np.float32), 1)

    # ---- Params & QC JSON
    tdict = {
        "tx_px": float(params[0]),
        "ty_px": float(params[1]),
        "rot_deg": float(params[2]),
        "sx": float(params[3]),
        "sy": float(params[4]),
        "shear_rad": float(params[5]),
        "shear_deg": float(np.degrees(params[5])),
    }
    with open(out_dir / "affine_params.json", "w") as f:
        json.dump(tdict, f, indent=2)

    thresholds = {"median_err_px_max": 0.5, "ce90_px_max": 1.0}
    accepted = (
        qc_after["median_err"] is not None
        and qc_after["median_err"] <= thresholds["median_err_px_max"]
        and qc_after["ce90"] is not None
        and qc_after["ce90"] <= thresholds["ce90_px_max"]
    )
    with open(out_dir / "qc_summary.json", "w") as f:
        json.dump({"before": qc_before, "after": qc_after, "thresholds": thresholds, "accepted": accepted}, f, indent=2)

    print("Affine params:", tdict)
    print("QC before:", qc_before)
    print("QC after :", qc_after)
    print("Accepted :", accepted)
    print("Outputs in:", out_dir)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sar", required=True, help="S1 input: .SAFE folder OR measurement/*.tiff OR TC GeoTIFF")
    ap.add_argument("--msi", required=True, help="S2 input: B08_10m.jp2 OR GeoTIFF")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--aoi", default=None, help="Optional AOI polygon (GeoJSON, WGS84)")
    ap.add_argument("--res", type=float, default=10.0, help="Target pixel size (meters)")
    ap.add_argument("--utm", default=None, help="Target CRS (e.g., EPSG:32651). Auto if omitted.")
    ap.add_argument("--levels", type=int, default=3, help="Pyramid levels for NGF refinement")
    ap.add_argument("--tile", type=int, default=64, help="Checkerboard tile size (pixels)")
    ap.add_argument("--s1_preproc", action="store_true",
                    help="Use pyroSAR+SNAP to geocode S1 input to sigma0 terrain-corrected GeoTIFF")
    ap.add_argument("--s1_pol", default="VV", choices=["VV", "VH"], help="S1 polarization for preprocessing")
    ap.add_argument("--s1_dem", default="Copernicus 30m Global DEM",
                    help="DEM name for SNAP (e.g., 'Copernicus 30m Global DEM', 'SRTM 3Sec')")
    ap.add_argument("--s2_convert", action="store_true",
                    help="Convert S2 JP2 to GeoTIFF via GDAL before processing")
    args = ap.parse_args()

    main(
        sar_path=args.sar,
        msi_path=args.msi,
        out_dir=args.out,
        aoi_geojson=args.aoi,
        target_res=args.res,
        target_crs=args.utm,
        levels=args.levels,
        tile=args.tile,
        s1_preproc=args.s1_preproc,
        s1_pol=args.s1_pol,
        s1_dem=args.s1_dem,
        s2_convert=args.s2_convert
    )
