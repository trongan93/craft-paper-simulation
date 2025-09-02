# (A) RAW inputs → auto pre-process + co-registration
python coregister_sar_optical.py \
  --sar S1A_IW_GRDH_1SDV_20250710T215222_..._07751C_388A.SAFE \
  --msi GRANULE/.../IMG_DATA/R10m/T51RUH_20250713T022611_B08_10m.jp2 \
  --out ./out_coreg \
  --utm EPSG:32651 --res 10 --levels 3 \
  --s1_preproc --s2_convert

# (B) If you already have terrain-corrected S1 and a GeoTIFF for S2
python coregister_sar_optical.py \
  --sar s1_sigma0_vv_tc_10m.tif \
  --msi S2_B08_10m.tif \
  --out ./out_coreg --utm EPSG:32651 --res 10 --levels 3

# core libs
pip install numpy rasterio shapely scikit-image scipy matplotlib tqdm
# pyrosar + SNAP bridge
pip install pyroSAR snapista
# make sure GDAL is present so JP2 → GeoTIFF works (system package or wheel)
# Ubuntu example:
# sudo apt-get install gdal-bin libgdal-dev


[//]: # (updated by An 19:33 8/30)
# --- System prerequisites ---
sudo apt-get update
sudo apt-get install -y \
  gdal-bin libgdal-dev \
  proj-bin libproj-dev \
  libgeos-dev build-essential

# (optional but recommended: Java for SNAP tools you already installed)
# sudo apt-get install -y openjdk-11-jre-headless

# --- Activate your existing venv ---
source ~/Projects/craft-paper-simulation/.venv_sar_optical/bin/activate

# --- Start fresh to avoid wheel mismatches ---
pip uninstall -y gdal rasterio shapely pyproj spatialist pyroSAR
pip cache purge

# --- Point pip at system GDAL headers/libs ---
export GDAL_CONFIG=$(command -v gdal-config)
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal

# --- Reinstall in the right order ---
pip install --upgrade pip setuptools wheel
pip install "gdal==$(gdal-config --version).*"      # Must match system GDAL exactly
pip install numpy
pip install --no-binary rasterio rasterio           # Build against system GDAL/PROJ
pip install --no-binary shapely shapely             # Build against system GEOS
pip install --no-binary pyproj pyproj               # Build against system PROJ
pip install scikit-image scipy matplotlib tqdm
pip install spatialist pyroSAR

# --- Sanity checks (should all print versions, no exceptions) ---
python - <<'PY'
from osgeo import gdal, gdal_array
print("GDAL Python OK:", gdal.VersionInfo())
import rasterio, shapely, pyproj, spatialist, pyroSAR
print("rasterio:", rasterio.__version__)
print("shapely :", shapely.__version__)
print("pyproj  :", pyproj.__version__)
print("spatialist:", spatialist.__version__)
print("pyroSAR   :", pyroSAR.__version__)
PY


# Ensure SNAP gpt is on PATH (you already did this)
export PATH="$HOME/esa-snap/bin:$PATH"

# Run
python coregister_sar_optical.py \
  --sar data/S1A_IW_GRDH_1SDV_20250710T215222_20250710T215251_060027_07751C_388A.SAFE \
  --msi TestData/T51RUH_20250713T022611_B08_10m.jp2 \
  --out ./out_coreg \
  --utm EPSG:32651 --res 10 --levels 3 \
  --s1_preproc --s2_convert

Current run on 8/30 23:39
python coregister_sar_optical.py   --sar data/S1A_IW_GRDH_1SDV_20250710T215222_20250710T215251_060027_07751C_388A.SAFE   --msi TestData/T51RUH_20250713T022611_B08_10m.jp2   --out ./out_coreg   --utm EPSG:32651 --res 10 --levels 3   --s1_preproc --s2_convert


# pseudocode (drop-in) for coarse→fine
// All buffers are uint8 unless noted; Q15 for gradients/ECC.
preprocess_sar_log_median(sar);
normalize_optical(opt);

build_pyramids(sar, opt);

for (lvl=top; lvl>=mid; --lvl) {
  census_sar = census5x5(sar[lvl]);
  census_opt = census5x5(opt[lvl]);
  if (lvl==top) {
    T0 = phase_correlation_shift(census_sar, census_opt); // or log-polar first
    apply_transform(opt[lvl], T0);
  }
}

// Feature stage at lvl = mid or 0
census_sar0 = census5x5(sar[0]);
census_opt0 = census5x5(opt[0]);

K1 = FAST(census_sar0);  K2 = FAST(census_opt0);
D1 = ORB(census_sar0, K1); D2 = ORB(census_opt0, K2);

M = knn_hamming(D1, D2, k=2);
M = ratio_and_mutual(M, 0.75);

model, inliers = RANSAC_similarity(M, thresh=3);
if (residuals_wavy) model = piecewise_affine(inliers, grid=48);

#ifdef SUBPIX
refine_matches_ECC_gradient(sar[0], opt[0], inliers);
reestimate_model(inliers);
#endif

return model;
