import heapq
import os

import rasterio
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

from rasterio.transform import rowcol, xy
from rasterio.warp import reproject, Resampling
from shapely.geometry import LineString, Point


# INPUT DATA PATHS
dem_path = "data/dem.tif"
points_path = "data/start_points.gpkg"
sink_mask_path = "data/sink_mask.tif"
tpi_path = "data/tpi.tif"
slope_path = "data/slope.tif"
valley_path = "data/valley_mask_cervinia.tif"
roughness_path = "data/roughness.tif"
aspect_path = "data/aspect_map.tif"
obstacle_path = "data/obstacle_mask.tif"

# OUTPUT DATA PATHS
output_ski_runs_path = "data/generated_ski_runs.gpkg"
output_corridor_raster_path = "data/generated_corridors.tif"
output_cost_raster_path = "data/cost_raster.tif"

# PARAMETERS
NUM_SKI_RUNS = 2 # number of alternative runs per start point
CORRIDOR_RELAXATION = 0.5 # combined cost threshold for corridor (0.5 = 50% above optimal)
DIVERSITY_PENALTY = 100.0 # penalty cost around already selected ski runs
DIVERSITY_BUFFER_CELLS = 10 # ski run buffer zone in raster cells around already found ski runs
ALLOW_EQUAL_ELEVATION = True # true = allow flat moves, false = strictly downhill

# WEIGHT PARAMETERS FOR COST FACTORS
K_SLOPE_LOW = 0.3 # used to calculate slope cost for slopes below ideal range
K_SLOPE_HIGH = 0.5 # used to calculate slope cost for slopes above ideal range
W_SLOPE = 0.60
W_ROUGHNESS = 0.20
W_ASPECT = 0.20

# WEIGHT PARAMETERS FOR DIFFICULTY SCORE
K_MEAN_SLOPE = 0.3
K_MAX_SLOPE = 0.5
K_MEAN_ROUGHNESS = 0.2

SHOW_PLOT = False # for debugging

# DIFFICULTY GRADING THRESHOLDS
BLUE_PERCENTILE = 40.0 # blue runs will be those with difficulty score below this percentile (relative to all runs in this area)
RED_PERCENTILE  = 80.0 # red runs will be those with difficulty score between BLUE_PERCENTILE and this RED_PERCENTILE
# black runs will be the remaining with difficulty score above RED_PERCENTILE

os.makedirs("data", exist_ok=True)

# convert x and y coordinates to row and column of the corresponding cell
def point_to_rc(x, y, transform):
    r, c = rowcol(transform, x, y)
    return int(r), int(c)

# convert row and column to x and y coordinates of the cell center
def rc_to_xy(transform, row, col):
    x, y = xy(transform, row, col, offset="center")
    return x, y

# check if row and column are within the bounds of the raster shape
def in_bounds(row, col, shape):
    rows, cols = shape[0], shape[1]
    row_ok = row >= 0 and row < rows
    col_ok = col >= 0 and col < cols
    return row_ok and col_ok

# get neighboring cells
def get_8_neighbors(row, col):
    neighbors = [
        (-1, -1, np.sqrt(2.0)), # upper-left
        (-1,  0, 1.0), # up
        (-1,  1, np.sqrt(2.0)), # upper-right
        ( 0, -1, 1.0), # left
        ( 0,  1, 1.0), # right
        ( 1, -1, np.sqrt(2.0)), # lower-left
        ( 1,  0, 1.0), # down
        ( 1,  1, np.sqrt(2.0)), # lower-right
    ]

    for dr, dc, step in neighbors:
        yield row + dr, col + dc, step

# reconstruct path from Dijkstra prev dictionary
def reconstruct_path(prev, end_rc):
    path = []
    cur = end_rc
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path

# convert a path of row, column indices to a LineString in real coordinates
def path_to_linestring(path, transform):
    coords = [rc_to_xy(transform, r, c) for r, c in path]
    return LineString(coords)

#  penalty buffer zone of a boolean mask by a given radius in cells
def penalty_mask(mask, radius):
    if radius <= 0:
        return mask.copy()

    out = mask.copy()
    rows, cols = np.where(mask)
    for r, c in zip(rows, cols):
        r0 = max(0, r - radius)
        r1 = min(mask.shape[0], r + radius + 1)
        c0 = max(0, c - radius)
        c1 = min(mask.shape[1], c + radius + 1)
        out[r0:r1, c0:c1] = True
    return out

# create a mask of the path cells for easy masking operations
def path_to_mask(path, shape):
    m = np.zeros(shape, dtype=bool)
    for r, c in path:
        if in_bounds(r, c, shape):
            m[r, c] = True
    return m


# MOVEMENT RULES
def can_move_downhill(z_from, z_to):
    if np.isnan(z_from) or np.isnan(z_to):
        return False
    if ALLOW_EQUAL_ELEVATION:
        return z_to <= z_from
    return z_to < z_from


def can_move_reverse(z_current, z_neighbor):
    if np.isnan(z_current) or np.isnan(z_neighbor):
        return False
    if ALLOW_EQUAL_ELEVATION:
        return z_current <= z_neighbor
    return z_current < z_neighbor


# DIJKSTRA
# Forward cost from one start point to all reachable cells, downhill only
def dijkstra_from_start(start_rc, cost_arr, dem_arr):
    shape = cost_arr.shape
    sr, sc = start_rc

    dist = np.full(shape, np.inf, dtype=np.float64)
    prev = {}
    visited = np.zeros(shape, dtype=bool)

    if not in_bounds(sr, sc, shape):
        return dist, prev
    if not np.isfinite(cost_arr[sr, sc]):
        return dist, prev

    dist[sr, sc] = 0.0
    prev[(sr, sc)] = None
    pq = [(0.0, (sr, sc))]

    while pq:
        cur_dist, (r, c) = heapq.heappop(pq)

        if visited[r, c]:
            continue
        visited[r, c] = True

        z_cur = dem_arr[r, c]

        for nr, nc, step_len in get_8_neighbors(r, c):
            if not in_bounds(nr, nc, shape):
                continue
            if visited[nr, nc]:
                continue
            if not np.isfinite(cost_arr[nr, nc]):
                continue

            z_next = dem_arr[nr, nc]
            if not can_move_downhill(z_cur, z_next):
                continue

            step_cost = cost_arr[nr, nc] * step_len
            new_dist = cur_dist + step_cost

            if new_dist < dist[nr, nc]:
                dist[nr, nc] = new_dist
                prev[(nr, nc)] = (r, c)
                heapq.heappush(pq, (new_dist, (nr, nc)))

    return dist, prev

# Reverse cost from all valley targets to all reachable cells, uphill only
def dijkstra_from_targets(target_mask, cost_arr, dem_arr):
    shape = cost_arr.shape
    dist = np.full(shape, np.inf, dtype=np.float64)
    visited = np.zeros(shape, dtype=bool)

    pq = []
    target_rows, target_cols = np.where(target_mask)
    for r, c in zip(target_rows, target_cols):
        if np.isfinite(cost_arr[r, c]):
            dist[r, c] = 0.0
            heapq.heappush(pq, (0.0, (r, c)))

    while pq:
        cur_dist, (r, c) = heapq.heappop(pq)

        if visited[r, c]:
            continue
        visited[r, c] = True

        z_cur = dem_arr[r, c]

        for nr, nc, step_len in get_8_neighbors(r, c):
            if not in_bounds(nr, nc, shape):
                continue
            if visited[nr, nc]:
                continue
            if not np.isfinite(cost_arr[nr, nc]):
                continue

            z_n = dem_arr[nr, nc]
            if not can_move_reverse(z_cur, z_n):
                continue

            step_cost = cost_arr[nr, nc] * step_len
            new_dist = cur_dist + step_cost

            if new_dist < dist[nr, nc]:
                dist[nr, nc] = new_dist
                heapq.heappush(pq, (new_dist, (nr, nc)))

    return dist

# RUN EXTRACTION INSIDE CORRIDOR TO FIRST TARGET CELL
def shortest_path_in_corridor(start_rc, corridor_mask, base_cost_arr, dem_arr, target_mask, extra_penalty=None):
    # Dijkstra to find lowest cost path from start to any target cell, but only through the corridor and with downhill movement rules, and with optional extra penalty added to the base cost for diversity between runs
    shape = base_cost_arr.shape
    sr, sc = start_rc

    if extra_penalty is None:
        extra_penalty = np.zeros(shape, dtype=np.float64)

    dist = np.full(shape, np.inf, dtype=np.float64)
    prev = {}
    visited = np.zeros(shape, dtype=bool)

    if not in_bounds(sr, sc, shape):
        return None, np.inf
    if not corridor_mask[sr, sc]:
        return None, np.inf
    if not np.isfinite(base_cost_arr[sr, sc]):
        return None, np.inf

    dist[sr, sc] = 0.0
    prev[(sr, sc)] = None
    pq = [(0.0, (sr, sc))]

    while pq:
        cur_dist, (r, c) = heapq.heappop(pq)

        if visited[r, c]:
            continue
        visited[r, c] = True

        if target_mask[r, c]:
            return reconstruct_path(prev, (r, c)), cur_dist

        z_cur = dem_arr[r, c]

        for nr, nc, step_len in get_8_neighbors(r, c):
            if not in_bounds(nr, nc, shape):
                continue
            if visited[nr, nc]:
                continue
            if not corridor_mask[nr, nc]: # must be inside corridor
                continue
            if not np.isfinite(base_cost_arr[nr, nc]): # must be valid cost cell
                continue

            z_next = dem_arr[nr, nc]
            if not can_move_downhill(z_cur, z_next): # must be downhill move
                continue

            step_cost = (base_cost_arr[nr, nc] + extra_penalty[nr, nc]) * step_len # total step cost including any extra penalty
            new_dist = cur_dist + step_cost

            if new_dist < dist[nr, nc]:
                dist[nr, nc] = new_dist
                prev[(nr, nc)] = (r, c)
                heapq.heappush(pq, (new_dist, (nr, nc)))

    return None, np.inf



# LOAD DATA
dem = rasterio.open(dem_path)
sink_mask = rasterio.open(sink_mask_path)
tpi = rasterio.open(tpi_path)
slope = rasterio.open(slope_path)
valley_mask_src = rasterio.open(valley_path)
roughness = rasterio.open(roughness_path)
aspect = rasterio.open(aspect_path)

start_points = gpd.read_file(points_path)

# start_points = start_points.iloc[[0]] # for testing with a single start point

if start_points.crs != dem.crs:
    start_points = start_points.to_crs(dem.crs) # reproject start points to DEM CRS if needed


dem_arr = dem.read(1).astype(float)
sink_mask_arr = sink_mask.read(1).astype(float)
tpi_arr = tpi.read(1).astype(float)
slope_arr = slope.read(1).astype(float)
roughness_arr = roughness.read(1).astype(float)
aspect_arr = aspect.read(1).astype(float)

obstacle_arr = None
if os.path.exists(obstacle_path):
    obstacles_src = rasterio.open(obstacle_path)
    obstacle_arr = np.zeros(dem_arr.shape, dtype=np.float32)
    reproject(
        source=rasterio.band(obstacles_src, 1),
        destination=obstacle_arr,
        src_transform=obstacles_src.transform,
        src_crs=obstacles_src.crs,
        dst_transform=dem.transform,
        dst_crs=dem.crs,
        dst_width=dem.width,
        dst_height=dem.height,
        resampling=Resampling.nearest
    )

# assign invalid data (nodata values) to np.nan for easier masking later
dem_arr[dem_arr == dem.nodata] = np.nan
sink_mask_arr[sink_mask_arr == sink_mask.nodata] = np.nan
tpi_arr[tpi_arr == tpi.nodata] = np.nan
slope_arr[slope_arr == slope.nodata] = np.nan
roughness_arr[roughness_arr == roughness.nodata] = np.nan
aspect_arr[aspect_arr == aspect.nodata] = np.nan

# reproject valley mask to DEM resolution and CRS if needed
valley_mask_arr = np.zeros(dem_arr.shape, dtype=np.uint8)
reproject(
    source=rasterio.band(valley_mask_src, 1),
    destination=valley_mask_arr,
    src_transform=valley_mask_src.transform,
    src_crs=valley_mask_src.crs,
    dst_transform=dem.transform,
    dst_crs=dem.crs,
    dst_width=dem.width,
    dst_height=dem.height,
    resampling=Resampling.nearest
)

# identify unique valley zones from the reprojected valley mask
zone_values = np.unique(valley_mask_arr[valley_mask_arr > 0])
print(f"Found {len(zone_values)} valley zones: {zone_values}")

print("Raster / CRS check")
for name, crs in {
    "dem": dem.crs,
    "sink_mask": sink_mask.crs,
    "tpi": tpi.crs,
    "slope": slope.crs,
    "start_points": start_points.crs,
    "valley_mask": valley_mask_src.crs,
    "roughness": roughness.crs,
    "aspect": aspect.crs,
    "obstacles": obstacles_src.crs if obstacle_arr is not None else None
}.items():
    print(f"{name}: {crs}")

print("DEM shape:", dem_arr.shape)
print("Valley shape:", valley_mask_arr.shape)


# COST SURFACE CREATION
valid_mask = (
    ~np.isnan(slope_arr) &
    ~np.isnan(roughness_arr) &
    ~np.isnan(sink_mask_arr) &
    ~np.isnan(dem_arr) &
    ~np.isnan(aspect_arr)
)

# slope cost
slope_raw = np.full_like(slope_arr, np.nan, dtype=float)

ideal = (slope_arr >= 15) & (slope_arr <= 30) & valid_mask
low   = (slope_arr < 15)  & valid_mask
high  = (slope_arr > 30)  & valid_mask

slope_raw[ideal] = 0.0
slope_raw[low] = (15.0 - slope_arr[low]) * K_SLOPE_LOW # k_low
slope_raw[high] = (slope_arr[high] - 30.0) * K_SLOPE_HIGH # k_high # maybe exponent?

p95_slope = np.nanpercentile(slope_raw[valid_mask], 95)
if p95_slope == 0 or np.isnan(p95_slope):
    p95_slope = 1.0

slope_cost = np.full_like(slope_arr, np.nan, dtype=float)
slope_norm = slope_raw / p95_slope


slope_cost[valid_mask] = np.clip(slope_norm[valid_mask], 0.0, 1.0) # normalize slope to [0;1] based on 95th percentile

# roughness cost
rough_cost = np.full_like(roughness_arr, np.nan, dtype=float)

p95 = np.nanpercentile(roughness_arr[valid_mask], 95)
if p95 == 0 or np.isnan(p95):
    p95 = 1.0

rough_norm = roughness_arr / p95

rough_cost[valid_mask] = np.clip(rough_norm[valid_mask], 0.0, 1.0) # normalize roughness to [0;1] based on 95th percentile

# aspect cost
aspect_cost = np.full_like(aspect_arr, np.nan, dtype=float)

aspect_rad = np.deg2rad(aspect_arr)
southness = np.cos(aspect_rad - np.pi)
aspect_cost[valid_mask] = (southness[valid_mask] + 1.0) / 2.0 # normalize to [0;1], where 1 = south-facing, 0 = north-facing

# combine cost factors with weights
cost = np.full_like(slope_arr, np.inf, dtype=float)
cost[valid_mask] = (
    W_SLOPE * slope_cost[valid_mask] +
    W_ROUGHNESS * rough_cost[valid_mask] +
    W_ASPECT * aspect_cost[valid_mask]
)

# hard constraints
# sinks
cost[(sink_mask_arr == 1) & valid_mask] = np.inf
# invalid data
cost[~valid_mask] = np.inf

# obstacles (if they have been fetched in QGIS model at all)
if obstacle_arr is not None:
    cost[obstacle_arr == 1] = np.inf

# create valley zone masks for each unique zone value
valley_zones = {} # valley_zones = { 1: mask zone 1, 2: mask zone 2 }}
for val in zone_values:
    valley_zones[val] = (valley_mask_arr == val) & np.isfinite(cost)

# all targets combined for reverse Dijkstra
valley_targets = (valley_mask_arr > 0) & np.isfinite(cost)


# REVERSE COST-TO-VALLEY
print("Computing reverse accumulated cost from valley targets.")
reverse_dist = dijkstra_from_targets(valley_targets, cost, dem_arr)


# PROCESS START POINTS
ski_runs_records = []
corridor_id_raster = np.zeros(dem_arr.shape, dtype=np.int32)

for start_idx, geom in enumerate(start_points.geometry):
    print(f"Processing start point {start_idx}/{len(start_points)}")

    start_rc = point_to_rc(geom.x, geom.y, dem.transform)
    sr, sc = start_rc

    if not in_bounds(sr, sc, dem_arr.shape):
        print(f"- skipped: start point {start_idx} out of bounds")
        continue

    if not np.isfinite(cost[sr, sc]):
        print(f"- skipped: start point {start_idx} on invalid cell")
        continue

    forward_dist, prev = dijkstra_from_start(start_rc, cost, dem_arr)

    valid_end_mask = valley_targets & np.isfinite(forward_dist)
    if not np.any(valid_end_mask):
        print(f"- skipped: no downhill connection to valley for start {start_idx}")
        continue

    # use combined surface for best cost to avoid floating point mismatch
    combined = forward_dist + reverse_dist
    valid_combined = combined[valley_targets & np.isfinite(combined)]
    if len(valid_combined) == 0:
        print(f"- skipped: no finite combined cost for start {start_idx}")
        continue

    best_total_cost = float(np.min(valid_combined))
    if not np.isfinite(best_total_cost):
        print(f"- skipped: no finite path for start {start_idx}")
        continue

    # create corridor mask of threshold around best cost
    corridor_mask = (
        np.isfinite(combined) &
        (combined <= best_total_cost * (1.0 + CORRIDOR_RELAXATION))
    )
    corridor_mask &= np.isfinite(cost)

    # guarantee optimal path is always inside corridor (even if there are floating point issues that break up the corridor)
    best_valley_rc = np.unravel_index(
        np.argmin(np.where(valley_targets & np.isfinite(combined), combined, np.inf)), # find best valley cell based on combined cost
        combined.shape
    )
    optimal_path = reconstruct_path(prev, best_valley_rc)
    optimal_mask = path_to_mask(optimal_path, cost.shape)
    corridor_mask = corridor_mask | optimal_mask
    

    corridor_id_raster[corridor_mask] = start_idx + 1 # +1 to avoid 0 which is nodata value
    
    used_mask = np.zeros(cost.shape, dtype=bool)
    selected_paths = []

    for zone_val, zone_targets in valley_zones.items():
        for alt_id in range(NUM_SKI_RUNS):
            # apply diversity penalty around already selected runs
            if np.any(used_mask):
                penalty_zone = penalty_mask(used_mask, DIVERSITY_BUFFER_CELLS) # create a buffer zone around already selected runs where penalty will be applied
                extra_penalty = np.where(
                    penalty_zone & corridor_mask,
                    DIVERSITY_PENALTY, # apply penalty to corridor cells around already selected runs
                    0.0 # no penalty elsewhere
                )
            else:
                # no runs selected yet, so no penalty
                extra_penalty = np.zeros(cost.shape, dtype=np.float64)

            path, path_cost = shortest_path_in_corridor(
                start_rc=start_rc,
                corridor_mask=corridor_mask,
                base_cost_arr=cost,
                dem_arr=dem_arr,
                target_mask=zone_targets & corridor_mask,
                extra_penalty=extra_penalty
            )

            if path is None:
                print(f"- no more runs found for start {start_idx} zone {zone_val}")
                break

            # reject path that can't be converted into a valid linestring
            if len(path) < 2:
                print(f"- rejected single point path for start {start_idx} zone {zone_val}")
                continue

            selected_paths.append((path, path_cost))
            used_mask |= path_to_mask(path, cost.shape)

    if len(selected_paths) == 0:
        print(f"- no valid runs exist for start {start_idx}")
        continue

    # collect each run terrain stats for later difficulty grading
    for _, (path, path_cost) in enumerate(selected_paths, start=1):
        line = path_to_linestring(path, dem.transform)

        # (row, col)
        path_cells = [(r, c) for r, c in path if in_bounds(r, c, dem_arr.shape)]

        slope_vals = [slope_arr[r, c] for r, c in path_cells if not np.isnan(slope_arr[r, c])]
        if len(slope_vals) > 0:
            mean_slope = float(np.mean(slope_vals))
            max_slope = float(np.max(slope_vals))
        else:
            mean_slope = 0.0
            max_slope = 0.0

        roughness_vals = [roughness_arr[r, c] for r, c in path_cells if not np.isnan(roughness_arr[r, c])]
        if len(roughness_vals) > 0:
            mean_roughness = float(np.mean(roughness_vals))
        else:
            mean_roughness = 0.0

        # gather other stats for export
        elev_top = dem_arr[path_cells[0][0], path_cells[0][1]] # (row, col) of top cell of the run
        elev_bottom = dem_arr[path_cells[-1][0], path_cells[-1][1]] # (row, col) of bottom cell of the run
        elev_drop = elev_top - elev_bottom
        length_m = line.length

        ski_runs_records.append({
            "start_id": start_idx,
            "path_cost": float(path_cost),
            "n_cells": int(len(path)),
            "length_m": float(length_m),
            "elev_drop": float(elev_drop),
            "mean_slope": mean_slope,
            "max_slope": max_slope,
            "mean_roughness": mean_roughness,
            "geometry": line
        })

def normalise(values):
        arr = np.array(values, dtype=float)
        lo, hi = arr.min(), arr.max()
        if hi == lo:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

# ASSIGN DIFFICULTY (relative to selected area)
if len(ski_runs_records) > 0:

    max_slopes = [r["max_slope"] for r in ski_runs_records]
    mean_slopes = [r["mean_slope"] for r in ski_runs_records]
    mean_roughness = [r["mean_roughness"] for r in ski_runs_records]

    max_slope_n = normalise(max_slopes)
    slope_n = normalise(mean_slopes)
    roughness_n = normalise(mean_roughness)

    difficulty_scores = (
        max_slope_n * K_MAX_SLOPE + # k1
        roughness_n * K_MEAN_ROUGHNESS + # k2
        slope_n * K_MEAN_SLOPE # k3
    )

    p_blue = np.percentile(difficulty_scores, BLUE_PERCENTILE)
    p_red  = np.percentile(difficulty_scores, RED_PERCENTILE)
    print(f"Difficulty score thresholds — blue: <{p_blue:.3f}, red: <{p_red:.3f}, black: above")

    # difficulty category colors for export - for easy QGIS styling
    DIFF_COLOR = {"blue": "#1E90FF", "red": "#DC143C", "black": "#1A1A1A"}

    for i, (rec, score) in enumerate(zip(ski_runs_records, difficulty_scores)):
        rec["difficulty_score"] = float(score)
        if score < p_blue:
            rec["difficulty"] = "blue"
        elif score < p_red:
            rec["difficulty"] = "red"
        else:
            rec["difficulty"] = "black"

        rec["color_hex"]  = DIFF_COLOR[rec["difficulty"]]

        rec["norm_slope"] = float(slope_n[i])
        rec["norm_roughness"] = float(roughness_n[i])
        rec["norm_max_slope"] = float(max_slope_n[i])

    ski_runs_gdf = gpd.GeoDataFrame(ski_runs_records, crs=dem.crs)

    # clear existing layers from previous writes
    empty_gdf = gpd.GeoDataFrame(columns=["geometry"], crs=dem.crs)
    for layer in ("ski_runs_all", "ski_runs_blue", "ski_runs_red", "ski_runs_black"):
        empty_gdf.to_file(output_ski_runs_path, driver="GPKG", layer=layer)

    # single layer with all runs (all fields including difficulty + color)
    ski_runs_gdf.to_file(output_ski_runs_path, driver="GPKG", layer="ski_runs_all")

    # one layer per difficulty level
    for diff in ("blue", "red", "black"):
        subset = ski_runs_gdf[ski_runs_gdf["difficulty"] == diff]
        if len(subset):
            subset.to_file(
                output_ski_runs_path, driver="GPKG", layer=f"ski_runs_{diff}"
            )

    print(f"Saved ski runs to: {output_ski_runs_path} "
          f"(layers: ski_runs_all, ski_runs_blue, ski_runs_red, ski_runs_black)")

else:
    ski_runs_gdf = gpd.GeoDataFrame(
        columns=["start_id", 
                 "path_cost", 
                 "n_cells", 
                 "length_m", 
                 "mean_slope", 
                 "max_slope", 
                 "mean_roughness",
                 "elev_drop", 
                 "difficulty_score", 
                 "difficulty",
                 "color_hex",
                 "geometry"],
        crs=dem.crs
    )
    print("No ski runs were generated.")


# EXPORT CORRIDOR RASTER
corridor_profile = dem.profile.copy()
corridor_profile.update(
    dtype=rasterio.int32,
    count=1,
    nodata=0,
    compress="lzw"
)

with rasterio.open(output_corridor_raster_path, "w", **corridor_profile) as dst:
    dst.write(corridor_id_raster, 1)

print(f"Saved corridor raster to: {output_corridor_raster_path}")

# EXPORT COST RASTER
cost_profile = dem.profile.copy()
cost_profile.update(
    dtype=rasterio.float64,
    count=1,
    nodata=np.nan,
    compress="lzw"
)

display_cost = cost.copy()
display_cost[np.isinf(display_cost)] = np.nan

with rasterio.open(output_cost_raster_path, "w", **cost_profile) as dst:
    dst.write(display_cost, 1)

print(f"Saved cost raster to: {output_cost_raster_path}")

# PLOT FOR DEBUGGING
if SHOW_PLOT:
    fig, ax = plt.subplots(figsize=(12, 12))
    bounds = dem.bounds

    display_cost = cost.copy().astype(float)
    display_cost[np.isinf(display_cost)] = np.nan

    img = ax.imshow(
        display_cost,
        cmap="viridis",
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper"
    )
    plt.colorbar(img, ax=ax, label="Base cost")

    # plot corridors
    corridor_display = corridor_id_raster.astype(float)
    corridor_display[corridor_display == 0] = np.nan
    ax.imshow(
        corridor_display,
        cmap="Blues",
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        alpha=0.28
    )

    # plot end zone
    display_valley = valley_mask_arr.astype(float)
    display_valley[display_valley == 0] = np.nan
    ax.imshow(
        display_valley,
        cmap="Reds",
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        alpha=0.20
    )

    # plot start points
    start_points.plot(
        ax=ax,
        color="yellow",
        markersize=20,
        edgecolor="black"
    )

    # run coloured by difficulty
    diff_styles = {
        "blue": {"color": "#1E90FF", "linewidth": 2.5},
        "red": {"color": "#DC143C", "linewidth": 2.5},
        "black": {"color": "#1A1A1A", "linewidth": 2.5},
    }

    if len(ski_runs_gdf) > 0:
        for diff, style in diff_styles.items():
            subset = ski_runs_gdf[ski_runs_gdf["difficulty"] == diff]
            if len(subset):
                subset.plot(ax=ax, label=diff.capitalize(), **style)

    ax.set_title("Least-cost corridors and ski runs")
    ax.legend()
    plt.show()