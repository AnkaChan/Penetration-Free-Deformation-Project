#!/usr/bin/env python3
"""
viewer.py — Polyscope viewer with Prev/Next buttons
Allows viewing frames of a cloth simulation with per-vertex bounds
where .ply files are loaded from `Data2` or `Data3` directories
(adjust paths as needed).

Note: This script assumes you have already computed the bounds
using `compute_conservative_bounds.py` and have the .ply files ready.

** It takes a while to load the meshes and compute bounds, so be patient! **
"""

import os, glob, sys
import trimesh, numpy as np
import polyscope as ps
import polyscope.imgui as psim

# Make sure we can import compute_bounds from P01
HERE    = os.path.dirname(__file__)
P01_DIR = os.path.abspath(os.path.join(HERE, "..", "P01_ConservativeBounds"))
sys.path.insert(0, P01_DIR)
from compute_conservative_bounds import compute_bounds

# 1) Collect all truncated .plys from Data2/Data3
script_dir = os.path.dirname(__file__)
data2 = os.path.join(script_dir, "Data2")
data3 = os.path.join(script_dir, "Data3")
plys = sorted(glob.glob(os.path.join(data2, "step_*.ply")))

if not plys:
    raise RuntimeError(f"No .ply files in {data2} or {data3}")

# 2) Pre-load meshes & compute per-vertex bounds
meshes, bounds = [], []
print("Precomputing bounds…")
for ply in plys:
    mesh = trimesh.load(ply, process=False)
    bnd  = compute_bounds(mesh, max_dist=1e3, debug=False)
    meshes.append(mesh)
    bounds.append(bnd.astype(np.float32))
print("Done.")

# 3) Initialize Polyscope, register first mesh + scalar
ps.init()
surf = ps.register_surface_mesh("cloth",
    meshes[0].vertices, meshes[0].faces
)
surf.add_scalar_quantity("Bounds", bounds[0], defined_on="vertices")

# 4) Frame state
idx = 0

def set_frame(i):
    global idx
    idx = i % len(meshes)
    m   = meshes[idx]
    b   = bounds[idx]
    #  a) update geometry
    surf.update_vertex_positions(m.vertices)
    #  b) remove old scalar (if present)
    try:
        surf.remove_scalar_quantity("Bounds")
    except Exception:
        pass
    #  c) add new scalar
    surf.add_scalar_quantity("Bounds", b, defined_on="vertices")

# 5) ImGui button‐based UI
def user_callback():
    global idx

    # Prev / Next buttons (as before)
    if psim.Button("Prev"):
        set_frame(idx - 1)
    if psim.IsItemActive():         # continuous while held
        set_frame(idx - 1)

    psim.SameLine()

    if psim.Button("Next"):
        set_frame(idx + 1)
    if psim.IsItemActive():
        set_frame(idx + 1)

    # ← / → keys with repeat
    if psim.IsKeyPressed(psim.ImGuiKey_LeftArrow,  repeat=True):
        set_frame(idx - 1)
    if psim.IsKeyPressed(psim.ImGuiKey_RightArrow, repeat=True):
        set_frame(idx + 1)

ps.set_user_callback(user_callback)

print("Click Prev / Next (or ← / → arrow keys) to step through frames.")
ps.show()

