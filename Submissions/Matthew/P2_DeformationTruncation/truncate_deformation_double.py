#!/usr/bin/env python3
"""
truncate_deformation_double.py

For every “step_*_pre_initialization_shape.ply” and
its matching “step_*_initialization_displacement.npy”:
 - compute conservative bounds (via P01),
 - clamp the displacement,
 - run a true double-precision triangle-triangle intersection test,
 - emit a truncated .ply and report the intersection count
 
Note: uses CPU, not GPU (Warp), so it is very slow for large meshes
"""

import os
import glob
import numpy as np
import trimesh

from compute_conservative_bounds import compute_bounds  # P01 routine

def tri_tri_intersect(v0, v1, v2, u0, u1, u2, eps=1e-12):
    """
    True double-precision triangle-triangle overlap test
    (Möller’s plane-splitting + coplanar SAT fallback).
    Returns True if triangles (v0,v1,v2) and (u0,u1,u2) overlap.
    """
    # ensure double precision arrays
    v0, v1, v2 = map(lambda x: np.asarray(x, dtype=np.float64), (v0, v1, v2))
    u0, u1, u2 = map(lambda x: np.asarray(x, dtype=np.float64), (u0, u1, u2))

    # Reject by plane separation ---
    # Triangle 1 plane
    N1 = np.cross(v1 - v0, v2 - v0)
    d1 = -np.dot(N1, v0)
    du = np.dot(N1, u0) + d1, np.dot(N1, u1) + d1, np.dot(N1, u2) + d1
    du = tuple(0.0 if abs(d) < eps else d for d in du)
    if (du[0] > 0 and du[1] > 0 and du[2] > 0) or (du[0] < 0 and du[1] < 0 and du[2] < 0):
        return False

    # Triangle 2 plane
    N2 = np.cross(u1 - u0, u2 - u0)
    d2 = -np.dot(N2, u0)
    dv = np.dot(N2, v0) + d2, np.dot(N2, v1) + d2, np.dot(N2, v2) + d2
    dv = tuple(0.0 if abs(d) < eps else d for d in dv)
    if (dv[0] > 0 and dv[1] > 0 and dv[2] > 0) or (dv[0] < 0 and dv[1] < 0 and dv[2] < 0):
        return False

    # If planes nearly parallel → coplanar test
    D = np.cross(N1, N2)
    normD = np.linalg.norm(D)
    if normD < eps:
        # pick projection axis
        idx = np.argmax(np.abs(N1))
        def proj2(p):
            return {0: p[1:], 1: p[[0,2]], 2: p[:2]}[idx]
        T1 = [proj2(v) for v in (v0, v1, v2)]
        T2 = [proj2(u) for u in (u0, u1, u2)]

        # 2D SAT
        def axes(tri):
            out = []
            for i in range(3):
                e = tri[(i+1)%3] - tri[i]
                a = np.array([-e[1], e[0]])
                n = np.linalg.norm(a)
                if n > eps:
                    out.append(a / n)
            return out

        for axis in axes(T1) + axes(T2):
            p1 = [np.dot(pt, axis) for pt in T1]
            p2 = [np.dot(pt, axis) for pt in T2]
            if max(p1) < min(p2) or max(p2) < min(p1):
                return False
        return True

    # Non-coplanar: intersect each triangle with the other's plane ---
    def seg_plane(tri, N, d):
        pts = tri
        ds = [np.dot(N, p) + d for p in pts]
        pts_out = []
        for i in range(3):
            j = (i+1) % 3
            if abs(ds[i]) < eps:
                pts_out.append(pts[i])
            if ds[i] * ds[j] < 0:
                t = ds[i] / (ds[i] - ds[j])
                pts_out.append(pts[i] + (pts[j] - pts[i]) * t)
        # we expect 2 unique points
        uniq = []
        for p in pts_out:
            if not any(np.linalg.norm(p - q) < eps for q in uniq):
                uniq.append(p)
        return uniq[:2] if len(uniq) >= 2 else None

    seg1 = seg_plane((v0, v1, v2), N2, d2)
    seg2 = seg_plane((u0, u1, u2), N1, d1)
    if seg1 is None or seg2 is None:
        return False

    #  Project both segments onto intersection line and test interval overlap
    D_unit = D / normD
    t1 = [np.dot(p, D_unit) for p in seg1]
    t2 = [np.dot(p, D_unit) for p in seg2]
    if max(t1) < min(t2) or max(t2) < min(t1):
        return False

    return True


def check_intersections_double(vertices: np.ndarray,
                                faces: np.ndarray) -> int:
    """
    Brute-force self-intersection check:
    loop over all non-adjacent triangle pairs,
    test with tri_tri_intersect, and count overlaps.
    """
    count = 0
    n_faces = faces.shape[0]
    for i in range(n_faces):
        idx1 = faces[i]
        tri1 = vertices[idx1]
        for j in range(i + 1, n_faces):
            idx2 = faces[j]
            # skip sharing a vertex (adjacent)
            if set(idx1).intersection(idx2):
                continue
            tri2 = vertices[idx2]
            if tri_tri_intersect(*tri1, *tri2):
                count += 1
    return count


def truncate_displacement(vert: np.ndarray,
                          disp: np.ndarray,
                          bounds: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(disp, axis=1)
    safe = norms > 0
    factors = np.ones_like(norms)
    factors[safe] = np.minimum(1.0, bounds[safe] / norms[safe])
    return disp * factors[:, None]


def main():
    data_path = "Data"
    script_dir = os.path.dirname(__file__)
    out_dir    = os.path.join(script_dir, "Data3")
    os.makedirs(out_dir, exist_ok=True)

    pre_files  = sorted(glob.glob(os.path.join(data_path, "step_*_pre_initialization_shape.ply")))
    disp_files = sorted(glob.glob(os.path.join(data_path, "step_*_initialization_displacement.npy")))

    for mesh_path, disp_path in zip(pre_files, disp_files):
        step = os.path.basename(mesh_path).split("_")[1]
        mesh = trimesh.load(mesh_path, process=False)
        disp = np.load(disp_path)  # (n_vertices, 3)

        # conservative bounds
        bnd = compute_bounds(mesh)

        # truncate
        truncated_disp = truncate_displacement(mesh.vertices, disp, bnd)
        new_verts      = mesh.vertices + truncated_disp

        # double-precision self‐intersection check
        n_int = check_intersections_double(new_verts, mesh.faces)
        print(f"Step {step:>4s} → intersections: {n_int}")

        # export truncated mesh
        out_name = f"step_{step}_truncated_shape.ply"
        out_path = os.path.join(out_dir, out_name)
        out_mesh = trimesh.Trimesh(vertices=new_verts, faces=mesh.faces)
        out_mesh.export(out_path)


if __name__ == "__main__":
    main()
