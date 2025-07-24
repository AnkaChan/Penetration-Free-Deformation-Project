#!/usr/bin/env python3
"""
truncate_deformation.py

For every time step:
  1) Load pre-init mesh + per-iteration displacement guesses
  2) Compute conservative bounds
  3) Inner optimization:
       • collision detection
       • evaluate E_e + E_c and their gradients
       • descent Δx, clamp by bounds
       • update x, break if converged
  4) Export optimized mesh
"""

import os
import glob
import numpy as np
import trimesh
import warp as wp
from warp.sim.collide import TriMeshCollisionDetector

# import the function from P01
from compute_conservative_bounds import compute_bounds

# -----------------------------------------------------------------------------
# Hyper-parameters
MAX_ITERS   = 20      # inner-loop max iterations
CONV_TOL    = 1e-5    # convergence on update norm
MAX_DIST    = 1e3     # for collision queries
ALPHA       = 1e-3    # gradient descent step size
EDGE_KE     = 10.0    # spring stiffness for elastic energy
# -----------------------------------------------------------------------------

# back-tracking line search
def line_search(x, grad, E0, mesh, rest_edges, rest_lengths, detector):
    n = 1.0
    while n > 1e-6:
        x_trial = x - n*grad
        E_e, _ = evaluate_elastic_energy(x_trial, mesh, rest_edges, rest_lengths, EDGE_KE)
        E_c, _ = evaluate_collision_energy(x_trial, detector, mesh)
        if E_e + E_c < E0:
            return -n * grad
        n *= 0.5
    return -n * grad

def evaluate_elastic_energy(x: np.ndarray,
                            mesh: trimesh.Trimesh,
                            rest_edges: np.ndarray,
                            rest_lengths: np.ndarray,
                            ke: float
                           ) -> (float, np.ndarray):
    """
    Simple spring energy on mesh edges:
      E = ½·ke·∑ₑ (‖x_i - x_j‖ - L₀ₑ)²
    Returns (E, grad) where grad has shape (n_verts,3).
    """
    verts = x
    edges = rest_edges
    diffs = verts[edges[:,0]] - verts[edges[:,1]]          # (n_edges,3)
    lens  = np.linalg.norm(diffs, axis=1)                   # (n_edges,)
    dl    = lens - rest_lengths                            # (n_edges,)

    E = 0.5 * ke * np.sum(dl**2)

    # compute gradient
    grad = np.zeros_like(verts)
    # avoid division by zero
    nonzero = lens > 1e-12
    # force magnitude for each edge
    f_mag = np.zeros_like(lens)
    f_mag[nonzero] = ke * dl[nonzero] / lens[nonzero]
    # accumulate
    for e,(i,j) in enumerate(edges):
        f = f_mag[e] * diffs[e]
        grad[i] +=  f
        grad[j] -=  f

    return E, grad

def evaluate_collision_energy_old(x: np.ndarray,
                              detector: TriMeshCollisionDetector
                             ) -> (float, np.ndarray):
    """
    Placeholder zero-energy+zero-gradient.
    You could plug in barrier penalties based on detector distances here.
    """
    n = x.shape[0]
    return 0.0, np.zeros((n,3), dtype=x.dtype)

def evaluate_collision_energy(x: np.ndarray,
                              detector: TriMeshCollisionDetector,
                              mesh: trimesh.Trimesh,
                              kc: float = 1e4,   # barrier stiffness
                              r:  float = 0.0    # barrier radius
                             ) -> (float, np.ndarray):
    """
    Barrier energy & gradient using per-vertex normals from the mesh.
    """
    # run the vertex–triangle query on x
    x_wp = wp.array(x.astype(np.float32), dtype=wp.vec3)
    detector.refit(x_wp)
    detector.vertex_triangle_collision_detection(MAX_DIST)

    # pull out the distances as a NumPy array
    dists = detector.vertex_colliding_triangles_min_dist.numpy()  # (n_verts,)

    # compute barrier violation = max(0, r - d)
    violation = np.maximum(0.0, r - dists)    # (n_verts,)

    # energy = ½·kc·∑ violation²
    E_c = 0.5 * kc * np.sum(violation**2)

    # gradient = -kc·violation·nᵢ  (using mesh‐computed vertex normals)
    normals = mesh.vertex_normals           # (n_verts,3), unit-length
    grad_c = - (kc * violation)[:, None] * normals

    return E_c, grad_c

# -----------------------------------------------------------------------------
def truncate_deformation_pipeline(data_path: str, out_dir: str):
    # read files
    pre_files  = sorted(glob.glob(os.path.join(data_path, "step_*_pre_initialization_shape.ply")))
    iter_files = sorted(glob.glob(os.path.join(data_path, "step_*_iter_*_displacement.npy")))

    # group iteration files by timestep index
    from collections import defaultdict
    grouped = defaultdict(list)
    for f in iter_files:
        idx = os.path.basename(f).split("_")[1]
        grouped[idx].append(f)
    for idx in grouped:
        grouped[idx].sort()

    os.makedirs(out_dir, exist_ok=True)

    # loop over timesteps
    for mesh_path in pre_files:
        step_idx = os.path.basename(mesh_path).split("_")[1]
        disp_list = grouped.get(step_idx, [])
        if not disp_list:
            print(f"[!] No iteration files for step {step_idx}, skipping.")
            continue

        print(f"=== Time step {step_idx} ===")
        # load pre-init mesh
        mesh = trimesh.load(mesh_path, process=False)
        X0   = mesh.vertices.copy()        # rest positions

        # build spring rest data
        rest_edges   = np.array(mesh.edges_unique, dtype=np.int32)
        rest_lengths = np.linalg.norm(
            X0[rest_edges[:,0]] - X0[rest_edges[:,1]],
            axis=1
        )

        # initial guess x = X0 + first iter displacement
        disp0 = np.load(disp_list[0])
        x     = X0 + disp0

        # build Warp CCD detector once
        wp.init()
        builder = wp.sim.ModelBuilder()
        verts_wp = [wp.vec3(*v) for v in X0]
        idx_wp   = mesh.faces.reshape(-1).tolist()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vertices=verts_wp,
            indices=idx_wp,
            vel=wp.vec3(0.0),
            density=1.0,
            tri_ke=1e5, tri_ka=1e5, tri_kd=2e-6,
            edge_ke=EDGE_KE
        )
        model    = builder.finalize()
        detector = TriMeshCollisionDetector(model)
        
        #print([a for a in dir(detector) if "intersect" in a.lower()])

        # compute conservative bounds once
        bnd = compute_bounds(mesh, max_dist=MAX_DIST, debug=False)
        
        # gather all the iteration‐displacement files for this time step
        iter_files = sorted(glob.glob(os.path.join(data_path, f"step_{step_idx}_iter_*_displacement.npy")))
        print(f"  → conservative bounds: min={bnd.min():.4f}, max={bnd.max():.4f}")

        # inner optimization loop
        for i in range(MAX_ITERS):
            # ——— Warm start from provided file, if available ———
            if i < len(iter_files):
                # load displacement for iteration i
                disp_i = np.load(iter_files[i])           # shape (n_verts,3)
                x      = X0 + disp_i                      # override x
                print(f"    iter {i}: warm‐start from {os.path.basename(iter_files[i])}")
                dx_set = False
            else:
                # No more warm‐starts... perform gradient‐descent + clamp
                # evaluate energies & gradients:
                E_e, grad_e = evaluate_elastic_energy(x, mesh, rest_edges, rest_lengths, EDGE_KE)
                #E_c, grad_c = evaluate_collision_energy(x, detector)
                E_c, grad_c = evaluate_collision_energy(x, detector, mesh, kc=1e4, r=0.0)
                grad = grad_e + grad_c # total gradient (force = ∇E)
                #dx = -ALPHA * grad # descent step
                E0 = E_e + E_c
                dx = line_search(x, grad, E0, mesh, rest_edges, rest_lengths, detector)

                # clamp by conservative bounds
                norms = np.linalg.norm(dx, axis=1)
                mask  = norms > 0
                factors = np.ones_like(norms)
                factors[mask] = np.minimum(1.0, bnd[mask] / norms[mask])
                dx = dx * factors[:, None]

                #    d) apply update
                x = x + dx
                dx_set = True

                print(f"    iter {i}: grad‐step ‖Δx‖={np.linalg.norm(dx):.3e}")
            
            # collision detection on current x
            x_wp = wp.array(x.astype(np.float32), dtype=wp.vec3)
            detector.refit(x_wp)

            # Convergence check: break if no intersections
            detector.triangle_triangle_intersection_detection() # triangle–triangle intersection kernel
            count_wp = detector.triangle_intersecting_triangles_count # 0D Warp array
            counts = count_wp.numpy()
            if np.all(counts == 0):
                print(f"    iter {i}: no collisions → converged")
                break
                
            # check convergence based on tiny update step
            if dx_set and np.linalg.norm(dx) < CONV_TOL:
                print(f"  iter {i}: converged (‖Δx‖={np.linalg.norm(dx):.3e})")
                break
        else:
            print(f"  reached max iterations ({MAX_ITERS})")

        # write out optimized mesh
        out_path = os.path.join(out_dir, f"step_{step_idx}_optimized_shape.ply")
        out_mesh = trimesh.Trimesh(vertices=x, faces=mesh.faces)
        out_mesh.export(out_path)
        print(f"→ Wrote {out_path}")

# -----------------------------------------------------------------------------
def main():
    data_path = "Data"
    script_dir = os.path.dirname(__file__)
    out_dir    = os.path.join(script_dir, "Data2")
    os.makedirs(out_dir, exist_ok=True)
    truncate_deformation_pipeline(data_path, out_dir)

if __name__ == "__main__":
    main()
