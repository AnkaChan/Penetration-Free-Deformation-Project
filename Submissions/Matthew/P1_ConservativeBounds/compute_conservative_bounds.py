#!/usr/bin/env python3
"""
compute_conservative_bounds.py

Given an input triangular mesh (.ply), build a Warp model,
run vertex–triangle & edge–edge collision queries, and
output a per-vertex bound b_i = min { d_vertex_triangle, d_edge_edge }.

Usage:
    python compute_conservative_bounds.py INPUT_MESH.ply -o bounds.npy
    -or-
    with the above hashbang and chmod +x compute_conservative_bounts.py:
    ./compute_conservative_bounds.py INPUT_MESH.ply -o bounds.npy
    e.g.:
    ./compute_conservative_bounds.py Cube_subdivided.ply -o my_bounds.npy
"""

import argparse
import numpy as np
import trimesh
import warp as wp
from warp.sim.collide import TriMeshCollisionDetector
from warp.sim.integrator_vbd import get_vertex_num_adjacent_edges, get_vertex_adjacent_edge_id_order, get_vertex_num_adjacent_faces, get_vertex_adjacent_face_id_order, ForceElementAdjacencyInfo

@wp.kernel
# how to iterate over neighbor elements
def iterate_vertex_neighbor_primitives(
    adjacency: ForceElementAdjacencyInfo
):
    particle_idx = wp.tid()

    # iterating over neighbor faces
    num_adj_faces = get_vertex_num_adjacent_faces(adjacency, particle_idx)
    for face_counter in range(num_adj_faces):
        adj_face_idx, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_idx, face_counter)
    # iterating over neighbor edges
    num_adj_edges = get_vertex_num_adjacent_edges(adjacency, particle_idx)
    for edge_counter in range(num_adj_edges):
        edge_idx, v_order = get_vertex_adjacent_edge_id_order(adjacency, particle_idx, edge_counter)

@wp.kernel
def aggregate_bounds_kernel(
    edges: wp.array(dtype=wp.int32, ndim=2),  # shape (n_edges,2)
    d_ee:   wp.array(dtype=float),            # shape (n_edges,)
    b:      wp.array(dtype=float)             # shape (n_verts,)
):
    e = wp.tid()                    # edge index
    i = edges[e, 0]                 # vertex i
    j = edges[e, 1]                 # vertex j
    d = d_ee[e]                     # edge–edge distance
    # atomically take the min with each endpoint’s bound
    wp.atomic_min(b, i, d)
    wp.atomic_min(b, j, d)

def build_warp_model(mesh: trimesh.Trimesh) -> wp.sim.Model:
    wp.init()  # must call once per process
    builder = wp.sim.ModelBuilder()

    # convert numpy vertices -> warp.vec3 list
    verts = [wp.vec3(*mesh.vertices[i]) for i in range(len(mesh.vertices))]
    idx  = mesh.faces.reshape(-1).tolist()

    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vertices=verts,
        indices=idx,
        vel=wp.vec3(0.0, 0.0, 0.0),
        density=0.02,
        tri_ke=1.0e5,
        tri_ka=1.0e5,
        tri_kd=2.0e-6,
        edge_ke=10,
    )

    # assign a default coloring so VBDIntegrator can initialize adjacency info
    builder.set_coloring([0] * len(mesh.vertices))

    return builder.finalize()

def compute_bounds(mesh: trimesh.Trimesh, max_dist: float = 1e3, debug = True) -> np.ndarray:
    """
    Returns b: per-vertex conservative bounds b_i using Warp kernels.
    """
    model    = build_warp_model(mesh)
    
    # to access ForceElementAdjacencyInfo, you need to construct a VBDIntegrator (you dont need to understand what it is)
    vbd_integrator = wp.sim.VBDIntegrator(model)
    
    # launch the adjacency‐iteration kernel (for debugging or data‐collection)
    wp.launch(
        iterate_vertex_neighbor_primitives,
        dim=model.particle_count,
        inputs=[vbd_integrator.adjacency]
    )
    
    collision_detector = TriMeshCollisionDetector(model)

    # run detection
    collision_detector.vertex_triangle_collision_detection(max_dist)
    collision_detector.edge_edge_collision_detection(max_dist)

    # debugging output
    if debug:
        print("# d^v_min")
        print(collision_detector.vertex_colliding_triangles_min_dist)
        print("# d^E_min")
        print(collision_detector.edge_colliding_edges_min_dist)
        print("# d^T_min")
        print(collision_detector.triangle_colliding_vertices_min_dist)

    '''
    # read out distances (warp arrays -> numpy)
    d_vt = collision_detector.vertex_colliding_triangles_min_dist.numpy()  # (n_verts,)
    d_ee = collision_detector.edge_colliding_edges_min_dist.numpy()        # (n_edges,)

    # build per-vertex min over edge distances
    b = d_vt.copy()
    edges = np.array(mesh.edges_unique)  # (n_edges, 2)
    for e_idx, (i, j) in enumerate(edges):
        d = d_ee[e_idx]
        # clamp both endpoints
        if d < b[i]:
            b[i] = d
        if d < b[j]:
            b[j] = d
            
    return b
    '''
    
    # Implement conservative bounds computation using the provided instructions
    # It must be implemented using @warp.kernel to maximize efficiency

    # grab Warp arrays directly (no .numpy())
    d_vt_wp = collision_detector.vertex_colliding_triangles_min_dist   # wp.array float32, shape (n_verts,)
    d_ee_wp = collision_detector.edge_colliding_edges_min_dist         # wp.array float32, shape (n_edges,)
    b_wp = wp.clone(d_vt_wp) # prepare the per-vertex bound buffer as a copy of d_vt_wp

    # pack edges into a Warp array
    edges_np = np.array(mesh.edges_unique, dtype=np.int32)
    edges_wp = wp.array(edges_np)  # wp.array int32, shape (n_edges,2)

    # launch the aggregation kernel
    wp.launch(
        aggregate_bounds_kernel,
        dim=edges_np.shape[0],
        inputs=[edges_wp, d_ee_wp, b_wp]
    )

    # return as numpy array
    return b_wp.numpy()

def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mesh', help="Input .ply mesh")
    p.add_argument('-o', '--output', default='bounds.npy', help="Output NumPy file for b_i")
    args = p.parse_args()

    mesh = trimesh.load(args.mesh, process=False)
    b    = compute_bounds(mesh, 5.0) # 5.0 used in reference example
    np.save(args.output, b)
    print(f"[+] Saved {len(b)} bounds to '{args.output}'")

if __name__ == '__main__':
    main()
