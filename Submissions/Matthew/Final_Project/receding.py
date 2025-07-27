import numpy as np
import warp as wp
import trimesh
import polyscope as ps
from warp.sim.collide import TriMeshCollisionDetector

import glob
from os.path import join

device = wp.get_device('cuda')

data_path = r"Data"

shape_pre_initialization_files = sorted(glob.glob(join(data_path, "step_*_pre_initialization_shape.ply")))
displacement_initialization_files = sorted(glob.glob(join(data_path, "step_*_initialization_displacement.npy")))

shape_pre_iteration_files = sorted(glob.glob(join(data_path, "step_*_iter_*_shape.ply")))
displacement_iteration_files = sorted(glob.glob(join(data_path, "step_*_iter_*_displacement.npy")))

mesh_pre_init = trimesh.load(shape_pre_initialization_files[0])
displacement_init = np.load(displacement_initialization_files[0])
mesh_pre_init.show()

# Bounds Calculation Functions

@wp.func
def vertex_moving_away_from_vertex(v: wp.vec3, u: wp.vec3, d: wp.vec3):
  dir = wp.normalize(u - v)
  d = wp.normalize(d)
  return wp.dot(dir, d) > 0

# For a simple version, we can say that a vertex is moving away from a triangle if it moves away from all of the triangle's corners
@wp.func
def vertex_moving_away_from_triangle(v: wp.vec3, u1:wp.vec3, u2:wp.vec3, u3:wp.vec3, d:wp.vec3):
  return vertex_moving_away_from_vertex(v, u1, d) and vertex_moving_away_from_vertex(v, u2, d) and vertex_moving_away_from_vertex(v, u3, d)

@wp.func
def triangle_moving_away_from_vertex(v: wp.vec3, u1:wp.vec3, u2:wp.vec3, u3:wp.vec3, d1:wp.vec3, d2:wp.vec3, d3:wp.vec3):
  return vertex_moving_away_from_vertex(u1, v, d1) and vertex_moving_away_from_vertex(u2, v, d2) and vertex_moving_away_from_vertex(u3, v, d3)

@wp.func
def dist_from_edge_to_edge(v1: wp.vec3, v2: wp.vec3, u1: wp.vec3, u2: wp.vec3):
    # edge directions
    u = v2 - v1
    v = u2 - u1
    w = v1 - u1

    a = wp.dot(u, u)
    b = wp.dot(u, v)
    c = wp.dot(v, v)
    d = wp.dot(u, w)
    e = wp.dot(v, w)

    denom = a*c - b*b

    # first find s (on [0,1]) for closest point on edge v1→v2
    s = 0.0
    if denom > 1e-8:
        s = (b*e - c*d) / denom
        s = wp.min(1.0, wp.max(0.0, s))

    # then find t (on [0,1]) for closest point on edge u1→u2
    t = 0.0
    if c > 1e-8:
        t = (b*s + e) / c
        t = wp.min(1.0, wp.max(0.0, t))
    # refine s against this t
    if a > 1e-8:
        s = (b*t - d) / a
        s = wp.min(1.0, wp.max(0.0, s))

    # compute the two closest points and return their distance
    pc = v1 + u*s
    qc = u1 + v*t
    diff = pc - qc
    return wp.sqrt(wp.dot(diff, diff))

# for a more sophisitcated version, these functions should be edited to get more precise exclusions (For example, by checking if the trajectories do not intersect)
@wp.func
def exclude_vertex_triangle_pair(v:wp.vec3, u1:wp.vec3, u2:wp.vec3, u3:wp.vec3, d:wp.vec3, d1:wp.vec3, d2:wp.vec3, d3:wp.vec3):
  return vertex_moving_away_from_triangle(v, u1, u2, u3, d) and triangle_moving_away_from_vertex(v, u1, u2, u3, d1, d2, d3)
@wp.func
def exclude_edge_edge_pair(v1:wp.vec3, v2:wp.vec3, u1:wp.vec3, u2:wp.vec3, dv1:wp.vec3, dv2:wp.vec3, du1:wp.vec3, du2: wp.vec3):
  return 0 # STILL REQUIRES IMPLEMENTATION

# compute d^E_{min,v} using a kernel that allows looping over edge-edge pairs
@wp.kernel
def compute_d_min_E(
  vertices: wp.array(dtype=wp.vec3),
  edges: wp.array(dtype=wp.array(dtype=wp.int32)),
  displacements: wp.array(dtype=wp.vec3),
  d_min_edge: wp.array(dtype=wp.float32)
):
  i = wp.tid() # edge index
  d_min_edge[i] = 1e9 # Initializing the value with a large number
  v1 = vertices[edges[i][0]]
  v2 = vertices[edges[i][1]]
  dv1 = displacements[edges[i][0]]
  dv2 = displacements[edges[i][1]]
  for j in range(len(edges)):
    if i == j:
      continue
    dist = dist_from_edge_to_edge(v1, v2, vertices[edges[j][0]], vertices[edges[j][1]])
    if d_min_edge[edges[i][0]] < dist and d_min_edge[edges[i][1]] < dist: # Avoiding Unnecessary computations
      continue
    u1 = vertices[edges[j][0]]
    u2 = vertices[edges[j][1]]
    du1 = displacements[edges[j][0]]
    du2 = displacements[edges[j][1]]
    if exclude_edge_edge_pair(v1, v2, u1, u2, dv1, dv2, du1, du2):
      continue
    d_min_edge[edges[i][0]] = dist
    d_min_edge[edges[i][1]] = dist

@wp.kernel
def compute_d_min_v_and_d_min_T(
  vertices: wp.array(dtype=wp.vec3),
  faces: wp.array(dtype=wp.array(dtype=wp.int32)),
  displacements: wp.array(dtype=wp.vec3),
  d_min_v: wp.array(dtype=wp.float32),
  d_min_T: wp.array(dtype=wp.float32)
):
  i = wp.tid() # face index

@wp.kernel
def compute_new_bounds(
  d_min_v: wp.array(dtype=wp.float32),
  d_min_E: wp.array(dtype=wp.float32),
  d_min_T: wp.array(dtype=wp.float32),
  new_bounds: wp.array(dtype=wp.float32)
):
  i = wp.tid() # vertex index
  new_bounds[i] = min(d_min_v[i], d_min_E[i], d_min_T[i])
