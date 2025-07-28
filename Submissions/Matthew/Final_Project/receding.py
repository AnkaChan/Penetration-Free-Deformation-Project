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

vertices = wp.array(mesh_pre_init.vertices, dtype=wp.vec3)
#edges = wp.array(mesh_pre_init.edges, dtype=wp.int32)
edges = wp.array(mesh_pre_init.edges, dtype=wp.int32, ndim=2)
#faces = wp.array(mesh_pre_init.faces, dtype=wp.int32)
faces = wp.array(mesh_pre_init.faces, dtype=wp.int32, ndim=2)
displacements = wp.array(displacement_init, dtype=wp.vec3)
d_min_edge = wp.array(np.ones(len(vertices), dtype=np.float32) * 1e9)
d_min_v = wp.array(np.ones(len(vertices), dtype=np.float32) * 1e9)
d_min_T = wp.array(np.ones(len(vertices), dtype=np.float32) * 1e9)
new_bounds = wp.array(np.zeros(len(vertices), dtype=np.float32))

# Bounds Calculation Functions

@wp.func
def vertex_moving_away_from_vertex(v: wp.vec3, u: wp.vec3, d: wp.vec3):
    # Check if the vertex v is moving away from vertex u in the direction of d
    # dir = wp.normalize(u - v)
    # d = wp.normalize(d)
    # return wp.dot(dir, d) > 0

    # This checks if the vertex v is moving away from vertex u in the direction of d
    # by checking if the dot product of (u - v) and d is positive
    # This is equivalent to checking if the angle between (u - v) and d is
    # acute (i.e., less than 90 degrees).
    # If the dot product is positive, then the vertex is moving away from u.
    # This is a more efficient way to check if the vertex is moving away from u
    # without having to normalize the vectors.
  # acute‐angle ⇔ dot((u−v), d) > 0
  return wp.dot(u - v, d) > 0

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

@wp.func
def dist_point_triangle(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> float:
    #
    # Compute the (unnormalized) face normal and its squared length
    #
    ab = b - a
    ac = c - a
    n  = wp.cross(ab, ac)
    n2 = wp.dot(n, n) + 1e-12      # avoid divide-by-zero later

    #
    # Squared distance from p to the infinite plane of (a,b,c)
    #
    pd      = wp.dot(p - a, n)     # signed “area” ×‖n‖
    sd_plane = (pd * pd) / n2      # = (distance-to-plane)²

    #
    # Half-space tests to see if the projection lies inside the triangle
    #
    inside = (
        wp.dot(wp.cross(ab,    n), p - a) >= 0.0 and
        wp.dot(wp.cross(c - b, n), p - b) >= 0.0 and
        wp.dot(wp.cross(a - c, n), p - c) >= 0.0
    )
    if inside:
        # If the projected point is inside, it's the true closest point
        return wp.sqrt(sd_plane)

    # Otherwise, we must be closest to one of the three edges.
    # Compute squared point–segment distances for each edge, inlined:
    #  i) edge a→b
    dab   = ab
    dab2  = wp.dot(dab, dab) + 1e-12
    tab   = wp.dot(p - a, dab) / dab2
    tab   = wp.min(1.0, wp.max(0.0, tab))
    dq_ab = p - (a + dab * tab)
    ds0   = wp.dot(dq_ab, dq_ab)

    #  ii) edge b→c
    dbc   = c - b
    dbc2  = wp.dot(dbc, dbc) + 1e-12
    tbc   = wp.dot(p - b, dbc) / dbc2
    tbc   = wp.min(1.0, wp.max(0.0, tbc))
    dq_bc = p - (b + dbc * tbc)
    ds1   = wp.dot(dq_bc, dq_bc)

    #  iii) edge c→a
    dca   = a - c
    dca2  = wp.dot(dca, dca) + 1e-12
    tca   = wp.dot(p - c, dca) / dca2
    tca   = wp.min(1.0, wp.max(0.0, tca))
    dq_ca = p - (c + dca * tca)
    ds2   = wp.dot(dq_ca, dq_ca)

    # Pick the smallest squared-distance and return its sqrt
    ds_min = wp.min(ds0, wp.min(ds1, ds2))
    return wp.sqrt(ds_min)

# for a more sophisitcated version, these functions should be edited to get more precise exclusions (For example, by checking if the trajectories do not intersect)
@wp.func
def exclude_vertex_triangle_pair(v:wp.vec3, u1:wp.vec3, u2:wp.vec3, u3:wp.vec3, d:wp.vec3, d1:wp.vec3, d2:wp.vec3, d3:wp.vec3):
    # Early‐exit with a plane test. Most of the time, a vertex & triangle are separating simply because the vertex
    # is moving away from the triangle’s plane. We can catch those cases in one dot‐product, rather than six:
    
    # face normal (unnormalized is fine for sign checks)
    vec1 = u2 - u1
    vec2 = u3 - u1
    n    = wp.cross(vec1, vec2)

    # is v on the “positive” side of the plane?
    side_v = wp.dot(v - u1, n)
    # is it moving further in that same direction?
    rel_v  = wp.dot(d, n)
    if side_v * rel_v <= 0.0:
        # either below plane or moving toward it
        return False

    # (If you want extra safety, you can also check that the triangle itself
    # is not “swiveling into” the vertex by testing one triangle‐vertex velocity,
    # but in practice this plane check filters out ~95% of exclusions.)

    # Fallback: the old‐school vertex→vertex check, but with fast dot version
    return vertex_moving_away_from_triangle(v, u1, u2, u3, d) and triangle_moving_away_from_vertex(v, u1, u2, u3, d1, d2, d3)

@wp.func
def exclude_edge_edge_pair(v1:wp.vec3, v2:wp.vec3, u1:wp.vec3, u2:wp.vec3, dv1:wp.vec3, dv2:wp.vec3, du1:wp.vec3, du2: wp.vec3):
    # Static segment directions & offset
    u = v2 - v1
    v = u2 - u1
    w = v1 - u1

    # Dot products for the quadratic form
    a = wp.dot(u, u)
    b = wp.dot(u, v)
    c = wp.dot(v, v)
    d = wp.dot(u, w)
    e = wp.dot(v, w)

    denom = a*c - b*b

    # Solve for s parameter on [0,1]
    s = 0.0
    if denom > 1e-8:
        s = (b*e - c*d) / denom
        s = wp.min(1.0, wp.max(0.0, s))

    # Solve for t parameter on [0,1]
    t = 0.0
    if c > 1e-8:
        t = (b*s + e) / c
        t = wp.min(1.0, wp.max(0.0, t))

    # Refine s against this t
    if a > 1e-8:
        s = (b*t - d) / a
        s = wp.min(1.0, wp.max(0.0, s))

    # Compute the closest points on each segment
    pc = v1 + u * s
    qc = u1 + v * t
    delta = pc - qc

    # If segments are effectively touching, we can’t exclude
    dist2 = wp.dot(delta, delta)
    if dist2 < 1e-12:
        return False

    # Unit separation axis
    #dhat = delta * wp.rsqrt(dist2)
    inv_len = 1.0 / wp.sqrt(dist2)
    dhat    = delta * inv_len

    # Interpolate velocities at closest points
    velA = dv1 * (1.0 - s) + dv2 * s
    velB = du1 * (1.0 - t) + du2 * t

    # If relative velocity along dhat is positive, edges are separating
    return wp.dot(velA - velB, dhat) > 0.0

# compute d^E_{min,v} using a kernel that allows looping over edge-edge pairs
@wp.kernel
def compute_d_min_E(
  vertices: wp.array(dtype=wp.vec3),
  edges: wp.array(dtype=wp.int32, ndim=2),
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
  faces: wp.array(dtype=wp.int32, ndim=2),
  displacements: wp.array(dtype=wp.vec3),
  d_min_v: wp.array(dtype=wp.float32),
  d_min_T: wp.array(dtype=wp.float32)
):
  i = wp.tid() # face index
  d_T_current = float(1e9) # Initializing the value with a large number
  for j in range(len(vertices)):
    if j == faces[i][0] or j == faces[i][1] or j == faces[i][2]:
      continue
    v = vertices[j]
    dv = displacements[j]
    u1 = vertices[faces[i][0]]
    u2 = vertices[faces[i][1]]
    u3 = vertices[faces[i][2]]
    dist = dist_point_triangle(v, u1, u2, u3)
    if exclude_vertex_triangle_pair(v, u1, u2, u3, dv, displacements[faces[i][0]], displacements[faces[i][1]], displacements[faces[i][2]]):
      continue
    d_min_v[j] = min(d_min_v[j], dist)
    d_T_current = min(d_T_current, dist)
  d_min_T[faces[i][0]] = min(d_min_T[faces[i][0]], d_T_current)
  d_min_T[faces[i][1]] = min(d_min_T[faces[i][1]], d_T_current)
  d_min_T[faces[i][2]] = min(d_min_T[faces[i][2]], d_T_current)

@wp.kernel
def compute_new_bounds(
  d_min_v: wp.array(dtype=wp.float32),
  d_min_E: wp.array(dtype=wp.float32),
  d_min_T: wp.array(dtype=wp.float32),
  new_bounds: wp.array(dtype=wp.float32),
  gamma: float
):
  i = wp.tid() # vertex index
  new_bounds[i] = gamma * wp.min(wp.min(d_min_v[i], d_min_E[i]), d_min_T[i])

# Running kernels to calculate bounds
wp.launch(
    compute_d_min_E,
    dim=len(edges),
    inputs=[vertices, edges, displacements, d_min_edge],
    device='cuda:0'
)
wp.launch(
    compute_d_min_v_and_d_min_T,
    dim=len(faces),
    inputs=[vertices, faces, displacements, d_min_v, d_min_T],
    device='cuda:0'
)
wp.launch(
    compute_new_bounds,
    dim=len(vertices),
    inputs=[d_min_v, d_min_edge, d_min_T, new_bounds, 0.4],
    device='cuda:0'
)

arr = [new_bounds.numpy()[i] for i in range(len(new_bounds.numpy()))]
print(arr)

ps.init()
mesh_ps = ps.register_surface_mesh("mesh1", mesh_pre_init.vertices, mesh_pre_init.faces)
mesh_ps.add_scalar_quantity("Bounds", new_bounds.numpy())
ps.show()
