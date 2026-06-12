import torch

EPSILON = 1e-8

def get_cosine(a, b):
    if a.dim() == 1: a = a.unsqueeze(0)
    if b.dim() == 1: b = b.unsqueeze(0)
    a_magnitudes = torch.linalg.vector_norm(a, dim=1)
    b_magnitudes = torch.linalg.vector_norm(b, dim=1)
    dot_ab = torch.linalg.vecdot(a, b, dim=1)
    return torch.clamp(dot_ab / (a_magnitudes * b_magnitudes), -1 + EPSILON, 1 - EPSILON)

def get_angle(a, b):
    return torch.acos(get_cosine(a, b))

def get_intersection_points(p1, p2, a, b, c, d):
    n = torch.stack([a, b, c])
    v = p2 - p1
    denominator = (n * v).sum(dim=1)
    if torch.any(denominator.abs() < EPSILON): 
        raise RuntimeError("Line is parallel to near clipping plane.")
    t = -((n * p1).sum(dim=1) + d) / denominator
    return p1 + t.unsqueeze(1) * v

def get_normals(surfaces):
    ab = surfaces[:, 1, :] - surfaces[:, 0, :]
    ac = surfaces[:, 2, :] - surfaces[:, 0, :]
    return torch.linalg.cross(ab, ac, dim=1)

def get_depth_planes(screen_coordinates):
    screen_normals = get_normals(screen_coordinates)
    a, b, c = screen_normals[:, 0], screen_normals[:, 1], screen_normals[:, 2]
    d = -a * screen_coordinates[:, 0, 0] - b * screen_coordinates[:, 0, 1] - c * screen_coordinates[:, 0, 2]
    return torch.stack((a, b, c, d), dim=1)

def get_general_equations(screen_coordinates):
    x, y = screen_coordinates[:, :, 0], screen_coordinates[:, :, 1]
    a = y[:, [0, 1, 2]] - y[:, [1, 2, 0]]
    b = x[:, [1, 2, 0]] - x[:, [0, 1, 2]]
    c = -a * x[:, [0, 1, 2]] - b * y[:, [0, 1, 2]]

    double_corners = ((x[:, [0, 1, 2]] - x[:, [1, 2, 0]]) ** 2 + (y[:, [0, 1, 2]] - y[:, [1, 2, 0]]) ** 2 < EPSILON).any(dim=1)
    substitution = a * x[:, [2, 0, 1]] + b * y[:, [2, 0, 1]] + c
    flat = (torch.abs(substitution) < EPSILON).any(dim=1)
    degenerate = torch.logical_or(double_corners, flat)
    abc = torch.stack([a, b, c], dim=2)
    return torch.where((substitution > 0).unsqueeze(2), abc, -abc), degenerate

def get_clipped_mask_2(surfaces, a, b, c, d):
    new_corners_a1 = get_intersection_points(surfaces[:, 0, :], surfaces[:, 1, :], a, b, c, d)
    new_corners_a2 = get_intersection_points(surfaces[:, 0, :], surfaces[:, 2, :], a, b, c, d)
    new_surfaces_1 = torch.stack((new_corners_a1, surfaces[:, 1, :], surfaces[:, 2, :]), dim=1)
    new_surfaces_2 = torch.stack((new_corners_a1, new_corners_a2, surfaces[:, 2, :]), dim=1)
    return torch.cat((new_surfaces_1, new_surfaces_2), dim=0)
        
def get_clipped_mask_1(surfaces, a, b, c, d):
    new_corners_a = get_intersection_points(surfaces[:, 0, :], surfaces[:, 2, :], a, b, c, d)
    new_corners_b = get_intersection_points(surfaces[:, 1, :], surfaces[:, 2, :], a, b, c, d)
    return torch.stack((new_corners_a, new_corners_b, surfaces[:, 2, :]), dim=1)