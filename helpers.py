import numpy as np
def compute_coefficients(distances, cosines):
    d1, d2, d3 = distances
    cos_a, cos_b, cos_g = cosines

    A4 = ((d1**2 - d3**2) / d2**2 - 1)**2 - (4 * d3**2 / d2**2 * cos_a**2)
    A3 = 4 * ((d1**2 - d3**2) / d2**2 * (1 - (d1**2 - d3**2) / d2**2) * cos_b - (1 - (d1**2 + d3**2) / d2**2) * cos_a * cos_g + 2 * d3**2 / d2**2 * cos_a**2 * cos_b)
    A2 = 2 * (((d1**2 - d3**2) / d2**2)**2 - 1 + 2 * ((d1**2 - d3**2) / d2**2)**2 * cos_b**2 + 2 * (d2**2 - d3**2) / d2**2 * cos_a**2 - 4 * (d1**2 + d3**2) / d2**2 * cos_a * cos_b * cos_g + 2 * (d2**2 - d1**2) / d2**2 * cos_g**2)
    A1 = 4 * (-(d1**2 - d3**2) / d2**2 * (1 + (d1**2 - d3**2) / d2**2) * cos_b + 2 * d1**2 / d2**2 * cos_g**2 * cos_b - (1 - (d1**2 + d3**2) / d2**2) * cos_a * cos_g)
    A0 = (1 + (d1**2 - d3**2) / d2**2)**2 - 4 * d1**2 / d2**2 * cos_g**2

    return [A4, A3, A2, A1, A0]

def find_real_roots(coefficients):
    return [np.real(root) for root in np.roots(coefficients) if np.isreal(root)]

def compute_u(v, distances, cosines):
    d1, d2, d3 = distances
    cos_a, cos_b, cos_g = cosines

    numerator = (-1 + (d1**2 - d3**2) / d2**2) * v**2 - 2 * (d1**2 - d3**2) / d2**2 * cos_b * v + 1 + (d1**2 - d3**2) / d2**2
    denominator = 2 * (cos_g - v * cos_a)
    u = numerator / denominator
    return u

def compute_s1_square(u, v, distances, cosines):
    d3 = distances[2]
    cos_g = cosines[2]

    s1_square = d3**2 / (1 + u**2 - 2 * u * cos_g)
    return s1_square

def compute_scales(s1_square, u, v):
    s1 = np.sqrt(s1_square)
    s2 = u * s1
    s3 = v * s1
    return s1, s2, s3

compute_3d_points = lambda scales, normalized_pixels: [s * pixel / np.linalg.norm(pixel) for s, pixel in zip(scales, normalized_pixels[1:])]
