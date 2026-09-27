"""NumPy CAHVOR geometry for a bounded, non-operational stereo pilot.

Projection equations follow NASA VICAR cmod_cahvor (C/A/H/V/O/R).
https://github.com/NASA-AMMOS/VICAR/blob/master/vos/p2/sub/cahvor/cmod_cahvor.c
Optional bounded orientation refinement is explicit; camera centers, metric scale
and distortion remain from metadata. No image-based scale estimation.
"""
import numpy as np


def unit(values):
    values = np.asarray(values, dtype=np.float64)
    length = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(~np.isfinite(length)) or np.any(length < 1e-12):
        raise ValueError('Cannot normalize a zero/nonfinite vector')
    return values / length


def vectors(model):
    result = {key: np.asarray(model[key], dtype=np.float64) for key in 'CAHVOR'}
    if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError('CAHVOR requires six finite 3-vectors')
    for key in 'AO':
        if abs(np.linalg.norm(result[key]) - 1) > 1e-5:
            raise ValueError('A/O must be unit vectors')
    return result


def project_direction(model, direction):
    m = vectors(model)
    direction = np.asarray(direction, dtype=np.float64)
    omega = direction @ m['O']
    lam = direction - omega[..., None] * m['O']
    tau = np.sum(lam * lam, axis=-1) / omega**2
    mu = m['R'][0] + m['R'][1]*tau + m['R'][2]*tau**2
    distorted = direction + mu[..., None] * lam
    alpha = distorted @ m['A']
    xy = np.stack((distorted @ m['H'] / alpha, distorted @ m['V'] / alpha), axis=-1)
    return xy, (alpha > 0) & (omega > 0) & np.all(np.isfinite(xy), axis=-1)


def pixel_rays(model, pixels):
    m = vectors(model)
    pixels = np.asarray(pixels, dtype=np.float64)
    ray = np.cross(m['V'] - pixels[..., 1, None]*m['A'],
                   m['H'] - pixels[..., 0, None]*m['A'])
    ray *= np.where(ray @ m['A'] < 0, -1, 1)[..., None]
    ray = unit(ray)
    omega = ray @ m['O']
    if np.any(omega <= 0):
        raise ValueError('Ray outside forward optical hemisphere')
    lam = ray - omega[..., None]*m['O']
    tau = np.sum(lam*lam, axis=-1) / omega**2
    k1, k3, k5 = 1+m['R'][0], m['R'][1]*tau, m['R'][2]*tau**2
    u = 1 - (m['R'][0] + k3 + k5)
    for _ in range(30):
        poly = ((k5*u*u + k3)*u*u + k1)*u - 1
        derivative = (5*k5*u*u + 3*k3)*u*u + k1
        if np.any(derivative <= 1e-12):
            raise ValueError('Noninvertible radial distortion')
        step = poly/derivative
        u -= step
        if np.max(np.abs(step)) < 1e-12:
            break
    if np.max(np.abs(((k5*u*u + k3)*u*u + k1)*u - 1)) > 1e-9:
        raise ValueError('Distortion inversion did not converge')
    return unit(ray - (1-u)[..., None]*lam)


def stereo_frame(left, right):
    for key in ('frame', 'frame_indices', 'frame_solution'):
        if left.get(key) is None or left[key] != right.get(key):
            raise ValueError('Camera centers must share the identical indexed coordinate frame')
    baseline = np.asarray(right['C']) - left['C']
    x = unit(baseline)
    look = unit(np.asarray(left['A']) + right['A'])
    z = unit(look - np.dot(look, x)*x)
    y = unit(np.cross(z, x))
    basis = np.stack((x, y, z))  # world direction -> rectified coordinates
    return basis, float(np.linalg.norm(baseline))


def rotate_camera(model, axis, angle):
    """Explicit orientation refinement; camera center and distortion stay fixed."""
    axis = unit(axis)
    result = dict(model)
    for key in 'AHVO':
        value = np.asarray(model[key])
        result[key] = (value*np.cos(angle) + np.cross(axis,value)*np.sin(angle)
                       + axis*np.dot(axis,value)*(1-np.cos(angle))).tolist()
    return result


def fit_epipolar_pitch(left, right, pixels_left, pixels_right, baseline_axis):
    """Fit one rotation to image correspondences, not to semantic/depth GT."""
    rays_left, rays_right = pixel_rays(left,pixels_left), pixel_rays(right,pixels_right)
    normals = unit(np.cross(baseline_axis,rays_left))
    a = np.sum(normals*rays_right,axis=-1)
    b = np.sum(normals*np.cross(baseline_axis,rays_right),axis=-1)
    angles = np.arctan2(-a,b)
    angles = (angles+np.pi/2) % np.pi-np.pi/2
    angle = float(np.median(angles))
    if abs(angle) > np.deg2rad(0.1):
        raise ValueError('Required pitch correction exceeds bounded pilot limit 0.1 degree')
    return angle


def rectification_parameters(left, right, width=768, height=512, center_depth=3.0):
    basis, baseline = stereo_frame(left, right)
    a = np.asarray(left['A'])
    h = np.asarray(left['H'])
    focal = float(np.linalg.norm(h - np.dot(h, a)*a) * 0.5)
    middle = [(right['axes']['Sample']-1)/2, (right['axes']['Line']-1)/2]
    center_ray = pixel_rays(right, middle) @ basis.T
    cx_right = (width-1)/2 - focal*center_ray[0]/center_ray[2]
    cy = (height-1)/2 - focal*center_ray[1]/center_ray[2]
    return {'basis': basis, 'baseline_m': baseline, 'focal_px': focal,
            'cx_left': cx_right-focal*baseline/center_depth, 'cx_right': cx_right,
            'cy': cy, 'width': width, 'height': height, 'center_depth_m': center_depth}


def rectification_map(model, params, eye):
    yy, xx = np.indices((params['height'], params['width']), dtype=np.float64)
    ray_rect = np.stack(((xx-params['cx_'+eye])/params['focal_px'],
                         (yy-params['cy'])/params['focal_px'], np.ones_like(xx)), axis=-1)
    source_xy, valid = project_direction(model, ray_rect @ params['basis'])
    valid &= ((source_xy[..., 0] >= 3) & (source_xy[..., 0] < model['axes']['Sample']-4)
              & (source_xy[..., 1] >= 3) & (source_xy[..., 1] < model['axes']['Line']-4))
    return source_xy.astype(np.float32), valid


def disparity_xyz(disparity, params):
    yy, xx = np.indices(disparity.shape, dtype=np.float64)
    effective = disparity - (params['cx_left']-params['cx_right'])
    with np.errstate(divide='ignore', invalid='ignore'):
        z = params['focal_px']*params['baseline_m']/effective
    rect = np.stack(((xx-params['cx_left'])*z/params['focal_px'],
                      (yy-params['cy'])*z/params['focal_px'], z), axis=-1)
    return rect @ params['basis'], z
