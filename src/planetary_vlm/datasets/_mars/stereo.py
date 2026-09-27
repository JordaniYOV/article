"""CPU-only CAHVOR/SGBM technical pilot. Never admits depth to the benchmark."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from planetary_vlm.datasets.cahvor import (rectification_parameters, rectification_map,
                                         disparity_xyz, project_direction, pixel_rays,
                                         rotate_camera, fit_epipolar_pitch)


def run(audit_path, output, sol, refine_pointing=False, *, left_product=None, right_product=None):
    if output.exists():
        raise FileExistsError('Pilot version exists; select a new output')
    started = perf_counter()
    audit = json.loads(audit_path.read_text())
    candidates = [row for row in audit['pairs'] if row['sol'] == sol
                  and (left_product is None or row['left'] == left_product)
                  and (right_product is None or row['right'] == right_product)]
    if len(candidates) != 1:
        raise ValueError('Select exactly one pair; use explicit left/right IDs when a sol is ambiguous')
    pair = candidates[0]
    left, right = [next(row for row in audit['labels'] if row['product_id'] == pair[eye]) for eye in ('left','right')]
    params = rectification_parameters(left, right)
    output.mkdir(parents=True)
    cv2.setNumThreads(2)
    images, masks, maps = [], [], []
    for eye, model in zip(('left','right'), (left,right)):
        image = cv2.imread(model['decoded_image'])
        if image is None:
            raise ValueError('Missing decoded source frame')
        xy, mask = rectification_map(model, params, eye)
        rectified = cv2.remap(image, xy[...,0], xy[...,1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask = cv2.erode(mask.astype(np.uint8), np.ones((7,7), np.uint8)).astype(bool)
        rectified[~mask] = 0
        cv2.imwrite(str(output / (eye+'_rectified.png')), rectified)
        images.append(cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY))
        masks.append(mask)
        maps.append(xy)
    # Sparse image-based rectification QA, without fitting a corrective warp.
    sift = cv2.SIFT_create(nfeatures=1500)
    kp_l, ds_l = sift.detectAndCompute(images[0], masks[0].astype(np.uint8)*255)
    kp_r, ds_r = sift.detectAndCompute(images[1], masks[1].astype(np.uint8)*255)
    matches = [] if ds_l is None or ds_r is None else cv2.BFMatcher().knnMatch(ds_l, ds_r, k=2)
    accepted = [m for candidates in matches if len(candidates)==2
                for m,n in [candidates] if m.distance < 0.7*n.distance]
    vertical = np.array([abs(kp_l[m.queryIdx].pt[1]-kp_r[m.trainIdx].pt[1]) for m in accepted])
    qa = {'descriptor_matches_ratio_0_7': len(accepted), 'vertical_error_px_median': None,
          'vertical_error_px_p90': None, 'pointing_correction_applied': False}
    if len(vertical):
        qa.update(vertical_error_px_median=float(np.median(vertical)),
                  vertical_error_px_p90=float(np.percentile(vertical,90)))
    correction = None
    if refine_pointing:
        plausible = [m for m in accepted if abs(kp_l[m.queryIdx].pt[1]-kp_r[m.trainIdx].pt[1])<3]
        if len(plausible)<40:
            raise ValueError('Need at least 40 plausible matches for separate fit/holdout QA')
        rng = np.random.default_rng(42)
        rng.shuffle(plausible)
        count = len(plausible)//2
        source_pixels = []
        for map_xy, keypoints, attribute in zip(maps,(kp_l,kp_r),('queryIdx','trainIdx')):
            xy = np.array([keypoints[getattr(m,attribute)].pt for m in plausible],dtype=np.float32)
            mapped = cv2.remap(map_xy,xy[:,0,None],xy[:,1,None],cv2.INTER_LINEAR).reshape(-1,2)
            source_pixels.append(mapped)
        angle = fit_epipolar_pitch(left,right,source_pixels[0][:count],source_pixels[1][:count],params['basis'][0])
        original_right = right
        right = rotate_camera(right,params['basis'][0],angle)
        holdout_rect = [pixel_rays(model,pixels[count:]) @ params['basis'].T
                        for model,pixels in zip((left,right),source_pixels)]
        vertical = np.abs(params['focal_px']*(holdout_rect[0][:,1]/holdout_rect[0][:,2]
                                            -holdout_rect[1][:,1]/holdout_rect[1][:,2]))
        correction = {'method':'one_axis_epipolar_orientation_fit', 'axis_world':params['basis'][0].tolist(),
                      'angle_degrees':float(np.degrees(angle)), 'fit_matches':count,
                      'holdout_matches':len(plausible)-count, 'split_seed':42,
                      'prefilter_vertical_px':3, 'max_allowed_rotation_degrees':0.1,
                      'corrected_eye':'right', 'original_model':original_right,
                      'corrected_model':right, 'semantic_or_depth_targets_used':False}
        qa.update(pointing_correction_applied=True, vertical_error_px_median=float(np.median(vertical)),
                  vertical_error_px_p90=float(np.percentile(vertical,90)),
                  validation_source='held_out_initial_image_correspondences')
        # Re-render using the corrected physical model, not a cosmetic image shift.
        source = cv2.imread(right['decoded_image'])
        xy, mask = rectification_map(right,params,'right')
        mask = cv2.erode(mask.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
        rectified = cv2.remap(source,xy[...,0],xy[...,1],cv2.INTER_LINEAR)
        rectified[~mask] = 0
        cv2.imwrite(str(output/'right_rectified.png'),rectified)
        images[1], masks[1], maps[1] = cv2.cvtColor(rectified,cv2.COLOR_BGR2GRAY), mask, xy
    qa_pass = len(vertical)>=20 and np.median(vertical)<=1.0 and np.percentile(vertical,90)<=2.0
    min_d, number, block = -192, 448, 5
    settings = {'minDisparity': min_d, 'numDisparities': number, 'blockSize': block,
                'P1': 8*block**2, 'P2': 32*block**2, 'disp12MaxDiff': 1,
                'uniquenessRatio': 10, 'speckleWindowSize': 100, 'speckleRange': 2,
                'preFilterCap': 31, 'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY}
    disparity = cv2.StereoSGBM_create(**settings).compute(*images).astype(np.float32)/16
    reverse_settings = {**settings, 'minDisparity': -(min_d+number-1)}
    reverse = cv2.StereoSGBM_create(**reverse_settings).compute(images[1], images[0]).astype(np.float32)/16
    yy, xx = np.indices(disparity.shape)
    xr = np.rint(xx-disparity).astype(np.int32)
    inside = (xr>=0) & (xr<disparity.shape[1])
    xr_safe = np.clip(xr,0,disparity.shape[1]-1)
    reverse_sample = reverse[yy,xr_safe]
    consistency = np.abs(disparity+reverse_sample)
    xyz, axial = disparity_xyz(disparity, params)
    range_m = np.linalg.norm(xyz, axis=-1)
    valid = (masks[0] & inside & masks[1][yy,xr_safe] & (disparity>min_d-1)
             & (reverse_sample>reverse_settings['minDisparity']-1) & (consistency<=1)
             & np.isfinite(range_m) & (axial>0) & (range_m>=0.5) & (range_m<=30))
    # Verify projections of the triangulated points against matched source pixels.
    left_xy, left_forward = project_direction(left, xyz)
    right_xy, right_forward = project_direction(right, xyz+np.asarray(left['C'])-right['C'])
    reprojection = np.linalg.norm(left_xy-maps[0], axis=-1)
    reprojection_right = np.linalg.norm(right_xy-maps[1][yy,xr_safe], axis=-1)
    valid &= left_forward & right_forward & (reprojection<0.1) & (reprojection_right<3)
    # Reprojection is algebraic self-consistency, NOT independent depth accuracy.
    for name, array in (('disparity_px', disparity), ('left_right_error_px',consistency),
                        ('candidate_validity',valid), ('candidate_range_m',np.where(valid,range_m,np.nan).astype(np.float32)),
                        ('candidate_rectified_axial_depth_m',np.where(valid,axial,np.nan).astype(np.float32))):
        np.save(output / (name+'.npy'), array, allow_pickle=False)
    # Recover the right-reference view independently from reverse disparity.
    # Common rectification has unequal principal points; account for their offset.
    xl = np.rint(xx-reverse).astype(np.int32)
    inside_l = (xl>=0) & (xl<reverse.shape[1])
    xl_safe = np.clip(xl,0,reverse.shape[1]-1)
    effective_right = -reverse + params['cx_right']-params['cx_left']
    with np.errstate(divide='ignore',invalid='ignore'):
        z_right = params['focal_px']*params['baseline_m']/effective_right
    right_range = z_right*np.sqrt(1+((xx-params['cx_right'])/params['focal_px'])**2
                                  +((yy-params['cy'])/params['focal_px'])**2)
    right_valid = (masks[1] & inside_l & masks[0][yy,xl_safe]
                   & (reverse>reverse_settings['minDisparity']-1)
                   & (disparity[yy,xl_safe]>min_d-1)
                   & (np.abs(reverse+disparity[yy,xl_safe])<=1)
                   & (z_right>0) & np.isfinite(right_range) & (right_range>=0.5) & (right_range<=30))
    native_y, native_x = np.indices((right['axes']['Line'],right['axes']['Sample']),dtype=np.float64)
    rays = pixel_rays(right,np.stack((native_x,native_y),axis=-1))
    rect_rays = rays @ params['basis'].T
    sample_x = (params['focal_px']*rect_rays[...,0]/rect_rays[...,2]+params['cx_right']).astype(np.float32)
    sample_y = (params['focal_px']*rect_rays[...,1]/rect_rays[...,2]+params['cy']).astype(np.float32)
    native_valid = cv2.remap(right_valid.astype(np.uint8),sample_x,sample_y,cv2.INTER_NEAREST).astype(bool)
    native_valid &= rect_rays[...,2]>0
    native_range = cv2.remap(np.where(right_valid,right_range,0).astype(np.float32),sample_x,sample_y,cv2.INTER_NEAREST)
    native_range = np.where(native_valid,native_range,np.nan).astype(np.float32)
    native_axial = (native_range*(rays @ np.asarray(right['A']))).astype(np.float32)
    for name,array in (('right_source_range_m',native_range),('right_source_axial_depth_m',native_axial),
                       ('right_source_validity',native_valid)):
        np.save(output/(name+'.npy'),array,allow_pickle=False)
    cv2.imwrite(str(output/'right_source_validity.png'),native_valid.astype(np.uint8)*255)
    # The left reference is also exported on its native camera grid.
    ly, lx = np.indices((left['axes']['Line'], left['axes']['Sample']), dtype=np.float64)
    left_rays = pixel_rays(left, np.stack((lx, ly), axis=-1))
    left_rect_rays = left_rays @ params['basis'].T
    left_sx = (params['focal_px']*left_rect_rays[...,0]/left_rect_rays[...,2]+params['cx_left']).astype(np.float32)
    left_sy = (params['focal_px']*left_rect_rays[...,1]/left_rect_rays[...,2]+params['cy']).astype(np.float32)
    left_native_valid = cv2.remap(valid.astype(np.uint8), left_sx, left_sy, cv2.INTER_NEAREST).astype(bool)
    left_native_valid &= left_rect_rays[...,2] > 0
    left_native_range = cv2.remap(np.where(valid, range_m, 0).astype(np.float32), left_sx, left_sy, cv2.INTER_NEAREST)
    left_native_range = np.where(left_native_valid, left_native_range, np.nan).astype(np.float32)
    for name, array in (('left_source_range_m', left_native_range),
                        ('left_source_axial_depth_m', (left_native_range*(left_rays @ np.asarray(left['A']))).astype(np.float32)),
                        ('left_source_validity', left_native_valid)):
        np.save(output/(name+'.npy'), array, allow_pickle=False)
    cv2.imwrite(str(output/'candidate_validity.png'), valid.astype(np.uint8)*255)
    # Visualization has fixed 0.5-10m scaling, not per-image normalization.
    display = np.clip((range_m-0.5)/9.5*255,0,255).astype(np.uint8)
    display = cv2.applyColorMap(display, cv2.COLORMAP_TURBO)
    display[~valid] = 0
    cv2.imwrite(str(output/'candidate_range_preview.png'), display)
    report = {'status': 'technical_candidate_not_benchmark_depth' if qa_pass else 'rectification_qa_failed_not_admitted',
              'sol': sol, 'left_product': left['product_id'], 'right_product': right['product_id'],
              'source_image_hashes': [left['image_sha256'],right['image_sha256']],
              'audit_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest(),
              'geometry_code_sha256': hashlib.sha256((Path(__file__).parents[1]/'cahvor.py').read_bytes()).hexdigest(),
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'opencv': cv2.__version__, 'numpy': np.__version__,
              'rectification': {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in params.items()},
              'sgbm':settings, 'quality':qa, 'rectification_qa_pass':bool(qa_pass),
              'pointing_refinement':correction,
              'left_image_valid_fraction':float(masks[0].mean()), 'right_image_valid_fraction':float(masks[1].mean()),
              'candidate_valid_pixels':int(valid.sum()), 'candidate_canvas_valid_fraction':float(valid.mean()),
              'range_m_percentiles_5_50_95':np.percentile(range_m[valid],[5,50,95]).tolist() if valid.any() else [],
              'right_source_depth': {'shape':list(native_range.shape), 'reference_product':right['product_id'],
                                     'valid_fraction':float(native_valid.mean()),
                                     'valid_pixels':int(native_valid.sum()),
                                     'resampling':'nearest_neighbor_from_rectified_reference_no_hole_filling',
                                     'metric_accuracy':'not_independently_validated'},
              'left_source_depth': {'shape': list(left_native_range.shape), 'reference_product': left['product_id'],
                                    'valid_fraction': float(left_native_valid.mean()),
                                    'metric_accuracy': 'not_independently_validated'},
              'admitted_benchmark_depth_maps':0, 'independent_depth_gt_available':False,
              'mars_bench_pixel_registration':'pending_not_warped', 'ground_truth_masks_used':False,
              'hole_filling':False, 'GPU_used':False, 'elapsed_seconds':perf_counter()-started}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('status','sol','quality','right_source_depth',
                                                'admitted_benchmark_depth_maps','elapsed_seconds')},
                     indent=2,allow_nan=False))
    return output / 'report.json'
