"""Audit source geometry and decode a bounded pilot, without estimating depth."""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

def one(node, tag):
    found = node.findall('.//{*}' + tag)
    if len(found) != 1:
        raise ValueError(f'Expected one {tag}; found {len(found)}')
    return found[0]


def vector(model, tag):
    values = [float(child.text) for child in one(model, tag)]
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError('Camera vectors must have three finite components')
    return values


def inspect_label(path, destination, *, decode_image=True):
    root = ET.parse(path).getroot()
    if one(root, 'alternate_id').text != path.stem.upper():
        raise ValueError('Label identity differs from filename')
    camera = one(root, 'Camera_Model_Parameters')
    model = one(camera, 'CAHVOR_Model')
    frame = one(camera, 'Coordinate_Space_Indexed')
    axes = {one(axis, 'axis_name').text: int(one(axis, 'elements').text)
            for axis in one(root, 'Array_3D_Image').findall('{*}Axis_Array')}
    data = root.find('{*}File_Area_Observational')
    file_name = one(data.find('{*}File'), 'file_name').text
    result = {
        'product_id': path.stem.upper(), 'label_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'start_date_time': one(root, 'start_date_time').text,
        'camera_model_type': one(camera, 'model_type').text,
        'C': vector(model, 'Vector_Center'), 'A': vector(model, 'Vector_Axis'),
        'H': vector(model, 'Vector_Horizontal'), 'V': vector(model, 'Vector_Vertical'),
        'O': vector(model, 'Vector_Optical'), 'R': vector(model, 'Radial_Terms'),
        'frame': one(frame, 'coordinate_space_frame_type').text,
        'frame_indices': {one(index, 'index_id').text: one(index, 'index_value_number').text
                          for index in frame.findall('{*}Coordinate_Space_Index')},
        'frame_solution': one(frame, 'solution_id').text, 'axes': axes,
        'source_image': file_name, 'depth_reconstructed': False,
    }
    image_path = path.parent / file_name
    if decode_image and image_path.exists():
        array = one(root, 'Array_3D_Image')
        if one(array, 'data_type').text != 'UnsignedByte' or one(array, 'axis_index_order').text != 'Last Index Fastest':
            raise ValueError('Pilot decoder supports unsigned-byte BSQ only')
        axis_order = [one(axis, 'axis_name').text for axis in array.findall('{*}Axis_Array')]
        if axis_order != ['Band', 'Line', 'Sample'] or axes['Band'] != 3:
            raise ValueError('Unexpected array layout')
        content = image_path.read_bytes()
        expected_size = int(one(data.find('{*}File'), 'file_size').text)
        if len(content) != expected_size:
            raise ValueError('File size mismatch')
        offset = int(one(array, 'offset').text)
        count = axes['Band'] * axes['Line'] * axes['Sample']
        pixels = np.frombuffer(content, dtype=np.uint8, offset=offset, count=count)
        pixels = pixels.reshape(axes['Band'], axes['Line'], axes['Sample']).transpose(1, 2, 0)
        preview = destination / (path.stem + '.png')
        with preview.open('xb') as stream:
            Image.fromarray(pixels).save(stream)
        result.update(image_sha256=hashlib.sha256(content).hexdigest(), decoded_image=str(preview))
    return result


def audit(source, destination):
    if destination.exists():
        raise FileExistsError('Audit version exists; use a new output directory')
    destination.mkdir(parents=True)
    labels = [inspect_label(path, destination) for path in sorted(source.glob('*.xml'))]
    groups = {}
    for row in labels:
        sid = row['product_id']
        groups.setdefault((sid[:4], sid[6:12]), {})[sid[4:6]] = row
    pairs = []
    for (sol, sequence), eyes in groups.items():
        if set(eyes) != {'ML', 'MR'}:
            continue
        left, right = eyes['ML'], eyes['MR']
        frame_equal = all(left[key] == right[key] for key in ('frame', 'frame_indices', 'frame_solution'))
        dt = lambda row: datetime.fromisoformat(row['start_date_time'].replace('Z', '+00:00'))
        pair = {'sol': sol, 'sequence': sequence, 'left': left['product_id'], 'right': right['product_id'],
                'time_delta_seconds': abs((dt(left) - dt(right)).total_seconds()),
                'same_coordinate_frame': frame_equal,
                'status': 'candidate_requires_overlap_and_image_registration_checks',
                'depth_reconstructed': False}
        if frame_equal:
            pair['baseline_m'] = float(np.linalg.norm(np.asarray(left['C']) - right['C']))
            a, b = np.asarray(left['A']), np.asarray(right['A'])
            pair['optical_axis_separation_deg'] = math.degrees(math.acos(float(np.clip(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)), -1, 1))))
        pairs.append(pair)
    report = {'status': 'geometry_audit_not_depth_dataset', 'labels': labels, 'pairs': pairs,
              'verified_depth_maps': 0, 'mars_bench_alignment': 'pending',
              'reconstruction_quality': 'not_tested', 'source': 'https://planetarydata.jpl.nasa.gov/img/data/msl/msl_mmm/'}
    (destination / 'audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'pairs': pairs, 'decoded_images': sum('decoded_image' in row for row in labels)}, indent=2))
    return destination / 'audit.json'
