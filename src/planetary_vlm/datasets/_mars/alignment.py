"""Check explicit resize/crop hypotheses; never silently warp GT or depth."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from planetary_vlm.datasets.base import PROJECT_ROOT as PROJECT


def check(audit_path=None, samples_dir=None, output=None):
    audit_path = Path(audit_path) if audit_path is not None else PROJECT / 'data_rover/source_audit_v1/mastcam_stereo_pilot/audit.json'
    audit_dir = audit_path.parent
    audit = json.loads(audit_path.read_text())
    samples_dir = Path(samples_dir) if samples_dir is not None else PROJECT / 'data_rover/interim/mars_bench_msl_v1/test'
    samples = [json.loads(line) for line in (samples_dir / 'samples.jsonl').read_text().splitlines()]
    results = []
    for label in audit['labels']:
        record = next(row for row in samples if label['product_id'] in row['sample_id'])
        with Image.open(label['decoded_image']) as opened:
            source = opened.convert('RGB')
        with Image.open(samples_dir / record['image_path']) as opened:
            target = opened.convert('RGB')
        hypotheses = []
        for mode in (Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS):
            for strategy in ('stretch', 'resize_cover_center_crop'):
                if strategy == 'stretch':
                    size = target.size
                else:
                    scale = max(target.width/source.width, target.height/source.height)
                    size = (round(source.width*scale), round(source.height*scale))
                resized = source.resize(size, mode)
                x, y = (size[0]-target.width)//2, (size[1]-target.height)//2
                output = resized.crop((x,y,x+target.width,y+target.height))
                error = np.abs(np.asarray(output, dtype=np.float32)-np.asarray(target, dtype=np.float32))
                hypotheses.append({'strategy': strategy, 'resampling': mode.name,
                                   'resize_size': list(size), 'crop_box': [x,y,x+target.width,y+target.height],
                                   'mean_absolute_pixel_error_0_255': float(error.mean())})
        results.append({'product_id': label['product_id'], 'sample_id': record['sample_id'],
                        'source_size': list(source.size), 'benchmark_size': list(target.size),
                        'best_hypothesis': min(hypotheses,key=lambda row:row['mean_absolute_pixel_error_0_255']),
                        'hypotheses': hypotheses, 'status': 'diagnostic_not_a_verified_pixel_mapping'})
    report = {'status': 'alignment_hypotheses_require_qa', 'results': results,
              'masks_modified': False, 'depth_generated': False}
    output = Path(output) if output is not None else audit_dir / 'benchmark_alignment.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    for row in results:
        print(row['product_id'], row['best_hypothesis'])
    return output
