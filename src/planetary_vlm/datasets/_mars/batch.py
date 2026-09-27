"""Resumable stereo candidate processing; reconstructed depth is NOT depth GT."""
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from planetary_vlm.io import read_jsonl, write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_key(pair):
    return hashlib.sha256((pair['left']+'__'+pair['right']).encode()).hexdigest()[:24]


def checkpoint(path, value, *, jsonl=False):
    """Atomic replacement only for resumable batch-owned progress files."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
        if jsonl:
            for row in value:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
        else:
            json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def plan(data_root, output):
    """Use all available top-three hypotheses, never compression-ID aliases."""
    root, output = Path(data_root).resolve(), Path(output).resolve()
    search = root/'source_audit_v1/mastcam_partner_search'
    anchors = json.loads((search/'anchor_candidates.json').read_text())
    models = {m['product_id']: m for m in json.loads((search/'source_camera_models.json').read_text())}
    jobs, images = [], {}
    for anchor in anchors:
        for rank, pair in enumerate(anchor['best_candidates']):
            if any(models[pair[eye]]['axes']['Band'] != 3 for eye in ('left', 'right')):
                continue
            if any(models[pair[eye]]['axes']['Sample'] > 1600 or models[pair[eye]]['axes']['Line'] > 1200 for eye in ('left', 'right')):
                continue
            jobs.append({'sample_id': anchor['sample_id'], 'eye': 'left' if anchor['eye']=='ML' else 'right',
                         'anchor_product': anchor['archive_product_id'], 'rank': rank, 'pair': pair})
            for eye in ('left', 'right'):
                model = models[pair[eye]]
                if model['product_id'] in images:
                    continue
                label = root/'raw/mastcam_partner_search_v1/labels'/(model['product_id'].lower()+'.xml')
                xml = ET.parse(label).getroot()
                file = xml.find('{*}File_Area_Observational/{*}File')
                size = int(file.find('{*}file_size').text)
                images[model['product_id']] = {'product_id': model['product_id'], 'label': str(label),
                    'label_sha256': digest(label), 'image_name': model['source_image'],
                    'image_url': model['image_url'], 'image_bytes': size}
    output.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'mars-stereo-batch-v1', 'jobs': jobs, 'images': list(images.values()),
              'source_hashes': {name: digest(search/name) for name in ('anchor_candidates.json','source_camera_models.json')},
              'samples': str(root/'interim/mars_bench_msl_v1/test/samples.jsonl'),
              'source_samples_sha256': digest(root/'interim/mars_bench_msl_v1/test/samples.jsonl'),
              'policy': 'top_three_per_exact_archive_anchor_no_identity_aliases',
              'planned_photographs': len({j['sample_id'] for j in jobs}),
              'planned_pairs': len({pair_key(j['pair']) for j in jobs}),
              'planned_images': len(images), 'planned_image_bytes': sum(i['image_bytes'] for i in images.values()),
              'planner_sha256': digest(__file__)}
    write_json(output/'plan.json', result)
    return output/'plan.json'


def acquire(plan_path, cache):
    powershell = shutil.which('pwsh')
    if not powershell:
        raise ValueError('PDS acquisition requires PowerShell 7')
    subprocess.run([powershell, '-NoProfile', '-File', str(Path(__file__).with_name('acquire_batch.ps1')),
                    '-Plan', str(Path(plan_path).resolve()), '-Cache', str(Path(cache).resolve())], check=True)
    return Path(cache)/'acquisition.json'


def registration(source_path, target_path, max_mae):
    """Only resize/center-crop mappings; held image matches must be near identity."""
    import cv2
    import numpy as np
    from PIL import Image
    with Image.open(source_path) as im:
        source = im.convert('RGB')
    with Image.open(target_path) as im:
        target = im.convert('RGB')
    hypotheses = []
    for strategy in ('stretch', 'resize_cover_center_crop'):
        scale = max(target.width/source.width, target.height/source.height)
        size = target.size if strategy=='stretch' else (round(source.width*scale), round(source.height*scale))
        x, y = (size[0]-target.width)//2, (size[1]-target.height)//2
        box = (x,y,x+target.width,y+target.height)
        for method in (Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS):
            rendered = np.asarray(source.resize(size, method).crop(box))
            mae = float(np.abs(rendered.astype(float)-np.asarray(target).astype(float)).mean())
            hypotheses.append((mae, size, box, strategy, method.name, rendered))
    mae, size, box, strategy, method, rendered = min(hypotheses, key=lambda h: h[0])
    sift = cv2.SIFT_create(nfeatures=1800)
    a, da = sift.detectAndCompute(cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY), None)
    b, db = sift.detectAndCompute(cv2.cvtColor(np.asarray(target), cv2.COLOR_RGB2GRAY), None)
    matches = [] if da is None or db is None else cv2.BFMatcher().knnMatch(da, db, k=2)
    accepted = [m for row in matches if len(row)==2 for m,n in [row] if m.distance < .7*n.distance]
    errors = [float(np.linalg.norm(np.asarray(a[m.queryIdx].pt)-b[m.trainIdx].pt)) for m in accepted]
    median = float(np.median(errors)) if errors else None
    p90 = float(np.percentile(errors,90)) if errors else None
    passed = mae <= max_mae and len(errors)>=20 and median<=1 and p90<=2
    return {'pass': bool(passed), 'strategy': strategy, 'resampling': method, 'resize_size': list(size),
            'crop_box': list(box), 'mae_0_255': mae, 'matches': len(errors), 'median_error_px': median,
            'p90_error_px': p90, 'source_sha256': digest(source_path), 'benchmark_sha256': digest(target_path),
            'method': 'explicit_resize_crop_plus_identity_feature_match_QA_no_fitted_warp'}


def aligned_depth(depth_path, validity_path, mapping):
    import numpy as np
    from PIL import Image
    depth, valid = np.load(depth_path, allow_pickle=False), np.load(validity_path, allow_pickle=False)
    if depth.shape != valid.shape or depth.ndim != 2:
        raise ValueError('Depth/validity dimensions differ')
    resize = lambda a: np.asarray(Image.fromarray(a).resize(tuple(mapping['resize_size']), Image.Resampling.NEAREST).crop(tuple(mapping['crop_box'])))
    depth, valid = resize(depth), resize(valid.astype('uint8')).astype(bool)
    valid &= np.isfinite(depth) & (depth>0)
    return np.where(valid, depth, np.nan).astype('float32'), valid


def run(plan_path, cache, output, *, resume=False, min_coverage=.5, max_mae=8., limit=None):
    import cv2
    import numpy as np
    from PIL import Image
    from .labels import inspect_label
    from .stereo import run as reconstruct
    if not 0 < min_coverage <= 1 or not 0 < max_mae <= 255 or (limit is not None and limit < 1):
        raise ValueError('Invalid batch QA threshold/limit')
    plan_path, cache, output = map(lambda p: Path(p).resolve(), (plan_path, cache, output))
    data = json.loads(plan_path.read_text())
    source = Path(data['samples'])
    if digest(source) != data['source_samples_sha256']:
        raise ValueError('Source samples changed; rebuild plan')
    contract = {'plan_sha256': digest(plan_path), 'min_coverage': min_coverage, 'max_mae': max_mae,
                'batch_code_sha256': digest(__file__), 'stereo_code_sha256': digest(Path(__file__).with_name('stereo.py')),
                'geometry_code_sha256': digest(Path(__file__).parents[1]/'cahvor.py'),
                'label_decoder_sha256': digest(Path(__file__).with_name('labels.py'))}
    if output.exists():
        if not resume or json.loads((output/'contract.json').read_text()) != contract:
            raise ValueError('Existing batch requires --resume and identical plan/code/thresholds')
    else:
        output.mkdir(parents=True)
        write_json(output/'contract.json', contract)
    for folder in ('decoded', 'pairs', 'depth'):
        (output/folder).mkdir(exist_ok=True)
    records = read_jsonl(output/'results.jsonl') if (output/'results.jsonl').exists() else []
    done = {(r['sample_id'], r['pair_key']) for r in records if r['status'] != 'missing_source'}
    admitted = {r['sample_id'] for r in records if r['status']=='technical_depth_qa_pass'}
    samples = {r['sample_id']: r for r in read_jsonl(source)}
    image_records = {r['product_id']: r for r in data['images']}
    processed = 0
    for job in data['jobs']:
        key = pair_key(job['pair'])
        if job['sample_id'] in admitted or (job['sample_id'],key) in done:
            continue
        if limit is not None and processed >= limit:
            break
        processed += 1
        row = {'sample_id': job['sample_id'], 'pair_key': key, 'pair': job['pair'], 'eye': job['eye']}
        try:
            models = []
            for eye in ('left','right'):
                product = job['pair'][eye]
                rec = image_records[product]
                label = Path(rec['label'])
                if digest(label) != rec['label_sha256']:
                    raise ValueError('Source label changed')
                image = cache/rec['image_name']
                if not image.exists():
                    raise FileNotFoundError(str(image))
                copied_label = cache/label.name
                if not copied_label.exists():
                    shutil.copyfile(label,copied_label)
                if digest(copied_label) != rec['label_sha256']:
                    raise ValueError('Cached label hash differs')
                model_path = output/'decoded'/(product+'.json')
                if model_path.exists():
                    model = json.loads(model_path.read_text())
                    if model['image_sha256'] != digest(image):
                        raise ValueError('Cached image changed')
                else:
                    model = inspect_label(copied_label, output/'decoded')
                    write_json(model_path, model)
                models.append(model)
            sample = samples[job['sample_id']]
            rgb_path = (source.parent/sample['image_path']).resolve()
            model = models[0 if job['eye']=='left' else 1]
            if model['product_id'] != job['anchor_product']:
                raise ValueError('Depth reference differs from annotated anchor')
            mapping = registration(model['decoded_image'], rgb_path, max_mae)
            row['registration'] = mapping
            if not mapping['pass']:
                row['status'] = 'benchmark_registration_failed'
            else:
                pair_dir = output/'pairs'/key
                report_path = pair_dir/'report.json'
                if not report_path.exists():
                    if pair_dir.exists():
                        raise ValueError('Incomplete pair output retained; inspect before retry')
                    audit = output/'pairs'/(key+'.audit.json')
                    write_json(audit, {'labels': models, 'pairs': [job['pair']]})
                    with redirect_stdout(io.StringIO()):
                        reconstruct(audit, pair_dir, job['pair']['sol'], refine_pointing=True,
                                    left_product=job['pair']['left'], right_product=job['pair']['right'])
                report = json.loads(report_path.read_text())
                row['stereo_report'] = str(report_path)
                if not report['rectification_qa_pass']:
                    row['status'] = 'epipolar_qa_failed'
                else:
                    eye = job['eye']
                    depth, valid = aligned_depth(pair_dir/(eye+'_source_range_m.npy'), pair_dir/(eye+'_source_validity.npy'), mapping)
                    row['valid_fraction'] = float(valid.mean())
                    mask_path = (source.parent/sample['mask_path']).resolve()
                    with Image.open(mask_path) as mask:
                        if mask.size != (depth.shape[1],depth.shape[0]):
                            raise ValueError('Semantic/depth alignment dimensions differ')
                    if row['valid_fraction'] < min_coverage:
                        row['status'] = 'insufficient_depth_coverage'
                    else:
                        token = hashlib.sha256(job['sample_id'].encode()).hexdigest()
                        dp, vp = output/'depth'/(token+'.range_m.npy'), output/'depth'/(token+'.validity.npy')
                        np.save(dp, depth, allow_pickle=False)
                        np.save(vp, valid, allow_pickle=False)
                        row.update(status='technical_depth_qa_pass', image_path=str(rgb_path), mask_path=str(mask_path),
                            depth_path=str(dp), validity_path=str(vp), depth_sha256=digest(dp), validity_sha256=digest(vp),
                            group_id=sample['group_id'], split=sample['split'], license=sample['license'], source_url=sample['source_url'],
                            depth_provenance='stereo_reconstructed', depth_quantity='radial_camera_range_m',
                            independent_metric_validation=False)
                        admitted.add(job['sample_id'])
        except FileNotFoundError as exc:
            row.update(status='missing_source', error=str(exc))
        except (ValueError, OSError, KeyError, RuntimeError, cv2.error) as exc:
            row.update(status='processing_failed', error=str(exc))
        records = [r for r in records if (r['sample_id'],r['pair_key']) != (job['sample_id'],key)] + [row]
        checkpoint(output/'results.jsonl', records, jsonl=True)
        print(json.dumps({'processed_this_run':processed, 'sample':row['sample_id'], 'status':row['status'], 'qualified_photographs':len(admitted)}), flush=True)
    selected = [r for r in records if r['status']=='technical_depth_qa_pass']
    checkpoint(output/'depth_index.jsonl', selected, jsonl=True)
    checkpoint(output/'samples.jsonl', [dict(samples[r['sample_id']], image_path=r['image_path'], mask_path=r['mask_path']) for r in selected], jsonl=True)
    summary = {'status':'stereo_reconstructed_subset_technical_QA_not_independent_depth_GT',
               'planned_photographs':data['planned_photographs'], 'planned_pairs':data['planned_pairs'],
               'attempted_jobs':len(records), 'qualified_photographs':len(selected),
               'status_counts':dict(Counter(r['status'] for r in records)),
               'unprocessed_jobs':sum(j['sample_id'] not in admitted and (j['sample_id'],pair_key(j['pair'])) not in {(r['sample_id'],r['pair_key']) for r in records if r['status']!='missing_source'} for j in data['jobs']),
               'every_selected_photo_has_depth_and_validity':True, 'hole_filling':False,
               'independent_depth_gt_available':False, 'contract':contract,
               'depth_index':str(output/'depth_index.jsonl'), 'samples':str(output/'samples.jsonl')}
    checkpoint(output/'summary.json', summary)
    return output/'summary.json'
