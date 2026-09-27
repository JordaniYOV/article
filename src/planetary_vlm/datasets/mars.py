"""Public Mars-Bench/MSL lifecycle; domain algorithms stay behind MarsData."""
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import zipfile
from .sources import read_json, save_json, download
import shutil
import subprocess

from .base import BaseData, PROJECT_ROOT


class MarsData(BaseData):
    domain = "mars"

    @property
    def samples(self):
        return self.interim / "mars_bench_msl_v1/test/samples.jsonl"

    def download(self, *, archive=False):
        from .sources import collect
        return collect(self.root, [], archive, domains=("mars",))

    def download_stereo_sources(self, stage="catalog", *, images=False):
        """Explicit bounded acquisition using Windows certificate validation."""
        if images and stage != "pilot":
            raise ValueError("images=True is only applicable to pilot acquisition")
        stages = {"catalog": "Catalog", "labels": "Labels", "aliases": "Aliases", "images": "Images"}
        backend = Path(__file__).parent / "_mars"
        if stage == "pilot":
            script, extra = backend / "acquire_pilot.ps1", (["-Images"] if images else [])
        elif stage in stages:
            script, extra = backend / "acquire.ps1", ["-Stage", stages[stage]]
        else:
            raise ValueError("Unknown Mastcam acquisition stage")
        powershell = shutil.which("pwsh")
        if powershell is None:
            raise ValueError("NASA acquisition currently requires PowerShell 7 (pwsh)")
        subprocess.run([powershell, "-NoProfile", "-File", str(script),
                        "-DataRoot", str(self.root), *extra], check=True)
        return self.raw / ("mastcam_stereo_pilot_v1" if stage == "pilot" else "mastcam_partner_search_v1")

    def extract(self):
        from ._mars.extraction import extract
        return extract(self.root)

    def inspect(self):
        from .inspection import inspect
        return inspect(self.root, domains=("mars",))

    def prepare(self, *, output, specification=None, samples=None, depth_index=None):
        from .preparation import build
        destination = self.new_version(output)
        source = Path(samples).resolve() if samples is not None else self.samples
        if not source.exists() and samples is None:
            source = self.extract()
        if specification is None:
            configured = PROJECT_ROOT / "configs/datasets/mars_bench_clean_v1.toml"
            specification = configured if configured.exists() else Path(__file__).parent / "_mars/semantic.toml"
        return build(source, specification, destination, domain=self.domain, depth_index=depth_index)

    def search_pairs(self, stage="audit"):
        from ._mars import partners
        if stage == "requests":
            return partners.requests(self.root)
        if stage == "audit":
            return partners.audit(self.root)
        raise ValueError("Pair search stage must be requests or audit")

    def inspect_label(self, label, *, output, decode_image=True):
        from ._mars.labels import inspect_label
        output = Path(output).resolve()
        if decode_image:
            output.mkdir(parents=True, exist_ok=True)
        return inspect_label(Path(label).resolve(), output, decode_image=decode_image)

    def audit_stereo_sources(self, *, output, source=None):
        from ._mars.labels import audit
        return audit(Path(source or self.raw / "mastcam_stereo_pilot_v1").resolve(),
                     self.new_version(output))

    def check_pair_geometry(self, left, right):
        from ._mars.partners import candidate, rays
        return candidate(left, right, {m["product_id"]: rays(m) for m in (left, right)})

    def reconstruct_depth(self, *, audit, output, sol, refine_pointing=False,
                          left_product=None, right_product=None):
        from ._mars.stereo import run
        return run(Path(audit).resolve(), self.new_version(output), str(sol).zfill(4),
                   refine_pointing=refine_pointing,
                   left_product=left_product, right_product=right_product)

    def check_pair(self, **options):
        """Image-level epipolar QA + stereo depth; not metric-depth admission."""
        return self.reconstruct_depth(**options)

    def select_pilot_pairs(self):
        from ._mars.pilot import select
        return select(self.root)

    def depth_batch(self, *, stage, output=None, plan=None, cache=None, resume=False,
                    min_coverage=.5, max_mae=8., limit=None, build_output=None):
        from ._mars import batch
        if stage == 'plan':
            if output is None:
                raise ValueError('Planning requires output')
            return batch.plan(self.root, self.new_version(output))
        if plan is None:
            raise ValueError('Acquisition/verification requires a plan')
        cache = cache or self.raw/'mastcam_depth_batch_v1'
        if stage == 'acquire':
            return batch.acquire(plan, cache)
        if stage == 'run':
            if output is None:
                raise ValueError('Verification requires output')
            report = batch.run(plan, cache, output, resume=resume, min_coverage=min_coverage,
                               max_mae=max_mae, limit=limit)
            if build_output is not None:
                summary = json.loads(report.read_text())
                if summary['unprocessed_jobs']:
                    raise ValueError('Batch is incomplete; resume after acquiring missing sources before building final dataset')
                return self.prepare(output=build_output, samples=summary['samples'],
                                    depth_index=summary['depth_index'])
            return report
        raise ValueError('Unknown depth batch stage')

    def verify_pilot_pairs(self, *, output=None):
        from ._mars.pilot import verify
        destination = self.new_version(output or self.evidence / "mastcam_pair_verification_v2")
        return verify(self.root, destination)

    def check_alignment(self, *, audit, output):
        from ._mars.alignment import check
        output = Path(output).resolve()
        if output.exists():
            raise FileExistsError("Alignment report already exists")
        output.parent.mkdir(parents=True, exist_ok=True)
        return check(audit, self.samples.parent, output)

    def compare_depth_images(self, *, output):
        from ._mars.comparison import main
        return main(self.root, self.new_version(output))

    def _collect_sources(self, mars_archive, report):
        root, evidence = self.root, self.evidence
        try:
            metadata = read_json("https://zenodo.org/api/records/15494736")
            save_json(evidence / "mars_bench_metadata.json", metadata)
            assets = []
            names = {"mapping.json", "partitions.zip"}
            if mars_archive:
                names.add("data.zip")
            for item in metadata["files"]:
                if item["key"] in names:
                    assets.append(download(item["links"]["self"], root / "raw" / "mars_bench_msl" / item["key"],
                                           item["size"], item["checksum"]))
            report["sources"]["mars_bench_msl"] = {"record_id": 15494736,
                "assets": assets, "provided_depth_maps": "not_established_by_record",
                "physical_traversal_outcomes": "not_provided_by_semantic_masks"}
            if mars_archive:
                with zipfile.ZipFile(root / "raw" / "mars_bench_msl" / "data.zip") as archive:
                    members = [{"path": item.filename, "size": item.file_size, "crc32": item.CRC}
                               for item in archive.infolist() if not item.is_dir()]
                save_json(evidence / "mars_bench_archive_inventory.json", members)
                report["sources"]["mars_bench_msl"]["archive_members"] = len(members)
                report["sources"]["mars_bench_msl"]["archive_extensions"] = dict(Counter(
                    PurePosixPath(item["path"]).suffix for item in members))
        except Exception as error:
            report["errors"].append({"source": "mars_bench_msl", "error": str(error)})
