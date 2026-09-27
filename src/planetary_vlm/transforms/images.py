"""Architecture-independent image operations with explicit parameters."""

import random
from pathlib import Path


def apply_image_transform(source: str | Path, destination: str | Path,
                          name: str, parameters: dict, seed: int) -> None:
    from PIL import Image, ImageEnhance, ImageFilter

    def positive(key):
        value = parameters.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{key} must be a nonnegative number")
        return float(value)

    with Image.open(source) as opened:
        image = opened.convert("RGB")
    if name == "blur":
        image = image.filter(ImageFilter.GaussianBlur(positive("radius")))
    elif name == "low_light":
        factor = positive("factor")
        if factor > 1:
            raise ValueError("low_light.factor must lie in [0, 1]")
        image = ImageEnhance.Brightness(image).enhance(factor)
    elif name == "contrast":
        image = ImageEnhance.Contrast(image).enhance(positive("factor"))
    elif name == "noise":
        sigma = positive("sigma")
        generator = random.Random(seed)
        pixels = [tuple(max(0, min(255, round(channel + generator.gauss(0, sigma))))
                        for channel in pixel)
                  for pixel in getattr(image, "get_flattened_data", image.getdata)()]
        image.putdata(pixels)
    elif name == "occlusion":
        rectangle = parameters.get("rectangle")
        if (not isinstance(rectangle, list) or len(rectangle) != 4
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not 0 <= value <= 1 for value in rectangle)):
            raise ValueError("occlusion.rectangle needs four normalized coordinates")
        left, top, right, bottom = rectangle
        if left >= right or top >= bottom:
            raise ValueError("occlusion rectangle must have positive area")
        box = (int(left * image.width), int(top * image.height),
               int(right * image.width), int(bottom * image.height))
        image.paste((0, 0, 0), box)
    elif name != "clean":
        raise ValueError(f"Unknown image transform: {name}")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        image.save(stream, format="PNG")


def render_depth(source: str | Path, destination: str | Path, *,
                 minimum: float, maximum: float, provenance: str) -> None:
    """Render one scalar NPY map using a fixed protocol range, never as new GT."""
    import numpy as np
    from PIL import Image

    if provenance not in {"measured", "synthetic", "estimated"}:
        raise ValueError("Declare depth provenance: measured, synthetic or estimated")
    if not minimum < maximum:
        raise ValueError("Depth rendering requires minimum < maximum")
    depth = np.load(source, allow_pickle=False)
    if depth.ndim != 2:
        raise ValueError("Depth input must be a 2D NPY array")
    valid = np.isfinite(depth)
    normalized = np.where(valid, np.clip((depth - minimum) / (maximum - minimum), 0, 1), 0)
    # Reserve zero for missing data; a fixed grayscale scale is transparent to audit.
    pixels = np.where(valid, 1 + np.rint(normalized * 254), 0).astype(np.uint8)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        Image.fromarray(pixels).save(stream, format="PNG")


def compose_images(sources: tuple[str, ...], destination: str | Path) -> None:
    """Left-to-right composition; matching dimensions are required, never inferred."""
    from PIL import Image

    if not sources:
        raise ValueError("Composite requires images")
    images = []
    for source in sources:
        with Image.open(source) as opened:
            images.append(opened.convert("RGB"))
    if len({image.size for image in images}) != 1:
        raise ValueError("Composite images must have identical dimensions")
    width, height = images[0].size
    combined = Image.new("RGB", (width * len(images), height))
    for index, image in enumerate(images):
        combined.paste(image, (width * index, 0))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        combined.save(stream, format="PNG")
