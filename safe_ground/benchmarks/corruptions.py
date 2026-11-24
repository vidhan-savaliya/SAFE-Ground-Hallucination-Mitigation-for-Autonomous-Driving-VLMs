"""DriveBench 17 Input Settings & Corruption Transforms.

Implements corrupt(image, kind, severity) supporting clean, text-only (black image),
and 15 physical/sensor/transmission corruptions grouped into 5 real-world categories:
- Weather: Brightness, Dark, Snow, Fog, Rain
- External: Water Splash, Lens Obstacle
- Sensor: Camera Crash, Frame Lost, Saturate
- Motion: Motion Blur, Zoom Blur
- Transmission: Bit Error, Color Quantization, H.265 Compression
"""

import math
import numpy as np
import cv2
from PIL import Image as PILImage
from typing import Union, Dict, List, Tuple


# Group definitions for reporting
CORRUPTION_GROUPS: Dict[str, List[str]] = {
    "Weather": ["brightness", "dark", "snow", "fog", "rain"],
    "External": ["water_splash", "lens_obstacle"],
    "Sensor": ["camera_crash", "frame_lost", "saturate"],
    "Motion": ["motion_blur", "zoom_blur"],
    "Transmission": ["bit_error", "color_quantization", "h265_compression"]
}

ALL_15_CORRUPTIONS = [c for group in CORRUPTION_GROUPS.values() for c in group]
ALL_17_SETTINGS = ["clean", "text_only"] + ALL_15_CORRUPTIONS


def _ensure_numpy_uint8(image: Union[np.ndarray, PILImage.Image]) -> np.ndarray:
    """Converts input PIL image or array into uint8 RGB NumPy array."""
    if isinstance(image, PILImage.Image):
        img_arr = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        img_arr = image.copy()
        if img_arr.dtype == np.float32 or img_arr.dtype == np.float64:
            if img_arr.max() <= 1.0:
                img_arr = (img_arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img_arr = img_arr.clip(0, 255).astype(np.uint8)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    return img_arr


def corrupt_brightness(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Scales pixel intensity upwards."""
    c = [0.1, 0.2, 0.35, 0.5, 0.65][severity - 1]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + c), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def corrupt_dark(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Simulates low-light conditions using gamma adjustment and sensor shot noise."""
    gamma = [0.65, 0.5, 0.35, 0.25, 0.15][severity - 1]
    norm = img / 255.0
    darkened = np.power(norm, 1.0 / gamma) * 255.0
    noise = np.random.normal(0, 4.0 * severity, img.shape)
    return np.clip(darkened + noise, 0, 255).astype(np.uint8)


def corrupt_snow(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Generates synthetic falling snow layer with motion blur."""
    h, w, _ = img.shape
    snow_layer = np.random.normal(size=(h, w), loc=0.05 * severity, scale=0.3)
    snow_layer = np.clip(snow_layer, 0, 1)
    kernel_size = 2 * severity + 3
    kernel = np.diag(np.ones(kernel_size)) / kernel_size
    snow_layer = cv2.filter2D(snow_layer, -1, kernel)
    snow_3d = np.repeat(snow_layer[:, :, np.newaxis], 3, axis=2)
    blended = img.astype(np.float32) * (1 - 0.4 * snow_3d) + snow_3d * 240.0
    return np.clip(blended, 0, 255).astype(np.uint8)


def corrupt_fog(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Simulates atmospheric fog using airlight scattering."""
    c = [0.25, 0.40, 0.55, 0.70, 0.85][severity - 1]
    h, w, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] *= (1.0 - c * 0.7)  # Desaturate
    desat = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    fog_val = 220.0
    blended = desat * (1.0 - c) + fog_val * c
    return np.clip(blended, 0, 255).astype(np.uint8)


def corrupt_rain(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Adds directional rain streaks with optical motion blur."""
    h, w, _ = img.shape
    rain_layer = np.zeros((h, w), dtype=np.float32)
    num_drops = int(500 * severity)
    for _ in range(num_drops):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        length = np.random.randint(10, 25 + 5 * severity)
        cv2.line(rain_layer, (x, y), (x + int(length * 0.3), y + length), 1.0, 1)
    kernel = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float32) / 3.0
    rain_layer = cv2.filter2D(rain_layer, -1, kernel)
    rain_3d = np.repeat(rain_layer[:, :, np.newaxis], 3, axis=2)
    blended = img.astype(np.float32) + rain_3d * 90.0
    return np.clip(blended, 0, 255).astype(np.uint8)


def corrupt_water_splash(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Simulates transparent water droplet splatters with optical distortion."""
    h, w, _ = img.shape
    out = img.copy()
    num_splashes = 3 + 2 * severity
    for _ in range(num_splashes):
        cx = np.random.randint(int(w * 0.1), int(w * 0.9))
        cy = np.random.randint(int(h * 0.1), int(h * 0.9))
        radius = np.random.randint(20, 45 + 10 * severity)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        blurred_roi = cv2.GaussianBlur(out, (25, 25), 10)
        out = np.where(mask[:, :, np.newaxis] == 255, blurred_roi, out)
        cv2.circle(out, (cx, cy), radius, (240, 240, 240), 2)
    return out


def corrupt_lens_obstacle(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Creates lens obstruction (mud/debris) covering part of the camera view."""
    h, w, _ = img.shape
    out = img.copy().astype(np.float32)
    cx = np.random.randint(int(w * 0.2), int(w * 0.8))
    cy = np.random.randint(int(h * 0.2), int(h * 0.8))
    axes = (np.random.randint(40, 70 + 15 * severity), np.random.randint(30, 60 + 10 * severity))
    angle = np.random.randint(0, 180)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 15)
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    mud_color = np.array([45.0, 35.0, 25.0])
    out = out * (1.0 - mask_3d * 0.9) + mud_color * (mask_3d * 0.9)
    return np.clip(out, 0, 255).astype(np.uint8)


def corrupt_camera_crash(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Simulates complete camera failure by zeroing out the entire camera frame."""
    return np.zeros_like(img)


def corrupt_frame_lost(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Simulates intermittent frame loss or partial transmission dropout."""
    h, w, _ = img.shape
    out = img.copy()
    num_black_bars = severity * 3
    for _ in range(num_black_bars):
        y = np.random.randint(0, h - 20)
        h_bar = np.random.randint(8, 25)
        out[y:y + h_bar, :, :] = 0
    return out


def corrupt_saturate(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Enhances or blows out color saturation channel."""
    scale = [1.8, 2.4, 3.0, 3.8, 4.6][severity - 1]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def corrupt_motion_blur(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Simulates high-speed vehicle linear motion blur."""
    size = [7, 13, 19, 25, 33][severity - 1]
    kernel = np.zeros((size, size))
    kernel[int((size - 1) / 2), :] = np.ones(size)
    kernel = kernel / size
    return cv2.filter2D(img, -1, kernel)


def corrupt_zoom_blur(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Simulates radial zoom blur centered on the road horizon."""
    h, w, _ = img.shape
    out = img.astype(np.float32)
    cx, cy = w // 2, h // 2
    factors = [1.01, 1.025, 1.04, 1.06, 1.08][:severity]
    for factor in factors:
        M = cv2.getRotationMatrix2D((cx, cy), 0, factor)
        zoomed = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        out += zoomed.astype(np.float32)
    out /= (len(factors) + 1)
    return np.clip(out, 0, 255).astype(np.uint8)


def corrupt_bit_error(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Introduces digital bit-level inversion noise mimicking transmission corruption."""
    prob = [0.01, 0.02, 0.04, 0.07, 0.10][severity - 1]
    noise_mask = np.random.rand(*img.shape) < prob
    noisy_img = np.bitwise_xor(img, (noise_mask * 128).astype(np.uint8))
    return noisy_img


def corrupt_color_quantization(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Reduces the image color palette to low-bit quantization levels."""
    levels = [64, 32, 16, 8, 4][severity - 1]
    factor = 256 // levels
    quantized = (img // factor) * factor + factor // 2
    return quantized.astype(np.uint8)


def corrupt_h265_compression(img: np.ndarray, severity: int = 3) -> np.ndarray:
    """Simulates aggressive H.265 video compression macroblocking and artifacts."""
    qp = [30, 20, 12, 7, 3][severity - 1]  # JPEG quality proxy for video block artifacts
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), qp]
    _, enc = cv2.imencode('.jpg', img, encode_param)
    dec = cv2.imdecode(enc, 1)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Master corrupt() Dispatcher
# ---------------------------------------------------------------------------

CORRUPT_FUNCS = {
    "brightness": corrupt_brightness,
    "dark": corrupt_dark,
    "snow": corrupt_snow,
    "fog": corrupt_fog,
    "rain": corrupt_rain,
    "water_splash": corrupt_water_splash,
    "lens_obstacle": corrupt_lens_obstacle,
    "camera_crash": corrupt_camera_crash,
    "frame_lost": corrupt_frame_lost,
    "saturate": corrupt_saturate,
    "motion_blur": corrupt_motion_blur,
    "zoom_blur": corrupt_zoom_blur,
    "bit_error": corrupt_bit_error,
    "color_quantization": corrupt_color_quantization,
    "h265_compression": corrupt_h265_compression,
}


def corrupt(
    image: Union[np.ndarray, PILImage.Image],
    kind: str,
    severity: int = 3
) -> np.ndarray:
    """Corrupts an image according to DriveBench specifications.

    Parameters
    ----------
    image : np.ndarray or PIL.Image
        Input camera image.
    kind : str
        One of 'clean', 'text_only' / 'black_image', or one of the 15 corruption types.
    severity : int
        Corruption severity level from 1 to 5 (default: 3).

    Returns
    -------
    np.ndarray
        Corrupted RGB image as uint8 NumPy array.
    """
    img_arr = _ensure_numpy_uint8(image)
    kind_norm = kind.strip().lower().replace(" ", "_").replace("-", "_")

    if kind_norm in ("clean", "none"):
        return img_arr

    # Text-only prior check: fully black image
    if kind_norm in ("text_only", "black_image", "black", "t_o", "to"):
        return np.zeros_like(img_arr)

    # Alias normalization
    alias_map = {
        "lens": "lens_obstacle",
        "water": "water_splash",
        "splash": "water_splash",
        "crash": "camera_crash",
        "frame": "frame_lost",
        "motion": "motion_blur",
        "zoom": "zoom_blur",
        "quant": "color_quantization",
        "quantization": "color_quantization",
        "color_quant": "color_quantization",
        "h265": "h265_compression",
        "h.265": "h265_compression",
        "compress": "h265_compression",
        "compression": "h265_compression",
    }
    kind_norm = alias_map.get(kind_norm, kind_norm)

    func = CORRUPT_FUNCS.get(kind_norm)
    if func is not None:
        return func(img_arr, severity=max(1, min(5, severity)))
    else:
        raise ValueError(f"Unknown corruption kind: '{kind}'. Must be one of: {ALL_17_SETTINGS}")

