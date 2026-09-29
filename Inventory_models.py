"""
Inventory models

Recursively inventories SafeTensors/GGUF/checkpoint files and writes a single
self-contained HTML report (opens directly in any browser): a sortable,
groupable, live-searchable table with clickable links straight to each model
file. SafeTensors inspection reads only the header plus tiny `.comfy_quant`
metadata blobs; full model weights are not loaded into RAM.

Examples:
  python model_inventory.py "E:\\SD\\checkpoints"
  python model_inventory.py "E:\\SD\\checkpoints" "E:\\SD\\Lora" --output models.html
  python model_inventory.py "D:\\AI\\Models" --output models.html --json models.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any


SUPPORTED_SUFFIXES = {".safetensors", ".gguf", ".ckpt", ".pt", ".pth"}

LORA_DIR_WORDS = {"lora", "loras", "lycoris", "adapters"}
TEXT_ENCODER_DIR_WORDS = {
    "text_encoder", "text_encoders", "clip", "clip_text", "textencoder"
}
VAE_DIR_WORDS = {"vae", "vaes"}

FAMILY_PATTERNS = [
    ("Krea2", [
        (re.compile(r"\bkrea(?:2|-2)?\b", re.I), 3),
        (re.compile(r"txtfusion", re.I), 3),
        (re.compile(r"single.?stream", re.I), 2),
    ]),
    ("Anima", [
        (re.compile(r"\banima\b", re.I), 4),
        (re.compile(r"llm_adapter", re.I), 4),
        (re.compile(r"cosmos", re.I), 3),
    ]),
    ("Qwen-Image", [
        (re.compile(r"qwen[-_ ]image", re.I), 4),
        (re.compile(r"time_text_embed", re.I), 4),
        (re.compile(r"img_mod\.1", re.I), 4),
    ]),
    ("Z-Image / ZIT", [
        (re.compile(r"\bz[-_ ]?image\b", re.I), 4),
        (re.compile(r"\bzit\b", re.I), 4),
        (re.compile(r"noise_refiner", re.I), 3),
        (re.compile(r"context_refiner", re.I), 3),
        (re.compile(r"cap_embedder", re.I), 3),
        (re.compile(r"adaLN_modulation", re.I), 3),
    ]),
    ("Flux", [
        (re.compile(r"\bflux(?:\.1)?\b", re.I), 4),
        (re.compile(r"double_stream_modulation", re.I), 3),
        (re.compile(r"single_stream_modulation", re.I), 3),
        (re.compile(r"txt_attn", re.I), 3),
    ]),
    ("Wan", [
        (re.compile(r"\bwan(?:[-_ ]?2\.\d+)?\b", re.I), 4),
    ]),
    ("Pony", [
        (re.compile(r"\bpony(?:xl)?\b", re.I), 5),
        (re.compile(r"pony[-_ ]diffusion", re.I), 5),
    ]),
    ("Illustrious", [
        (re.compile(r"\billustrious(?:[-_ ]?xl)?\b", re.I), 5),
    ]),
    ("SDXL", [
        (re.compile(r"\bsdxl\b", re.I), 4),
        (re.compile(r"input_blocks\.", re.I), 2),
        (re.compile(r"middle_block\.", re.I), 2),
        (re.compile(r"output_blocks\.", re.I), 2),
        (re.compile(r"label_emb", re.I), 2),
    ]),
]

QUANT_PATTERNS = [
    ("INT4 ConvRot", re.compile(r"(?:int4[_-]convrot|w4a4[_-]convrot)", re.I)),
    ("W4A8 ConvRot", re.compile(r"(?:w4a8[_-]convrot|asym[_-]w4a8[_-]int8)", re.I)),
    ("INT8 ConvRot", re.compile(r"int8[_-]convrot", re.I)),
    ("NVFP4", re.compile(r"nvfp4", re.I)),
    ("MXFP8", re.compile(r"mxfp8", re.I)),
    ("FP8", re.compile(r"fp8", re.I)),
    ("INT8", re.compile(r"int8", re.I)),
    ("INT4", re.compile(r"int4", re.I)),
    ("NF4", re.compile(r"nf4", re.I)),
    ("GGUF quant", re.compile(r"q\d(?:_[a-z0-9]+)+", re.I)),
]

# Bumped whenever ModelInfo's field set changes; guards against loading a
# cache written by an older/incompatible version of this script.
CACHE_SCHEMA_VERSION = 4

# ggml tensor type enum -> display name (llama.cpp / gguf spec). Unknown
# values fall back to "GGML_TYPE_<n>" rather than failing.
GGUF_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
}

# Fixed-width GGUF metadata value types -> struct format. Strings (8) and
# arrays (9) are handled separately since they're variable-length.
_GGUF_SCALAR_FMT = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d",
}


@dataclass
class ModelInfo:
    path: str
    full_path: str
    name: str
    extension: str
    role: str
    family: str
    family_confidence: str
    family_detection: str
    quantization: str
    quantization_source: str
    dtype_summary: str
    tensor_count: int | None
    file_size_bytes: int
    payload_bytes: int | None
    param_count: int | None
    quant_layers: int | None
    special_formats: list[str]
    metadata: dict[str, str]
    warnings: list[str]
    quick_hash: str = ""
    duplicate_paths: list[str] = field(default_factory=list)
    lora_keywords: list[str] = field(default_factory=list)
    lora_details: dict[str, str] = field(default_factory=dict)


def human_bytes(value: int | None) -> str:
    if value is None:
        return "—"
    n = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TB"


def human_params(value: int | None) -> str:
    if not value:
        return "—"
    n = float(value)
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000.0 or unit == "T":
            return f"{n:.1f}{unit}" if unit else f"{int(n)}"
        n /= 1000.0
    return f"{n:.1f}T"


def quick_hash(path: Path, size: int, sample: int = 1_048_576) -> str:
    """Cheap fingerprint for duplicate detection: hashes the size plus the
    first and last `sample` bytes, so multi-GB checkpoints don't need to be
    read in full. Good enough to flag likely duplicates, not a guarantee."""
    h = hashlib.sha256()
    h.update(str(size).encode("utf-8"))
    try:
        with path.open("rb") as fh:
            h.update(fh.read(sample))
            if size > sample:
                fh.seek(max(size - sample, 0))
                h.update(fh.read(sample))
    except OSError:
        return ""
    return h.hexdigest()


def mark_duplicates(infos: list[ModelInfo]) -> None:
    """Groups files that share the same quick fingerprint and records each
    other's full paths in `duplicate_paths`. These are likely duplicates,
    not cryptographic proof of identical full contents. Mutates infos in place."""
    groups: dict[tuple[int, str], list[ModelInfo]] = defaultdict(list)
    for info in infos:
        if info.quick_hash:
            groups[(info.file_size_bytes, info.quick_hash)].append(info)
    for group in groups.values():
        if len(group) < 2:
            continue
        paths = [g.full_path for g in group]
        for g in group:
            g.duplicate_paths = [p for p in paths if p != g.full_path]


def parse_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError("file is too small for a SafeTensors header")
        header_len = struct.unpack("<Q", raw)[0]
        if header_len <= 0 or header_len > 100_000_000:
            raise ValueError(f"invalid SafeTensors header length: {header_len:,}")
        data = fh.read(header_len)
        if len(data) != header_len:
            raise ValueError("truncated SafeTensors header")
    header = json.loads(data.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("SafeTensors header is not an object")
    return header


def tensor_payload_bytes(entry: dict[str, Any]) -> int:
    offsets = entry.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        return 0
    try:
        return max(0, int(offsets[1]) - int(offsets[0]))
    except Exception:
        return 0


def tensor_numel(entry: dict[str, Any]) -> int:
    """Number of elements (= parameters) implied by a tensor's shape,
    independent of dtype/quantization packing."""
    shape = entry.get("shape")
    if not isinstance(shape, list) or not shape:
        return 0
    n = 1
    for dim in shape:
        try:
            n *= int(dim)
        except Exception:
            return 0
    return n


def read_tiny_tensor_payload(
    path: Path,
    entry: dict[str, Any],
    max_bytes: int = 4096,
) -> bytes | None:
    offsets = entry.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        return None
    try:
        begin, end = int(offsets[0]), int(offsets[1])
    except Exception:
        return None
    size = end - begin
    if begin < 0 or size < 0 or size > max_bytes:
        return None

    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            return None
        header_len = struct.unpack("<Q", raw)[0]
        fh.seek(8 + header_len + begin)
        return fh.read(size)


def normalize_quant_format(info: dict[str, Any]) -> str:
    fmt = str(info.get("format", "")).strip().lower()
    if fmt in {"convrot_w4a4", "w4a4_convrot"}:
        return "INT4 ConvRot"
    if fmt in {"asym_w4a8_int8", "w4a8_convrot"}:
        return "W4A8 ConvRot"
    if fmt == "int8" and info.get("convrot"):
        return "INT8 ConvRot"
    if fmt == "int8":
        return "INT8"
    if fmt == "nvfp4":
        return "NVFP4"
    if fmt == "mxfp8":
        return "MXFP8"
    if fmt in {"fp8", "fp8_scaled"}:
        return "FP8"
    return str(info.get("format", ""))


def embedded_quant_configs(
    path: Path,
    header: dict[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for key, entry in header.items():
        if (
            key == "__metadata__"
            or not key.endswith(".comfy_quant")
            or not isinstance(entry, dict)
        ):
            continue
        raw = read_tiny_tensor_payload(path, entry)
        if not raw:
            continue
        try:
            info = json.loads(raw.decode("utf-8"))
            if isinstance(info, dict):
                result.append(info)
        except Exception:
            pass
    return result


def infer_quantization(
    path: Path,
    header: dict[str, Any],
) -> tuple[str, str, int | None, list[str]]:
    counts = Counter()
    meta = header.get("__metadata__", {})

    if isinstance(meta, dict):
        raw = meta.get("_quantization_metadata")
        if raw:
            try:
                obj = json.loads(str(raw))
                layers = obj.get("layers", {})
                if isinstance(layers, dict):
                    for info in layers.values():
                        if isinstance(info, dict):
                            q = normalize_quant_format(info)
                            if q:
                                counts[q] += 1
            except Exception:
                pass

    for info in embedded_quant_configs(path, header):
        q = normalize_quant_format(info)
        if q:
            counts[q] += 1

    if counts:
        names = sorted(counts)
        return (
            counts.most_common(1)[0][0],
            "internal metadata",
            sum(counts.values()),
            names,
        )

    for name, pattern in QUANT_PATTERNS:
        if pattern.search(path.stem):
            return name, "filename", None, [name]

    return "None detected", "none", None, []


def dtype_summary(header: dict[str, Any]) -> tuple[str, int, int, int]:
    counts = Counter()
    tensors = 0
    payload = 0
    params = 0

    for key, entry in header.items():
        if key == "__metadata__" or not isinstance(entry, dict):
            continue
        counts[str(entry.get("dtype", "?"))] += 1
        tensors += 1
        payload += tensor_payload_bytes(entry)
        params += tensor_numel(entry)

    summary = ", ".join(
        f"{dtype}×{count}" for dtype, count in counts.most_common()
    )
    return summary or "—", tensors, payload, params


def selected_metadata(header: dict[str, Any]) -> dict[str, str]:
    """Generic, cross-file-type metadata for display: architecture/title
    style keys. LoRA training internals (ss_*) are intentionally excluded
    here — they're extracted separately by extract_lora_info() into
    structured fields instead of being dumped as a raw, sometimes
    truncated, key/value soup."""
    raw = header.get("__metadata__", {})
    if not isinstance(raw, dict):
        return {}

    out = {}
    for key, value in raw.items():
        key = str(key)
        if (
            key == "converted_by"
            or key.startswith("_quantization_metadata")
            or key.startswith("modelspec.")
            or key.startswith("general.")
        ):
            text = str(value)
            out[key] = text if len(text) <= 500 else text[:497] + "..."
    return out


LORA_TRIGGER_KEYS = ("modelspec.trigger_phrase", "trigger_phrase", "activation_text")
LORA_TAG_FREQUENCY_KEY = "ss_tag_frequency"
LORA_DETAIL_KEYS: dict[str, tuple[str, ...]] = {
    "Base model": ("ss_sd_model_name", "modelspec.base_model", "ss_base_model_version"),
    "Network module": ("ss_network_module",),
    "Network dim": ("ss_network_dim",),
    "Network alpha": ("ss_network_alpha",),
    "Training images": ("ss_num_train_images",),
    "Epochs": ("ss_num_epochs",),
    "Learning rate": ("ss_learning_rate", "ss_unet_lr", "ss_text_encoder_lr"),
    "Resolution": ("ss_resolution",),
    "Output name": ("ss_output_name",),
}
LORA_MAX_KEYWORDS = 25


def _first_present(raw: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = raw.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def extract_lora_info(header: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    """Pulls LoRA/LyCORIS training metadata (kohya-ss sd-scripts style)
    into a structured, display-ready shape: a deduplicated keyword/trigger
    word list (from an explicit trigger phrase plus the most frequent
    caption tags seen during training) and a small table of training
    facts (base model, network dim/alpha, epochs, etc). Works on the raw
    __metadata__ block, not the truncated copy selected_metadata() makes,
    since ss_tag_frequency alone can be tens of KB of JSON."""
    raw = header.get("__metadata__", {})
    if not isinstance(raw, dict):
        return [], {}

    keywords: list[str] = []

    trigger = _first_present(raw, LORA_TRIGGER_KEYS)
    if trigger:
        for piece in re.split(r"[,\n;]+", trigger):
            piece = piece.strip()
            if piece and piece not in keywords:
                keywords.append(piece)

    tag_freq_raw = raw.get(LORA_TAG_FREQUENCY_KEY)
    if isinstance(tag_freq_raw, str):
        try:
            tag_freq = json.loads(tag_freq_raw)
        except Exception:
            tag_freq = None
        if isinstance(tag_freq, dict):
            totals: Counter = Counter()
            for bucket in tag_freq.values():
                if isinstance(bucket, dict):
                    for tag, count in bucket.items():
                        try:
                            totals[str(tag).strip()] += int(count)
                        except Exception:
                            continue
            for tag, _count in totals.most_common(LORA_MAX_KEYWORDS):
                if tag and tag not in keywords:
                    keywords.append(tag)

    details: dict[str, str] = {}
    for label, keys in LORA_DETAIL_KEYS.items():
        value = _first_present(raw, keys)
        if value:
            details[label] = value if len(value) <= 300 else value[:297] + "..."

    return keywords, details


def detect_role(path: Path, keys: list[str]) -> tuple[str, str]:
    parts = {part.lower() for part in path.parts}

    if parts & LORA_DIR_WORDS:
        return "LoRA / adapter", "path"
    if parts & TEXT_ENCODER_DIR_WORDS:
        return "Text Encoder", "path"
    if parts & VAE_DIR_WORDS:
        return "VAE", "path"

    key_text = " ".join(keys[:3000]).lower()
    if any("lora" in key for key in keys[:3000]):
        return "LoRA / adapter", "tensor keys"
    if "first_stage_model" in key_text or "post_quant_conv" in key_text:
        return "VAE", "tensor keys"
    if (
        "text_model.embeddings" in key_text
        or "embed_tokens" in key_text
        or "shared.weight" in key_text
    ):
        return "Text Encoder", "tensor keys"
    if "diffusion_model" in key_text or "transformer_blocks" in key_text:
        return "Diffusion Model", "tensor keys"
    if path.suffix.lower() == ".gguf":
        return "GGUF model", "extension"
    if path.suffix.lower() in {".ckpt", ".pt", ".pth"}:
        return "Checkpoint / weights", "extension"
    return "Unknown", "none"


def detect_family(
    path: Path,
    keys: list[str],
    metadata: dict[str, str],
) -> tuple[str, str, str]:
    filename_blob = path.name
    key_blob = " ".join(keys[:5000])
    meta_blob = " ".join(metadata.values())
    scores = Counter()
    sources: dict[str, set[str]] = defaultdict(set)

    for family, rules in FAMILY_PATTERNS:
        for pattern, weight in rules:
            if pattern.search(filename_blob):
                scores[family] += weight
                sources[family].add("filename")
            if pattern.search(meta_blob):
                scores[family] += weight
                sources[family].add("metadata")
            if pattern.search(key_blob):
                scores[family] += weight
                sources[family].add("tensor keys")

    if not scores:
        return "Unknown", "UNKNOWN", "no matching family signature"

    family, score = scores.most_common(1)[0]
    confidence = "HIGH" if score >= 5 else "MEDIUM" if score >= 3 else "LOW"
    return family, confidence, ", ".join(sorted(sources[family]))


def inspect_safetensors(path: Path):
    header = parse_safetensors_header(path)
    keys = [str(k) for k in header if k != "__metadata__"]
    dtypes, tensor_count, payload, params = dtype_summary(header)
    metadata = selected_metadata(header)
    lora_keywords, lora_details = extract_lora_info(header)
    quant, qsource, qlayers, special = infer_quantization(path, header)
    return (
        keys,
        dtypes,
        tensor_count,
        payload,
        params,
        metadata,
        lora_keywords,
        lora_details,
        quant,
        qsource,
        qlayers,
        special,
    )


def _gguf_read_string(fh) -> str:
    (length,) = struct.unpack("<Q", fh.read(8))
    return fh.read(length).decode("utf-8", errors="replace")


def _gguf_read_scalar(fh, vtype: int):
    fmt = _GGUF_SCALAR_FMT[vtype]
    size = struct.calcsize(fmt)
    raw = fh.read(size)
    if len(raw) != size:
        raise ValueError("truncated GGUF value")
    value = struct.unpack(fmt, raw)[0]
    return bool(value) if vtype == 7 else value


def _gguf_skip_value(fh, vtype: int) -> None:
    """Advances past a metadata value without materializing it. Arrays
    (vocab/merges/scores lists can be 100k+ entries) are skipped without
    decoding, which keeps GGUF metadata parsing fast."""
    if vtype == 8:  # string
        (length,) = struct.unpack("<Q", fh.read(8))
        fh.seek(length, 1)
        return
    if vtype == 9:  # array
        elem_type, count = struct.unpack("<IQ", fh.read(12))
        if elem_type == 8:
            for _ in range(count):
                (length,) = struct.unpack("<Q", fh.read(8))
                fh.seek(length, 1)
        elif elem_type == 9:
            for _ in range(count):
                _gguf_skip_value(fh, elem_type)
        else:
            size = struct.calcsize(_GGUF_SCALAR_FMT[elem_type])
            fh.seek(size * count, 1)
        return
    fh.seek(struct.calcsize(_GGUF_SCALAR_FMT[vtype]), 1)


GGUF_WANTED_METADATA_KEYS = {
    "general.architecture",
    "general.name",
    "general.file_type",
    "general.quantization_version",
    "general.size_label",
    "general.base_model.count",
    "quantize.imatrix.file",
}


def inspect_gguf(path: Path):
    """Fully parses the GGUF header (metadata + tensor descriptors) without
    touching the tensor payload. Returns tensor_count, selected metadata,
    warnings, total parameter count and a dtype/quantization summary built
    from the actual per-tensor ggml types (falls back to filename regex
    only if the header can't be parsed)."""
    warnings: list[str] = []
    metadata: dict[str, str] = {}
    tensor_count = None
    param_count = 0
    type_counts: Counter = Counter()

    try:
        with path.open("rb") as fh:
            if fh.read(4) != b"GGUF":
                warnings.append("GGUF extension but invalid GGUF magic.")
                return None, metadata, warnings, None, Counter()

            (version,) = struct.unpack("<I", fh.read(4))
            (tensor_count,) = struct.unpack("<Q", fh.read(8))
            (kv_count,) = struct.unpack("<Q", fh.read(8))
            metadata["gguf.version"] = str(version)

            for _ in range(kv_count):
                key = _gguf_read_string(fh)
                (vtype,) = struct.unpack("<I", fh.read(4))
                if vtype in (8,) and key in GGUF_WANTED_METADATA_KEYS:
                    metadata[key] = _gguf_read_string(fh)
                elif vtype not in (8, 9) and key in GGUF_WANTED_METADATA_KEYS:
                    metadata[key] = str(_gguf_read_scalar(fh, vtype))
                else:
                    _gguf_skip_value(fh, vtype)

            for _ in range(tensor_count):
                _gguf_read_string(fh)  # tensor name, not needed here
                (n_dims,) = struct.unpack("<I", fh.read(4))
                dims = struct.unpack(f"<{n_dims}Q", fh.read(8 * n_dims))
                (ggml_type,) = struct.unpack("<I", fh.read(4))
                fh.read(8)  # offset, unused (we never read tensor payload)
                numel = 1
                for d in dims:
                    numel *= d
                param_count += numel
                type_counts[GGUF_TYPE_NAMES.get(ggml_type, f"GGML_TYPE_{ggml_type}")] += 1
    except Exception as exc:
        warnings.append(f"GGUF header parse failed: {exc}")
        return tensor_count, metadata, warnings, None, Counter()

    return tensor_count, metadata, warnings, param_count, type_counts


def gguf_quant_label(type_counts: Counter) -> tuple[str, str]:
    """Picks the dominant *quantized* ggml type as the headline
    quantization, ignoring the handful of F32/F16/BF16 tensors (norms,
    embeddings) that quantized GGUF files typically keep at full precision."""
    quant_only = Counter({k: v for k, v in type_counts.items() if k not in ("F32", "F16", "BF16")})
    if quant_only:
        return quant_only.most_common(1)[0][0], "tensor types"
    if type_counts:
        return type_counts.most_common(1)[0][0], "tensor types"
    return "None detected", "none"


def inspect(path: Path, root: Path) -> ModelInfo:
    ext = path.suffix.lower()
    keys = []
    metadata = {}
    warnings = []
    tensor_count = None
    payload_bytes = None
    param_count = None
    dtype = "�"
    quant_layers = None
    special = []
    lora_keywords: list[str] = []
    lora_details: dict[str, str] = {}

    if ext == ".safetensors":
        try:
            (
                keys,
                dtype,
                tensor_count,
                payload_bytes,
                param_count,
                metadata,
                lora_keywords,
                lora_details,
                quantization,
                qsource,
                quant_layers,
                special,
            ) = inspect_safetensors(path)
        except Exception as exc:
            quantization = "Unreadable"
            qsource = "error"
            warnings.append(f"SafeTensors inspection failed: {exc}")
    elif ext == ".gguf":
        tensor_count, gguf_meta, gguf_warnings, param_count, type_counts = inspect_gguf(path)
        metadata.update(gguf_meta)
        warnings.extend(gguf_warnings)
        if type_counts:
            dtype = ", ".join(
                f"{t}?{c}" for t, c in type_counts.most_common()
            )
            quantization, qsource = gguf_quant_label(type_counts)
        else:
            quantization = "None detected"
            qsource = "none"
        if quantization == "None detected":
            # header couldn't be parsed, or the file is all F32/F16 with no
            # informative type mix; fall back to the filename heuristic.
            for name, pattern in QUANT_PATTERNS:
                if pattern.search(path.stem):
                    quantization = name
                    qsource = "filename"
                    break
    else:
        quantization = "None detected"
        qsource = "none"

    role, _role_source = detect_role(path, keys)
    family, confidence, family_detection = detect_family(
        path, keys, {**metadata, **lora_details}
    )

    is_lora = role.startswith("LoRA")
    if not is_lora:
        # Keep training-recipe fields out of non-LoRA items entirely, so
        # the two detail popups never mix content.
        lora_keywords = []
        lora_details = {}

    rel = path.relative_to(root) if path.is_relative_to(root) else path

    try:
        full_path = str(path.resolve())
    except Exception:
        full_path = str(path)

    size = path.stat().st_size

    return ModelInfo(
        path=str(rel),
        full_path=full_path,
        name=path.stem,
        extension=ext,
        role=role,
        family=family,
        family_confidence=confidence,
        family_detection=family_detection,
        quantization=quantization,
        quantization_source=qsource,
        dtype_summary=dtype,
        tensor_count=tensor_count,
        file_size_bytes=size,
        payload_bytes=payload_bytes,
        param_count=param_count,
        quant_layers=quant_layers,
        special_formats=special,
        metadata=metadata,
        warnings=warnings,
        quick_hash=quick_hash(path, size),
        lora_keywords=lora_keywords,
        lora_details=lora_details,
    )


def inspect_or_placeholder(path: Path, root: Path) -> ModelInfo:
    """Runs inspect() and never raises: on failure, returns a placeholder
    ModelInfo carrying the error as a warning. Used by the thread pool so a
    single bad file can't abort the whole scan."""
    try:
        return inspect(path, root)
    except Exception as exc:
        try:
            size = path.stat().st_size if path.exists() else 0
        except Exception:
            size = 0
        try:
            full_path = str(path.resolve())
        except Exception:
            full_path = str(path)
        return ModelInfo(
            path=str(path),
            full_path=full_path,
            name=path.stem,
            extension=path.suffix.lower(),
            role="Unknown",
            family="Unknown",
            family_confidence="UNKNOWN",
            family_detection="inspection error",
            quantization="Unreadable",
            quantization_source="error",
            dtype_summary="�",
            tensor_count=None,
            file_size_bytes=size,
            payload_bytes=None,
            param_count=None,
            quant_layers=None,
            special_formats=[],
            metadata={},
            warnings=[f"Unexpected inspection error: {exc}"],
        )


def load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[!] Ignoring unreadable cache {cache_path}: {exc}")
        return {}
    if not isinstance(raw, dict) or raw.get("schema_version") != CACHE_SCHEMA_VERSION:
        # Cache from an older/incompatible version of this script; ignore
        # rather than risk feeding stale fields into ModelInfo(**...).
        return {}
    entries = raw.get("entries")
    return entries if isinstance(entries, dict) else {}


def save_cache(cache_path: Path, entries: dict[str, dict[str, Any]]) -> None:
    payload = {"schema_version": CACHE_SCHEMA_VERSION, "entries": entries}
    try:
        cache_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:
        print(f"[!] Could not write cache {cache_path}: {exc}")


def collect_files(
    roots: list[Path],
    suffixes: set[str],
) -> list[tuple[Path, Path]]:
    seen: dict[str, tuple[Path, Path]] = {}

    for root in roots:
        if not root.exists():
            print(f"[!] Root not found: {root}")
            continue
        if not root.is_dir():
            print(f"[!] Not a directory: {root}")
            continue

        for path in root.rglob("*"):
            try:
                if path.is_file() and path.suffix.lower() in suffixes:
                    seen.setdefault(
                        str(path.resolve()).lower(),
                        (path, root),
                    )
            except OSError as exc:
                print(f"[!] Cannot inspect {path}: {exc}")

    return sorted(seen.values(), key=lambda x: str(x[0]).lower())


def esc_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def file_uri(full_path: str) -> str:
    try:
        p = Path(full_path)
        if p.is_absolute():
            return p.as_uri()
    except Exception:
        pass
    return ""


def build_html(
    infos: list[ModelInfo],
    roots: list[Path],
) -> str:
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    total = sum(i.file_size_bytes for i in infos)

    by_family: dict[str, list[ModelInfo]] = defaultdict(list)
    for info in infos:
        by_family[info.family].append(info)

    family_items = sorted(
        by_family.items(),
        key=lambda pair: (
            -sum(i.file_size_bytes for i in pair[1]),
            pair[0].lower(),
        ),
    )
    family_rows = "\n".join(
        f"<tr><td>{esc_html(family)}</td><td class='num'>{len(items)}</td>"
        f"<td class='num'>{human_bytes(sum(i.file_size_bytes for i in items))}</td></tr>"
        for family, items in family_items
    )

    qcount = Counter(i.quantization for i in infos)
    quant_rows = "\n".join(
        f"<tr><td>{esc_html(q)}</td><td class='num'>{count}</td></tr>"
        for q, count in sorted(qcount.items(), key=lambda x: (-x[1], x[0].lower()))
    )

    roots_list = "\n".join(f"<li><code>{esc_html(str(r))}</code></li>" for r in roots)

    dup_groups: dict[str, list[ModelInfo]] = {}
    for info in infos:
        if info.duplicate_paths:
            group_key = "|".join(sorted([info.full_path] + info.duplicate_paths))
            dup_groups.setdefault(group_key, []).append(info)
    wasted_bytes = sum(
        max(0, len(group) - 1) * group[0].file_size_bytes
        for group in dup_groups.values()
    )

    records = []
    for item in infos:
        item_root = "Unknown root"
        try:
            item_path = Path(item.full_path).resolve()
            matching_roots = []
            for root in roots:
                try:
                    resolved_root = root.resolve()
                    if item_path.is_relative_to(resolved_root):
                        matching_roots.append(resolved_root)
                except OSError:
                    pass
            if matching_roots:
                # If roots overlap, use the most specific (deepest) root.
                item_root = str(max(matching_roots, key=lambda p: len(p.parts)))
        except OSError:
            pass

        records.append({
            "name": item.name,
            "ext": item.extension,
            "root": item_root,
            "role": item.role,
            "family": item.family,
            "confidence": item.family_confidence,
            "detection": item.family_detection or "none",
            "quant": item.quantization,
            "quantSource": item.quantization_source,
            "dtype": item.dtype_summary,
            "size": item.file_size_bytes,
            "sizeH": human_bytes(item.file_size_bytes),
            "tensors": item.tensor_count,
            "params": item.param_count,
            "paramsH": human_params(item.param_count),
            "payloadBytes": item.payload_bytes,
            "payloadH": human_bytes(item.payload_bytes),
            "quantLayers": item.quant_layers,
            "specialFormats": item.special_formats,
            "path": item.path,
            "fullPath": item.full_path,
            "link": file_uri(item.full_path),
            "folderPath": str(Path(item.full_path).parent),
            "folderLink": file_uri(str(Path(item.full_path).parent)),
            "quickHash": item.quick_hash,
            "metadata": item.metadata,
            "loraKeywords": item.lora_keywords,
            "loraDetails": item.lora_details,
            "warnings": item.warnings,
            "duplicates": [
                {"path": d, "link": file_uri(d)} for d in item.duplicate_paths
            ],
        })

    data_json = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")

    html = HTML_TEMPLATE
    html = html.replace("__GENERATED__", esc_html(generated))
    html = html.replace("__FILE_COUNT__", str(len(infos)))
    html = html.replace("__TOTAL_SIZE__", human_bytes(total))
    html = html.replace("__FAMILY_ROWS__", family_rows)
    html = html.replace("__QUANT_ROWS__", quant_rows)
    html = html.replace("__ROOTS_LIST__", roots_list)
    html = html.replace("__DUP_COUNT__", str(len(dup_groups)))
    html = html.replace("__DUP_WASTED__", human_bytes(wasted_bytes))
    html = html.replace("__DATA_JSON__", data_json)
    return html


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Model Zoo Inventory</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #0f1115; --panel: #171a21; --border: #2a2f3a; --text: #e6e8ec;
    --muted: #9aa3b2; --accent: #5aa9ff; --accent2: #ffcf5a;
    --row-hover: #1f232c; --mark: #ffcf5a; --mark-text: #1b1b1b;
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg: #f6f7f9; --panel: #ffffff; --border: #dde1e7; --text: #1b1e24;
      --muted: #5a6272; --accent: #1a66d6; --accent2: #a86a00;
      --row-hover: #eef2f8; --mark: #ffe27a; --mark-text: #1b1b1b;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px; background: var(--bg); color: var(--text);
    font: 14px/1.5 -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  }
  h1 { margin: 0 0 4px; font-size: 22px; }
  .meta { color: var(--muted); margin-bottom: 20px; font-size: 12.5px; }
  .panel {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px; margin-bottom: 18px;
  }
  .summary-grid { display: flex; flex-wrap: wrap; gap: 16px; }
  .summary-grid .panel { flex: 1 1 260px; }
  h2 { margin: 0 0 10px; font-size: 14px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
  table { width: 100%; border-collapse: collapse; }
  th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--border); }
  td.num, th.num { text-align: right; }
  .stat { font-size: 26px; font-weight: 600; }
  .controls { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin-bottom: 14px; }
  #search {
    flex: 1 1 260px; padding: 9px 12px; border-radius: 8px; border: 1px solid var(--border);
    background: var(--panel); color: var(--text); font-size: 14px;
  }
  select {
    padding: 8px 10px; border-radius: 8px; border: 1px solid var(--border);
    background: var(--panel); color: var(--text); font-size: 14px;
  }
  #clearBtn {
    padding: 8px 14px; border-radius: 8px; border: 1px solid var(--border);
    background: var(--panel); color: var(--text); cursor: pointer;
  }
  #clearBtn:hover { background: var(--row-hover); }
  #matchInfo { color: var(--muted); font-size: 12.5px; }
  #modelsTable thead th {
    position: sticky; top: 0; background: var(--panel); cursor: pointer; user-select: none;
    white-space: nowrap; border-bottom: 2px solid var(--border);
  }
  #modelsTable thead th.active { color: var(--accent); }
  #modelsTable thead th .arrow { opacity: .6; font-size: 11px; margin-left: 3px; }
  #modelsTable tbody tr:hover { background: var(--row-hover); }
  .group-row td {
    background: var(--panel); font-weight: 600; color: var(--accent);
    border-top: 1px solid var(--border); border-bottom: 1px solid var(--border);
    padding-top: 10px; padding-bottom: 10px;
  }
  .model-link { color: var(--accent); text-decoration: none; }
  .model-link:hover { text-decoration: underline; }
  .no-link { color: var(--text); }
  mark { background: var(--mark); color: var(--mark-text); padding: 0 1px; border-radius: 2px; }
  .badge {
    display: inline-block; padding: 1px 7px; border-radius: 100px; font-size: 11.5px;
    border: 1px solid var(--border); color: var(--muted);
  }
  .badge.HIGH { color: #3fcf6e; border-color: #3fcf6e55; }
  .badge.MEDIUM { color: var(--accent2); border-color: #ffcf5a55; }
  .badge.LOW, .badge.UNKNOWN { color: var(--muted); }
  details.diag summary { cursor: pointer; color: var(--muted); font-size: 12.5px; }
  details.diag { margin-top: 4px; }
  details.diag div { padding: 6px 0 2px 12px; color: var(--muted); font-size: 12.5px; }
  .path-cell { color: var(--muted); font-size: 12.5px; word-break: break-all; }
  footer { color: var(--muted); font-size: 12px; margin-top: 20px; }
  footer code { color: var(--text); }
  .root-tabs {
    display: flex; gap: 7px; flex-wrap: wrap; margin: 0 0 14px;
    padding-bottom: 2px; border-bottom: 1px solid var(--border);
  }
  .root-tab {
    padding: 7px 11px; border: 1px solid transparent; border-radius: 8px 8px 0 0;
    background: transparent; color: var(--muted); cursor: pointer; font: inherit;
    max-width: min(420px, 90vw); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .root-tab:hover { background: var(--row-hover); color: var(--text); }
  .root-tab.active {
    color: var(--accent); border-color: var(--border); border-bottom-color: var(--panel);
    background: var(--panel); margin-bottom: -1px;
  }
  .root-tab .tab-count { opacity: .7; margin-left: 5px; font-size: 12px; }
  .empty-state { padding: 30px; text-align: center; color: var(--muted); }
  .model-row { cursor: pointer; }
  .model-row:hover { background: var(--row-hover); }
  .details-direct-link { margin-left: 4px; color: var(--accent); text-decoration: none; font-weight: 600; }
  .details-direct-link:hover { text-decoration: underline; }
  .details-modal[hidden] { display: none; }
  .details-modal {
    position: fixed; inset: 0; z-index: 1000; display: flex;
    align-items: center; justify-content: center; padding: 24px;
  }
  .details-backdrop {
    position: absolute; inset: 0; background: rgba(0,0,0,.62);
    backdrop-filter: blur(2px);
  }
  .details-dialog {
    position: relative; width: min(900px, 96vw); max-height: min(88vh, 900px);
    overflow: hidden; background: var(--panel); color: var(--text);
    border: 1px solid var(--border); border-radius: 14px;
    box-shadow: 0 24px 80px rgba(0,0,0,.45); display: flex; flex-direction: column;
  }
  .details-header {
    display: flex; align-items: center; gap: 12px; padding: 16px 18px;
    border-bottom: 1px solid var(--border);
  }
  .details-title {
    flex: 1; min-width: 0; font-size: 18px; font-weight: 650;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .details-close {
    width: 34px; height: 34px; border-radius: 8px;
    border: 1px solid var(--border); background: var(--panel);
    color: var(--text); cursor: pointer; font-size: 20px; line-height: 1;
  }
  .details-close:hover { background: var(--row-hover); }
  .details-body { overflow: auto; padding: 18px; }
  .details-summary {
    margin-bottom: 16px; padding: 12px 14px; border: 1px solid var(--border);
    border-radius: 10px; background: var(--row-hover);
  }
  .details-section { margin-top: 18px; }
  .details-section h3 {
    margin: 0 0 9px; font-size: 12px; text-transform: uppercase;
    letter-spacing: .05em; color: var(--muted);
  }
  .details-grid {
    display: grid; grid-template-columns: minmax(150px, .34fr) minmax(0, 1fr);
    border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
  }
  .details-grid > div {
    padding: 8px 10px; border-bottom: 1px solid var(--border);
  }
  .details-grid > div:nth-child(odd) {
    color: var(--muted); background: color-mix(in srgb, var(--row-hover) 55%, var(--panel));
  }
  .details-grid > div:nth-last-child(-n+2) { border-bottom: 0; }
  .details-value { overflow-wrap: anywhere; }
  .details-mono { font-family: Consolas, "Cascadia Mono", monospace; font-size: 12px; }
  .details-list { margin: 0; padding-left: 18px; }
  .details-list li { margin: 4px 0; overflow-wrap: anywhere; }
  .details-actions {
    display: flex; gap: 8px; flex-wrap: wrap; padding: 14px 18px;
    border-top: 1px solid var(--border); background: var(--panel);
  }
  .details-action {
    display: inline-flex; align-items: center; gap: 7px; padding: 8px 12px;
    border-radius: 8px; border: 1px solid var(--border); background: var(--panel);
    color: var(--text); text-decoration: none; cursor: pointer; font: inherit;
  }
  .details-action.primary { color: var(--accent); }
  .details-action:hover { background: var(--row-hover); }
  .details-muted { color: var(--muted); }
  .details-warning { color: var(--accent2); }
  .details-duplicate { margin: 4px 0; }
  .details-role-badge {
    flex-shrink: 0; padding: 3px 10px; border-radius: 100px; font-size: 11.5px;
    font-weight: 600; letter-spacing: .02em; border: 1px solid var(--border);
  }
  .details-role-badge.lora { color: #c084fc; border-color: #c084fc55; }
  .details-role-badge.model { color: var(--accent); border-color: color-mix(in srgb, var(--accent) 45%, transparent); }
  .chip-list { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip {
    display: inline-flex; align-items: center; padding: 4px 10px;
    border-radius: 100px; background: var(--row-hover); border: 1px solid var(--border);
    font-size: 12.5px; overflow-wrap: anywhere;
  }
  .chip.trigger { color: #c084fc; border-color: #c084fc55; font-weight: 600; }
</style>
</head>
<body>

<h1>Model Zoo Inventory</h1>
<div class="meta">Generated: <code>__GENERATED__</code></div>

<div class="summary-grid">
  <div class="panel">
    <h2>Overview</h2>
    <div class="stat">__FILE_COUNT__</div>
    <div class="meta" style="margin-bottom:0">files, __TOTAL_SIZE__ total</div>
  </div>
  <div class="panel">
    <h2>By family</h2>
    <table><tbody>__FAMILY_ROWS__</tbody></table>
  </div>
  <div class="panel">
    <h2>By quantization</h2>
    <table><tbody>__QUANT_ROWS__</tbody></table>
  </div>
  <div class="panel">
    <h2>Duplicates</h2>
    <div class="stat">__DUP_COUNT__</div>
    <div class="meta" style="margin-bottom:0">group(s) &middot; __DUP_WASTED__ reclaimable</div>
  </div>
</div>

<div class="panel">
  <div id="rootTabs" class="root-tabs" role="tablist" aria-label="Scanned folders"></div>
  <div class="controls">
    <input id="search" type="text" placeholder="Search name, family, role, quantization, path...">
    <label class="meta">Group by
      <select id="groupBy">
        <option value="family" selected>Family</option>
        <option value="role">Role</option>
        <option value="quant">Quantization</option>
        <option value="">None</option>
      </select>
    </label>
    <label class="meta"><input id="dupOnly" type="checkbox"> Duplicates only</label>
    <button id="clearBtn" type="button">Clear</button>
    <span id="matchInfo"></span>
  </div>

  <div style="overflow-x:auto">
  <table id="modelsTable">
    <thead>
      <tr>
        <th data-key="name">Model<span class="arrow"></span></th>
        <th data-key="role">Role<span class="arrow"></span></th>
        <th data-key="family">Family<span class="arrow"></span></th>
        <th data-key="quant">Quantization<span class="arrow"></span></th>
        <th data-key="dtype">Dtype<span class="arrow"></span></th>
        <th data-key="size" class="num">Size<span class="arrow"></span></th>
        <th data-key="params" class="num">Tensor elements<span class="arrow"></span></th>
        <th data-key="confidence">Confidence<span class="arrow"></span></th>
        <th data-key="path">Path<span class="arrow"></span></th>
      </tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>
  </div>
</div>

<footer>
  Family detection is heuristic; confidence reflects detector evidence, not model quality.
  Pony and Illustrious can share an SDXL-derived tensor structure, so filename/metadata evidence matters.
  "Unknown" means no strong signature was detected, not that the file is invalid.
  <br>Scanned roots:
  <ul>__ROOTS_LIST__</ul>
</footer>

<div id="detailsModal" class="details-modal" hidden>
  <div class="details-backdrop" data-close-details></div>
  <section class="details-dialog" role="dialog" aria-modal="true" aria-labelledby="detailsTitle">
    <div class="details-header">
      <div id="detailsTitle" class="details-title">Model details</div>
      <span id="detailsRoleBadge" class="details-role-badge model"></span>
      <button id="detailsClose" class="details-close" type="button" aria-label="Close">×</button>
    </div>
    <div id="detailsBody" class="details-body"></div>
    <div id="detailsActions" class="details-actions"></div>
  </section>
</div>

<script>
const DATA = __DATA_JSON__;

const state = {
  search: "", sortKey: "size", sortDir: -1, groupBy: "family",
  dupOnly: false, root: "__ALL__"
};

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
}

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function highlight(text, query) {
  const escaped = escapeHtml(text == null ? "" : String(text));
  if (!query) return escaped;
  const re = new RegExp(escapeRegExp(query), "ig");
  return escaped.replace(re, m => "<mark>" + m + "</mark>");
}

function matchesRoot(rec) {
  return state.root === "__ALL__" || rec.root === state.root;
}

function matches(rec, q) {
  if (!matchesRoot(rec)) return false;
  if (state.dupOnly && !(rec.duplicates && rec.duplicates.length)) return false;
  if (!q) return true;
  const hay = [rec.name, rec.role, rec.family, rec.quant, rec.dtype, rec.path, rec.fullPath]
    .join(" ").toLowerCase();
  return hay.includes(q.toLowerCase());
}

function compareVal(a, b, key) {
  let av = a[key], bv = b[key];
  if (av == null && bv == null) return 0;
  if (av == null) return -1;
  if (bv == null) return 1;
  if (typeof av === "number" && typeof bv === "number") return av - bv;
  return String(av).toLowerCase().localeCompare(String(bv).toLowerCase());
}

function renderDiagCell(rec) {
  const hasDiag = rec.quantLayers != null || (rec.specialFormats && rec.specialFormats.length)
    || Object.keys(rec.metadata || {}).length || (rec.warnings && rec.warnings.length)
    || (rec.duplicates && rec.duplicates.length);
  if (!hasDiag) return "";
  let body = "<div>Family detection via " + escapeHtml(rec.detection) + "</div>";
  if (rec.quantLayers != null) body += "<div>Quantized layer records: " + rec.quantLayers + "</div>";
  if (rec.specialFormats && rec.specialFormats.length)
    body += "<div>Quant formats: " + escapeHtml(rec.specialFormats.join(", ")) + "</div>";
  for (const [k, v] of Object.entries(rec.metadata || {}))
    body += "<div>" + escapeHtml(k) + ": " + escapeHtml(v) + "</div>";
  for (const w of (rec.warnings || [])) body += "<div>&#9888; " + escapeHtml(w) + "</div>";
  for (const d of (rec.duplicates || [])) {
    body += "<div>&#8942; duplicate of " + (
      d.link
        ? "<a class='model-link' href='" + d.link + "' target='_blank' rel='noopener'>" + escapeHtml(d.path) + "</a>"
        : escapeHtml(d.path)
    ) + "</div>";
  }
  return "<details class='diag'><summary>details</summary>" + body + "</details>";
}

function rowHtml(rec, q, index) {
  const nameCell = "<span class='model-link'>" + highlight(rec.name, q) + "</span>" +
    (rec.link
      ? " <a class='details-direct-link' href='" + rec.link + "' target='_blank' rel='noopener' title='Open file'>↗</a>"
      : "");
  const dupBadge = (rec.duplicates && rec.duplicates.length)
    ? " <span class='badge MEDIUM' title='Likely duplicate(s): same quick fingerprint'>dup &times;" + rec.duplicates.length + "</span>"
    : "";
  const loraKeywordCount = (rec.loraTriggerWords?.length || 0) + (rec.loraTags?.length || 0);
  const kwBadge = loraKeywordCount
    ? " <span class='badge HIGH' title='" + loraKeywordCount + " trigger word(s)/training tag(s) found'>🔑 " + loraKeywordCount + "</span>"
    : "";
  return "<tr class='model-row' data-index='" + index + "' title='Click for details'>" +
    "<td>" + nameCell + dupBadge + kwBadge + renderDiagCell(rec) + "</td>" +
    "<td>" + highlight(rec.role, q) + "</td>" +
    "<td>" + highlight(rec.family, q) + "</td>" +
    "<td>" + highlight(rec.quant, q) + "</td>" +
    "<td>" + highlight(rec.dtype, q) + "</td>" +
    "<td class='num'>" + rec.sizeH + "</td>" +
    "<td class='num'>" + rec.paramsH + "</td>" +
    "<td><span class='badge " + rec.confidence + "'>" + rec.confidence + "</span></td>" +
    "<td class='path-cell'>" + highlight(rec.path, q) + "</td>" +
    "</tr>";
}

function detailsText(value, fallback = "—") {
  if (value == null || value === "") return fallback;
  return escapeHtml(String(value));
}

function detailsLink(href, label) {
  if (!href) return "<span class='details-muted'>Unavailable</span>";
  return "<a class='model-link' href='" + href + "' target='_blank' rel='noopener'>" + escapeHtml(label) + "</a>";
}

function detailsRow(label, value, className = "") {
  return "<div>" + escapeHtml(label) + "</div><div class='details-value " + className + "'>" + value + "</div>";
}

function renderChips(items, chipClass = "") {
  if (!items || !items.length) return "";
  return "<div class='chip-list'>" + items.map(
    t => "<span class='chip " + chipClass + "'>" + escapeHtml(t) + "</span>"
  ).join("") + "</div>";
}

function renderWarningsAndDuplicates(rec) {
  let body = "";
  if (rec.warnings && rec.warnings.length) {
    body += "<div class='details-section'><h3>Warnings</h3><ul class='details-list'>";
    for (const warning of rec.warnings) {
      body += "<li class='details-warning'>" + escapeHtml(warning) + "</li>";
    }
    body += "</ul></div>";
  }
  if (rec.duplicates && rec.duplicates.length) {
    body += "<div class='details-section'><h3>Likely duplicates</h3>";
    for (const d of rec.duplicates) {
      body += "<div class='details-duplicate'>" +
        (d.link
          ? "<a class='model-link' href='" + d.link + "' target='_blank' rel='noopener'>" + escapeHtml(d.path) + "</a>"
          : escapeHtml(d.path)) +
        "</div>";
    }
    body += "</div>";
  }
  return body;
}

// Popup for checkpoints / diffusion models / VAEs / text encoders / GGUF —
// architecture and quantization take center stage, no LoRA training noise.
function renderModelDetailsBody(rec) {
  const summary = "This item is classified as <strong>" + escapeHtml(rec.role || "Unknown") +
    "</strong> in the <strong>" + escapeHtml(rec.family || "Unknown") + "</strong> family. " +
    "Quantization: <strong>" + escapeHtml(rec.quant || "None detected") + "</strong>. " +
    "Family confidence: <strong>" + escapeHtml(rec.confidence || "UNKNOWN") + "</strong>.";

  let body = "<div class='details-summary'>" + summary + "</div>";

  body += "<div class='details-section'><h3>File</h3><div class='details-grid'>" +
    detailsRow("Name", detailsText(rec.name)) +
    detailsRow("Extension", detailsText(rec.ext)) +
    detailsRow("Role", detailsText(rec.role)) +
    detailsRow("Family", detailsText(rec.family)) +
    detailsRow("Family confidence", detailsText(rec.confidence)) +
    detailsRow("Family detection", detailsText(rec.detection)) +
    detailsRow("Relative path", "<span class='details-mono'>" + detailsText(rec.path) + "</span>") +
    detailsRow("Full path", "<span class='details-mono'>" + detailsText(rec.fullPath) + "</span>") +
    "</div></div>";

  body += "<div class='details-section'><h3>Storage & structure</h3><div class='details-grid'>" +
    detailsRow("File size", detailsText(rec.sizeH) + " (" + detailsText(rec.size) + " bytes)") +
    detailsRow("Tensor payload", detailsText(rec.payloadH) + " (" + detailsText(rec.payloadBytes) + " bytes)") +
    detailsRow("Tensor count", detailsText(rec.tensors)) +
    detailsRow("Tensor elements", detailsText(rec.paramsH) + " (" + detailsText(rec.params) + ")") +
    detailsRow("Dtype", detailsText(rec.dtype)) +
    detailsRow("Quick fingerprint", "<span class='details-mono'>" + detailsText(rec.quickHash) + "</span>") +
    "</div></div>";

  body += "<div class='details-section'><h3>Quantization</h3><div class='details-grid'>" +
    detailsRow("Quantization", detailsText(rec.quant)) +
    detailsRow("Detected from", detailsText(rec.quantSource)) +
    detailsRow("Quantized layer records", detailsText(rec.quantLayers)) +
    detailsRow("Special formats", detailsText(
      rec.specialFormats && rec.specialFormats.length ? rec.specialFormats.join(", ") : null
    )) +
    "</div></div>";

  const metadataEntries = Object.entries(rec.metadata || {});
  if (metadataEntries.length) {
    let metadataHtml = "";
    for (const [key, value] of metadataEntries) {
      metadataHtml += detailsRow(key, detailsText(value));
    }
    body += "<div class='details-section'><h3>Model metadata</h3><div class='details-grid'>" +
      metadataHtml + "</div></div>";
  }

  body += renderWarningsAndDuplicates(rec);
  return body;
}

// Popup for LoRA / LyCORIS adapters — leads with trigger words/keywords and
// the training recipe (base model, network dim/alpha, epochs...), which is
// what you actually need when deciding which LoRA to load. Checkpoint-only
// fields (quantization, full architecture breakdown) are left out entirely
// so the two popups never mix content.
function renderLoraDetailsBody(rec) {
  const details = rec.loraDetails || {};
  const summary = "LoRA / adapter in the <strong>" + escapeHtml(rec.family || "Unknown") + "</strong> family" +
    (details["Base model"] ? ", trained on <strong>" + escapeHtml(details["Base model"]) + "</strong>" : "") +
    ". Family confidence: <strong>" + escapeHtml(rec.confidence || "UNKNOWN") + "</strong>.";

  let body = "<div class='details-summary'>" + summary + "</div>";

  body += "<div class='details-section'><h3>Trigger words / keywords</h3>" + (
    rec.loraKeywords && rec.loraKeywords.length
      ? renderChips(rec.loraKeywords, "trigger")
      : "<div class='details-muted'>No trigger phrase or caption-tag metadata embedded in this file.</div>"
  ) + "</div>";

  const trainingRows = [];
  for (const label of ["Base model", "Network module", "Network dim", "Network alpha",
                        "Training images", "Epochs", "Learning rate", "Resolution", "Output name"]) {
    if (details[label]) trainingRows.push(detailsRow(label, detailsText(details[label])));
  }
  if (trainingRows.length) {
    body += "<div class='details-section'><h3>Training recipe</h3><div class='details-grid'>" +
      trainingRows.join("") + "</div></div>";
  }

  body += "<div class='details-section'><h3>File</h3><div class='details-grid'>" +
    detailsRow("Name", detailsText(rec.name)) +
    detailsRow("Family", detailsText(rec.family)) +
    detailsRow("Family detection", detailsText(rec.detection)) +
    detailsRow("File size", detailsText(rec.sizeH)) +
    detailsRow("Relative path", "<span class='details-mono'>" + detailsText(rec.path) + "</span>") +
    detailsRow("Full path", "<span class='details-mono'>" + detailsText(rec.fullPath) + "</span>") +
    "</div></div>";

  const metadataEntries = Object.entries(rec.metadata || {});
  if (metadataEntries.length) {
    let metadataHtml = "";
    for (const [key, value] of metadataEntries) {
      metadataHtml += detailsRow(key, detailsText(value));
    }
    body += "<div class='details-section'><h3>Other metadata</h3><div class='details-grid'>" +
      metadataHtml + "</div></div>";
  }

  body += renderWarningsAndDuplicates(rec);
  return body;
}

function showDetails(index) {
  const rec = DATA[index];
  if (!rec) return;

  const isLora = !!(rec.role && rec.role.startsWith("LoRA"));

  document.getElementById("detailsTitle").textContent = rec.name || "Details";

  const badge = document.getElementById("detailsRoleBadge");
  badge.textContent = isLora ? "LoRA / adapter" : (rec.role || "Model");
  badge.className = "details-role-badge " + (isLora ? "lora" : "model");

  document.getElementById("detailsBody").innerHTML = isLora
    ? renderLoraDetailsBody(rec)
    : renderModelDetailsBody(rec);

  let actions = "";
  if (rec.link) {
    actions += "<a class='details-action primary' href='" + rec.link + "' target='_blank' rel='noopener'>📄 Open file</a>";
  }
  if (rec.folderLink) {
    actions += "<a class='details-action' href='" + rec.folderLink + "' target='_blank' rel='noopener'>📂 Open folder</a>";
  }
  document.getElementById("detailsActions").innerHTML = actions;
  document.getElementById("detailsModal").hidden = false;
  document.body.style.overflow = "hidden";
}

function closeDetails() {
  document.getElementById("detailsModal").hidden = true;
  document.body.style.overflow = "";
}

function groupRowHtml(label, items) {
  const size = items.reduce((s, r) => s + (r.size || 0), 0);
  return "<tr class='group-row'><td colspan='9'>" + escapeHtml(label) +
    " &nbsp;\u00b7&nbsp; " + items.length + " file" + (items.length === 1 ? "" : "s") +
    " &nbsp;\u00b7&nbsp; " + humanBytes(size) + "</td></tr>";
}

function humanBytes(n) {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  for (let i = 0; i < units.length; i++) {
    if (v < 1024 || units[i] === "TB") return v.toFixed(2) + " " + units[i];
    v /= 1024;
  }
  return v.toFixed(2) + " TB";
}

function renderRootTabs() {
  const tabs = document.getElementById("rootTabs");
  const roots = [...new Set(DATA.map(r => r.root).filter(Boolean))].sort((a, b) =>
    a.toLowerCase().localeCompare(b.toLowerCase())
  );
  const entries = [{ key: "__ALL__", label: "All folders", count: DATA.length }, ...roots.map(root => ({
    key: root,
    label: root,
    count: DATA.filter(r => r.root === root).length
  }))];

  tabs.innerHTML = entries.map(entry => {
    const active = state.root === entry.key ? " active" : "";
    return "<button class='root-tab" + active + "' type='button' role='tab' " +
      "aria-selected='" + (state.root === entry.key ? "true" : "false") + "' " +
      "title='" + escapeHtml(entry.label) + "' data-root='" + escapeHtml(entry.key) + "'>" +
      escapeHtml(entry.label) +
      "<span class='tab-count'>(" + entry.count + ")</span></button>";
  }).join("");

  tabs.querySelectorAll(".root-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      state.root = tab.dataset.root;
      renderRootTabs();
      render();
    });
  });
}

function render() {
  const q = state.search.trim();
  let rows = DATA.filter(r => matches(r, q));
  document.getElementById("matchInfo").textContent =
    q ? (rows.length + " / " + DATA.length + " match") : (DATA.length + " files");

  rows = rows.slice().sort((a, b) => state.sortDir * compareVal(a, b, state.sortKey));

  const tbody = document.getElementById("tbody");
  if (!rows.length) {
    tbody.innerHTML = "<tr><td colspan='9' class='empty-state'>No models match your search.</td></tr>";
    return;
  }

  let html = "";
  if (state.groupBy) {
    const groups = new Map();
    for (const r of rows) {
      const key = r[state.groupBy] || "Unknown";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(r);
    }
    const groupKeys = [...groups.keys()].sort((a, b) => {
      const sa = groups.get(a).reduce((s, r) => s + (r.size || 0), 0);
      const sb = groups.get(b).reduce((s, r) => s + (r.size || 0), 0);
      return sb - sa;
    });
    for (const key of groupKeys) {
      html += groupRowHtml(key, groups.get(key));
      for (const r of groups.get(key)) html += rowHtml(r, q, DATA.indexOf(r));
    }
  } else {
    for (const r of rows) html += rowHtml(r, q, DATA.indexOf(r));
  }
  tbody.innerHTML = html;
}

document.getElementById("search").addEventListener("input", e => {
  state.search = e.target.value;
  render();
});

document.getElementById("clearBtn").addEventListener("click", () => {
  state.search = "";
  document.getElementById("search").value = "";
  render();
});

document.getElementById("groupBy").addEventListener("change", e => {
  state.groupBy = e.target.value;
  render();
});

document.getElementById("dupOnly").addEventListener("change", e => {
  state.dupOnly = e.target.checked;
  render();
});

document.getElementById("tbody").addEventListener("click", e => {
  if (e.target.closest("a, button, input, select, details, summary")) return;
  const row = e.target.closest("tr.model-row");
  if (!row) return;
  const index = Number(row.dataset.index);
  if (Number.isInteger(index)) showDetails(index);
});

document.getElementById("detailsClose").addEventListener("click", closeDetails);
document.querySelector("[data-close-details]").addEventListener("click", closeDetails);
document.addEventListener("keydown", e => {
  if (e.key === "Escape" && !document.getElementById("detailsModal").hidden) {
    closeDetails();
  }
});

document.querySelectorAll("#modelsTable thead th").forEach(th => {
  th.addEventListener("click", () => {
    const key = th.dataset.key;
    if (state.sortKey === key) {
      state.sortDir *= -1;
    } else {
      state.sortKey = key;
      state.sortDir = 1;
    }
    document.querySelectorAll("#modelsTable thead th").forEach(h => {
      h.classList.remove("active");
      h.querySelector(".arrow").textContent = "";
    });
    th.classList.add("active");
    th.querySelector(".arrow").textContent = state.sortDir === 1 ? "\u25b2" : "\u25bc";
    render();
  });
});

renderRootTabs();
render();
</script>
</body>
</html>
"""


def write_html(path: Path, html: str) -> None:
    # Plain, self-contained HTML rather than MHTML: several browsers
    # (Chrome/Edge included) disable JavaScript execution when opening a
    # local .mhtml file as a security measure, which silently breaks the
    # search/sort/group controls. A .html file has no such restriction.
    path.write_text(html, encoding="utf-8")


def write_json(
    path: Path,
    infos: list[ModelInfo],
    roots: list[Path],
) -> None:
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "roots": [str(root) for root in roots],
        "files": [asdict(item) for item in infos],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inventory model files and generate an interactive HTML report."
    )
    parser.add_argument(
        "roots",
        nargs="*",
        help="folders to scan recursively",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="models.html",
        help="HTML report output path (interactive, sortable, searchable)",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        default=None,
        help="optional JSON output path",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=None,
        help="extensions to include, e.g. .safetensors .gguf",
    )
    parser.add_argument(
        "--cache",
        default=".model_inventory_cache.json",
        help="cache file path; unchanged files (same size+mtime_ns) are read "
             "from here instead of being re-inspected (default: %(default)s)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore and don't update the cache; always re-inspect everything",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="thread pool size for parallel file inspection (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.roots:
        roots = [Path(x).expanduser() for x in args.roots]
    else:
        roots = [
            Path(r"C:\checkpoints"),
        ]
        print("No roots supplied; using default Forge-style roots:")
        for root in roots:
            print(f"  {root}")

    suffixes = (
        {x.lower() for x in args.include}
        if args.include
        else set(SUPPORTED_SUFFIXES)
    )

    files = collect_files(roots, suffixes)
    print(f"Found {len(files)} model files.")

    cache_path = Path(args.cache)
    cache_entries = {} if args.no_cache else load_cache(cache_path)

    infos: list[ModelInfo] = []
    to_inspect: list[tuple[Path, Path]] = []
    path_stats: dict[str, tuple[int, int]] = {}

    for path, root in files:
        key = str(path.resolve())
        try:
            st = path.stat()
        except OSError as exc:
            infos.append(inspect_or_placeholder(path, root))
            print(f"[!] Cannot stat {path}: {exc}")
            continue

        path_stats[key] = (st.st_size, st.st_mtime_ns)
        cached = cache_entries.get(key)
        if (
            cached is not None
            and cached.get("size") == st.st_size
            and cached.get("mtime_ns") == st.st_mtime_ns
        ):
            try:
                info = ModelInfo(**cached["info"])
                rel = path.relative_to(root) if path.is_relative_to(root) else path
                info = replace(info, path=str(rel), duplicate_paths=[])
                infos.append(info)
                continue
            except Exception:
                pass  # malformed cache entry: fall through and re-inspect
        to_inspect.append((path, root))

    reused = len(files) - len(to_inspect)
    if to_inspect:
        print(f"Reusing {reused} cached, inspecting {len(to_inspect)} file(s) "
              f"with {args.workers} worker thread(s)...")
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            n = 0
            for info in pool.map(
                lambda pr: inspect_or_placeholder(*pr), to_inspect
            ):
                n += 1
                print(f"[{n}/{len(to_inspect)}] {info.full_path}")
                infos.append(info)
    else:
        print(f"Reusing all {reused} file(s) from cache; nothing to inspect.")

    mark_duplicates(infos)

    if not args.no_cache:
        info_by_full_path = {info.full_path: info for info in infos}
        new_entries = {
            key: {"size": size, "mtime_ns": mtime_ns, "info": asdict(info_by_full_path[key])}
            for key, (size, mtime_ns) in path_stats.items()
            if key in info_by_full_path
        }
        save_cache(cache_path, new_entries)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_html(output, build_html(infos, roots))

    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(json_path, infos, roots)

    total = sum(i.file_size_bytes for i in infos)
    dup_groups = {
        i.full_path for i in infos if i.duplicate_paths
    }
    print()
    print("Done.")
    print(f"  Files: {len(infos)}")
    print(f"  Total: {human_bytes(total)}")
    if dup_groups:
        print(f"  Likely duplicates: {len(dup_groups)} file(s) share a quick fingerprint with another file")
    print(f"  Report: {output}")
    if args.json_output:
        print(f"  JSON: {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
