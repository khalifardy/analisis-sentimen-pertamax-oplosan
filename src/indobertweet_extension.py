"""Controlled frozen-IndoBERTweet extension for the ICDSG 2026 experiment.

This module intentionally does not import TensorFlow or ``src.balancing``.
The existing balancing registry only wraps the default imbalanced-learn
samplers with ``random_state``; importing it would also initialize TensorFlow.
The equivalent sampler mapping is declared here and audited in the notebook.

No function in this module executes at import time. Long operations are only
started by explicit calls from ``experiment_49_indobertweet.ipynb``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


LABELS = [0, 1, 2]
LABEL_NAMES = ["negative", "neutral", "positive"]
SEEDS = [42, 123, 456]
SAMPLING_METHODS = [
    "No_Balancing",
    "ROS",
    "SMOTE",
    "RUS",
    "SMOTEENN",
    "SMOTETomek",
]
EXTENSION_METHODS = SAMPLING_METHODS + ["Class_Weighted"]
REQUIRED_EXISTING_METRICS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f1_macro",
    "f1_negative",
    "f1_neutral",
    "f1_positive",
}


def _jsonable(value: Any) -> Any:
    """Convert NumPy/path/container values to deterministic JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    """Write JSON through a temporary sibling and atomically replace target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npy_save(array: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, path)


def atomic_npz_save(path: Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_values(values: Iterable[int]) -> str:
    array = np.asarray(list(values), dtype=np.int64)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def distribution(values: Sequence[int] | np.ndarray | pd.Series) -> dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {str(label): int(counts.get(label, 0)) for label in LABELS}


def prepare_output_paths(project_root: Path) -> dict[str, Path]:
    project_root = Path(project_root).resolve()
    base = project_root / "results" / "indobertweet"
    paths = {
        "base": base,
        "cache": base / "cache",
        "predictions": base / "indobertweet_predictions",
        "reports": base / "classification_reports",
        "confusion_matrices": base / "confusion_matrices",
        "figures": base / "figures",
        "checkpoints": base / "checkpoints",
        "results": base / "indobertweet_results.json",
        "split": base / "split_indices.json",
        "combined": project_root / "results" / "combined_results_49.json",
    }
    for key, path in paths.items():
        target = path if key in {"base", "cache", "predictions", "reports", "confusion_matrices", "figures", "checkpoints"} else path.parent
        target.mkdir(parents=True, exist_ok=True)
    return paths


def audit_csv(path: Path, text_column: str = "clean_text", label_column: str = "label") -> dict[str, Any]:
    path = Path(path).resolve()
    frame = pd.read_csv(path)
    missing = {text_column, label_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    text = frame[text_column]
    empty = text.fillna("").astype(str).str.strip().eq("")
    labels = frame[label_column].astype(int)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "columns": frame.columns.tolist(),
        "label_distribution": distribution(labels),
        "null_text": int(text.isna().sum()),
        "empty_text": int(empty.sum()),
        "unique_text": int(text.nunique(dropna=False)),
        "duplicate_rows_beyond_first": int(text.duplicated().sum()),
    }


def validate_dataset_files(raw_path: Path, modeling_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw_path = Path(raw_path).resolve()
    modeling_path = Path(modeling_path).resolve()
    raw_audit = audit_csv(raw_path)
    modeling_audit = audit_csv(modeling_path)
    raw = pd.read_csv(raw_path)
    modeling = pd.read_csv(modeling_path)
    modeling["label"] = modeling["label"].astype(int)

    expected_raw = {"rows": 34_993, "label_distribution": {"0": 27_357, "1": 6_650, "2": 986}}
    expected_model = {"rows": 34_985, "label_distribution": {"0": 27_350, "1": 6_649, "2": 986}}
    for name, audit, expected in (
        ("raw adjudicated", raw_audit, expected_raw),
        ("modeling", modeling_audit, expected_model),
    ):
        if audit["rows"] != expected["rows"] or audit["label_distribution"] != expected["label_distribution"]:
            raise ValueError(f"Unexpected {name} dataset evidence: {audit}")
        if audit["null_text"] or audit["empty_text"]:
            raise ValueError(f"{name} dataset contains null/empty text: {audit}")

    return raw, modeling, {"raw_adjudicated": raw_audit, "modeling": modeling_audit}


def reproduce_alignment(raw: pd.DataFrame, modeling: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Re-run the recorded preprocessing while retaining source row indices.

    This is deliberately a long validation step. It never joins on tweet text.
    An exact row-by-row match is required before original tweets are used.
    """
    from src.preprocessing import full_preprocessing

    if raw["clean_text"].isna().any():
        raise ValueError("Raw adjudicated text contains null values.")
    working = pd.DataFrame(
        {
            "source_index": np.arange(len(raw), dtype=np.int64),
            "original_text": raw["clean_text"].astype(str),
            "label": raw["label"].astype(int),
        }
    )
    reproduced = full_preprocessing(working.copy(), kolom="original_text", use_stemming=True)
    reproduced["label"] = reproduced["label"].astype(int)
    same_count = len(reproduced) == len(modeling)
    same_labels = same_count and np.array_equal(reproduced["label"].to_numpy(), modeling["label"].to_numpy())
    same_text = same_count and reproduced["clean_text"].astype(str).reset_index(drop=True).equals(
        modeling["clean_text"].astype(str).reset_index(drop=True)
    )
    evidence = {
        "method": "row-preserving replay of src.preprocessing.full_preprocessing; no text join",
        "raw_rows": int(len(raw)),
        "reproduced_rows": int(len(reproduced)),
        "modeling_rows": int(len(modeling)),
        "removed_source_rows": sorted(set(range(len(raw))) - set(reproduced["source_index"].astype(int))),
        "exact_label_order_match": bool(same_labels),
        "exact_preprocessed_text_order_match": bool(same_text),
        "source_index_unique": bool(reproduced["source_index"].is_unique),
    }
    if not (same_count and same_labels and same_text and evidence["source_index_unique"]):
        raise RuntimeError(f"Original-text alignment could not be verified: {evidence}")
    return reproduced[["source_index", "original_text", "clean_text", "label"]].copy(), evidence


def minimally_prepare_tweet(text: str) -> str:
    """Preserve tweet content while normalizing URLs, mentions, and whitespace."""
    value = re.sub(r"https?://\S+|www\.\S+", "HTTPURL", str(text))
    value = re.sub(r"@\w+", "@USER", value)
    return re.sub(r"\s+", " ", value).strip()


def reconstruct_existing_split(modeling: pd.DataFrame, seed: int = 42) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    indices = np.arange(len(modeling), dtype=np.int64)
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=seed)
    y = modeling["label"].astype(int).to_numpy()
    train_text = set(modeling.iloc[train_indices]["clean_text"].astype(str))
    test_text = set(modeling.iloc[test_indices]["clean_text"].astype(str))
    overlap = train_text & test_text
    evidence = {
        "implementation": "sklearn.model_selection.train_test_split",
        "test_size": 0.2,
        "random_state": int(seed),
        "stratify": None,
        "matches_existing_code": True,
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "train_distribution": distribution(y[train_indices]),
        "test_distribution": distribution(y[test_indices]),
        "train_indices_sha256": sha256_values(train_indices),
        "test_indices_sha256": sha256_values(test_indices),
        "duplicate_text_overlap_unique": int(len(overlap)),
        "test_rows_with_text_seen_in_train": int(modeling.iloc[test_indices]["clean_text"].isin(train_text).sum()),
        "train_rows_with_text_seen_in_test": int(modeling.iloc[train_indices]["clean_text"].isin(test_text).sum()),
        "limitation": "The historical split is not stratified or group-aware; duplicate text overlaps train and test.",
    }
    return train_indices, test_indices, evidence


def save_split_indices(path: Path, train_indices: np.ndarray, test_indices: np.ndarray, evidence: Mapping[str, Any]) -> None:
    payload = dict(evidence)
    payload["train_indices"] = train_indices.astype(int).tolist()
    payload["test_indices"] = test_indices.astype(int).tolist()
    atomic_json_dump(payload, Path(path))


def validate_existing_results(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if list(payload) != ["Dataset_A"]:
        raise ValueError(f"Expected only Dataset_A, found: {list(payload)}")
    dataset = payload["Dataset_A"]
    details: list[dict[str, Any]] = []
    for balancing_name, models in dataset.items():
        for model_name, record in models.items():
            missing = REQUIRED_EXISTING_METRICS - set(record)
            if missing:
                raise ValueError(f"Missing metrics for {model_name} + {balancing_name}: {sorted(missing)}")
            for metric in REQUIRED_EXISTING_METRICS:
                metric_record = record[metric]
                if set(metric_record) < {"runs", "mean", "std"} or len(metric_record["runs"]) != 3:
                    raise ValueError(f"Incomplete {metric} evidence for {model_name} + {balancing_name}")
            details.append({"balancing": balancing_name, "model": model_name})
    if len(details) != 42:
        raise ValueError(f"Expected 42 existing configurations, found {len(details)}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "configuration_count": len(details),
        "runs_per_configuration": 3,
        "required_metrics": sorted(REQUIRED_EXISTING_METRICS),
        "payload": payload,
    }


def environment_report(expected_dimension: int = 768) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "logical_cores": os.cpu_count(),
        "expected_embedding_dimension": expected_dimension,
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        report["ram_total_gib"] = memory.total / 1024**3
        report["ram_available_gib"] = memory.available / 1024**3
        report["disk_free_gib"] = psutil.disk_usage(str(Path.cwd())).free / 1024**3
    except Exception as error:  # pragma: no cover - environment dependent
        report["psutil_error"] = repr(error)
    try:
        import torch

        report["torch_version"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            report["vram_total_gib"] = total / 1024**3
            report["vram_free_gib"] = free / 1024**3
    except Exception as error:
        report["torch_error"] = repr(error)
        report["cuda_available"] = False
    return report


def estimate_resources(train_distribution: Mapping[str, int], test_rows: int, dimension: int = 768) -> dict[str, Any]:
    counts = [int(train_distribution[str(label)]) for label in LABELS]
    majority = max(counts)
    minority = min(counts)
    expected_rows = {
        "No_Balancing": sum(counts),
        "ROS": majority * len(LABELS),
        "SMOTE": majority * len(LABELS),
        "RUS": minority * len(LABELS),
        "SMOTEENN": f"<= {majority * len(LABELS)} after ENN cleaning",
        "SMOTETomek": f"<= {majority * len(LABELS)} after Tomek cleaning",
        "Class_Weighted": sum(counts),
    }
    bytes_per_row = dimension * np.dtype(np.float32).itemsize
    cache_bytes = (sum(counts) + int(test_rows)) * bytes_per_row
    oversampled_bytes = majority * len(LABELS) * bytes_per_row
    return {
        "float_dtype": "float32",
        "bytes_per_embedding": bytes_per_row,
        "embedding_cache_mib": cache_bytes / 1024**2,
        "largest_resampled_embedding_matrix_mib": oversampled_bytes / 1024**2,
        "expected_training_rows": expected_rows,
        "disk_guidance_gib": 2.0,
        "risk": "SMOTEENN performs synthetic oversampling plus nearest-neighbor cleaning and can need several multiples of the dense matrix size in RAM.",
    }


def _cache_files(cache_dir: Path) -> dict[str, Path]:
    return {
        "train_embeddings": cache_dir / "train_embeddings.npy",
        "test_embeddings": cache_dir / "test_embeddings.npy",
        "y_train": cache_dir / "y_train.npy",
        "y_test": cache_dir / "y_test.npy",
        "train_indices": cache_dir / "train_indices.npy",
        "test_indices": cache_dir / "test_indices.npy",
        "metadata": cache_dir / "metadata.json",
    }


def validate_embedding_cache(cache_dir: Path, expected: Mapping[str, Any]) -> tuple[bool, list[str], dict[str, Any] | None]:
    files = _cache_files(Path(cache_dir))
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        return False, [f"missing cache file: {name}" for name in missing], None
    metadata = json.loads(files["metadata"].read_text(encoding="utf-8"))
    reasons = [f"metadata mismatch: {key}" for key, value in expected.items() if metadata.get(key) != _jsonable(value)]
    reproducibility = metadata.get("reproducibility_check", {})
    if not reproducibility.get("passed", False):
        reasons.append("missing or failed 100-row reproducibility check")
    if not reproducibility.get("exact_shape_equality", False):
        reasons.append("reproducibility-check shapes are not exactly equal")
    if not reproducibility.get("allclose", False):
        reasons.append("reproducibility-check outputs are not within the strict tolerance")
    try:
        arrays = {name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in files.items() if name != "metadata"}
        if arrays["train_embeddings"].dtype != np.float32 or arrays["test_embeddings"].dtype != np.float32:
            reasons.append("embeddings are not float32")
        if arrays["train_embeddings"].shape != tuple(metadata.get("train_embedding_shape", [])):
            reasons.append("train embedding shape mismatch")
        if arrays["test_embeddings"].shape != tuple(metadata.get("test_embedding_shape", [])):
            reasons.append("test embedding shape mismatch")
        for name, path in files.items():
            if name != "metadata" and metadata.get("file_sha256", {}).get(name) != sha256_file(path):
                reasons.append(f"file hash mismatch: {name}")
    except Exception as error:
        reasons.append(f"cache load error: {error!r}")
    return not reasons, reasons, metadata


def extract_embeddings_once(
    aligned: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    cache_dir: Path,
    dataset_evidence: Mapping[str, Any],
    checkpoint: str = "indolem/indobertweet-base-uncased",
    max_length: int = 128,
    batch_size: int = 32,
    expected_dimension: int = 768,
    reproducibility_subset_size: int = 100,
    reproducibility_subset_seed: int = 42,
    reproducibility_rtol: float = 1e-6,
    reproducibility_atol: float = 1e-7,
) -> dict[str, Any]:
    """Extract attention-mask-aware mean-pooled embeddings once and cache them."""
    import torch
    import transformers
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = _cache_files(cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expected = {
        "dataset_path": dataset_evidence["modeling"]["path"],
        "dataset_sha256": dataset_evidence["modeling"]["sha256"],
        "raw_dataset_sha256": dataset_evidence["raw_adjudicated"]["sha256"],
        "row_count": int(len(aligned)),
        "label_distribution": distribution(aligned["label"]),
        "checkpoint_id": checkpoint,
        "tokenizer_id": checkpoint,
        "text_input": "verified original text; URL->HTTPURL, mention->@USER, whitespace normalization; no stemming or stopword removal",
        "pooling_method": "attention-mask-aware mean pooling over last hidden state",
        "maximum_sequence_length": int(max_length),
        "embedding_dimension": int(expected_dimension),
        "train_indices_sha256": sha256_values(train_indices),
        "test_indices_sha256": sha256_values(test_indices),
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": str(device),
        "extraction_batch_size": int(batch_size),
        "reproducibility_protocol": {
            "subset_size": int(reproducibility_subset_size),
            "subset_seed": int(reproducibility_subset_seed),
            "passes": 2,
            "model_eval": True,
            "torch_no_grad": True,
            "rtol": float(reproducibility_rtol),
            "atol": float(reproducibility_atol),
        },
    }
    valid, reasons, metadata = validate_embedding_cache(cache_dir, expected)
    if valid:
        print("Verified embedding cache; extraction skipped.")
        return metadata or expected
    print("Embedding cache invalid or absent:", reasons)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint).to(device)
    model.eval()
    hidden_size = int(model.config.hidden_size)
    if hidden_size != expected_dimension:
        raise RuntimeError(f"Expected hidden size {expected_dimension}, model reports {hidden_size}")

    prepared = aligned["original_text"].map(minimally_prepare_tweet).tolist()

    def encode(index_array: np.ndarray, label: str, *, report_progress: bool = True) -> np.ndarray:
        texts = [prepared[int(index)] for index in index_array]
        output = np.empty((len(texts), hidden_size), dtype=np.float32)
        loader = DataLoader(texts, batch_size=batch_size, shuffle=False)
        cursor = 0
        with torch.no_grad():
            for batch in loader:
                encoded = tokenizer(
                    list(batch),
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                hidden = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
                values = pooled.detach().cpu().numpy().astype(np.float32, copy=False)
                output[cursor : cursor + len(values)] = values
                cursor += len(values)
                if report_progress and (cursor % (batch_size * 50) == 0 or cursor == len(texts)):
                    print(f"{label}: {cursor:,}/{len(texts):,}")
        return output

    if reproducibility_subset_size <= 0 or reproducibility_subset_size > len(aligned):
        raise ValueError(
            f"reproducibility_subset_size must be in [1, {len(aligned)}], "
            f"received {reproducibility_subset_size}"
        )
    subset_rng = np.random.default_rng(reproducibility_subset_seed)
    subset_indices = np.sort(
        subset_rng.choice(len(aligned), size=reproducibility_subset_size, replace=False)
    ).astype(np.int64)
    model.eval()
    first_subset = encode(subset_indices, "reproducibility pass 1", report_progress=False)
    model.eval()
    second_subset = encode(subset_indices, "reproducibility pass 2", report_progress=False)
    exact_shape_equality = first_subset.shape == second_subset.shape
    if exact_shape_equality and first_subset.size:
        maximum_absolute_difference = float(
            np.max(np.abs(first_subset.astype(np.float64) - second_subset.astype(np.float64)))
        )
    else:
        maximum_absolute_difference = None
    strict_allclose = bool(
        exact_shape_equality
        and np.allclose(
            first_subset,
            second_subset,
            rtol=reproducibility_rtol,
            atol=reproducibility_atol,
            equal_nan=False,
        )
    )
    reproducibility_check = {
        "passed": bool(exact_shape_equality and strict_allclose),
        "subset_size": int(reproducibility_subset_size),
        "subset_seed": int(reproducibility_subset_seed),
        "subset_indices": subset_indices.tolist(),
        "subset_indices_sha256": sha256_values(subset_indices),
        "first_shape": list(first_subset.shape),
        "second_shape": list(second_subset.shape),
        "exact_shape_equality": bool(exact_shape_equality),
        "maximum_absolute_difference": maximum_absolute_difference,
        "allclose": strict_allclose,
        "rtol": float(reproducibility_rtol),
        "atol": float(reproducibility_atol),
        "first_output_sha256": hashlib.sha256(first_subset.tobytes(order="C")).hexdigest(),
        "second_output_sha256": hashlib.sha256(second_subset.tobytes(order="C")).hexdigest(),
        "first_output_all_finite": bool(np.isfinite(first_subset).all()),
        "second_output_all_finite": bool(np.isfinite(second_subset).all()),
        "model_training_flag_after_eval": bool(model.training),
        "device": str(device),
        "torch_deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    print("100-row embedding reproducibility check:")
    print(json.dumps(reproducibility_check, indent=2))
    if not reproducibility_check["passed"]:
        failure_path = cache_dir / "reproducibility_failure.json"
        atomic_json_dump(
            {
                **reproducibility_check,
                "action": "Cache rejected; full-dataset extraction was not started.",
                "investigation": [
                    "Confirm model.training is False after model.eval().",
                    "Confirm both passes use identical subset indices, tokenization, padding, and device.",
                    "Inspect CUDA/cuDNN deterministic settings and retry the same 100-row check on CPU if needed.",
                    "Do not reuse or merge any cache until this check passes.",
                ],
            },
            failure_path,
        )
        raise RuntimeError(
            "Embedding reproducibility check failed; cache rejected and full extraction stopped. "
            f"Diagnostics: {failure_path}"
        )

    started = time.perf_counter()
    train_embeddings = encode(train_indices, "train")
    test_embeddings = encode(test_indices, "test")
    y = aligned["label"].astype(int).to_numpy()
    arrays = {
        "train_embeddings": train_embeddings,
        "test_embeddings": test_embeddings,
        "y_train": y[train_indices].astype(np.int64),
        "y_test": y[test_indices].astype(np.int64),
        "train_indices": train_indices.astype(np.int64),
        "test_indices": test_indices.astype(np.int64),
    }
    for name, array in arrays.items():
        atomic_npy_save(array, files[name])
    metadata = {
        **expected,
        "train_embedding_shape": list(train_embeddings.shape),
        "test_embedding_shape": list(test_embeddings.shape),
        "extraction_time_seconds": time.perf_counter() - started,
        "reproducibility_check": reproducibility_check,
        "file_sha256": {name: sha256_file(files[name]) for name in arrays},
    }
    atomic_json_dump(metadata, files["metadata"])
    valid, reasons, _ = validate_embedding_cache(cache_dir, expected)
    if not valid:
        raise RuntimeError(f"Newly written embedding cache failed validation: {reasons}")
    del model, tokenizer, train_embeddings, test_embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metadata


def load_verified_cache(cache_dir: Path, expected: Mapping[str, Any] | None = None) -> dict[str, np.ndarray]:
    files = _cache_files(Path(cache_dir))
    if expected is not None:
        valid, reasons, _ = validate_embedding_cache(cache_dir, expected)
        if not valid:
            raise RuntimeError(f"Embedding cache is invalid: {reasons}")
    return {name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in files.items() if name != "metadata"}


def _make_sampler(name: str, seed: int):
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    classes = {
        "ROS": RandomOverSampler,
        "SMOTE": SMOTE,
        "RUS": RandomUnderSampler,
        "SMOTEENN": SMOTEENN,
        "SMOTETomek": SMOTETomek,
    }
    return None if name in {"No_Balancing", "Class_Weighted"} else classes[name](random_state=seed)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict[str, float], dict[str, Any], np.ndarray]:
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", labels=LABELS, zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
    per_class = precision_recall_fscore_support(y_true, y_pred, average=None, labels=LABELS, zero_division=0)
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(weighted[0]),
        "recall": float(weighted[1]),
        "f1": float(weighted[2]),
        "f1_macro": float(macro[2]),
    }
    for index, name in enumerate(LABEL_NAMES):
        metrics[f"precision_{name}"] = float(per_class[0][index])
        metrics[f"recall_{name}"] = float(per_class[1][index])
        metrics[f"f1_{name}"] = float(per_class[2][index])
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    return metrics, report, matrix


def memory_snapshot() -> dict[str, Any]:
    try:
        import psutil

        process = psutil.Process()
        system = psutil.virtual_memory()
        return {
            "process_rss_gib": process.memory_info().rss / 1024**3,
            "system_available_gib": system.available / 1024**3,
            "system_percent_used": system.percent,
        }
    except Exception as error:  # pragma: no cover - environment dependent
        return {"error": repr(error)}


def _aggregate_runs(runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    complete = [run for run in runs if run.get("status") == "complete"]
    if not complete:
        return {"completed_runs": 0, "expected_runs": len(SEEDS), "metrics": {}}
    metric_names = sorted(complete[0]["metrics"])
    metrics = {}
    for metric in metric_names:
        values = np.asarray([run["metrics"][metric] for run in complete], dtype=float)
        metrics[metric] = {"runs": values.tolist(), "mean": float(values.mean()), "std": float(values.std(ddof=0))}
    return {"completed_runs": len(complete), "expected_runs": len(SEEDS), "metrics": metrics}


def collect_extension_results(checkpoints_dir: Path, output_path: Path, experiment_signature: Mapping[str, Any]) -> dict[str, Any]:
    runs_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in EXTENSION_METHODS}
    for path in sorted(Path(checkpoints_dir).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        method = record.get("method")
        if method in runs_by_method:
            runs_by_method[method].append(record)
    for method in runs_by_method:
        runs_by_method[method].sort(key=lambda item: SEEDS.index(int(item["seed"])) if int(item["seed"]) in SEEDS else 999)
    payload = {
        "experiment": _jsonable(experiment_signature),
        "configuration_count": len(EXTENSION_METHODS),
        "expected_runs": len(EXTENSION_METHODS) * len(SEEDS),
        "configurations": {
            method: {"runs": runs_by_method[method], "summary": _aggregate_runs(runs_by_method[method])}
            for method in EXTENSION_METHODS
        },
    }
    atomic_json_dump(payload, Path(output_path))
    return payload


def run_extension_experiments(
    cache_dir: Path,
    paths: Mapping[str, Path],
    experiment_signature: Mapping[str, Any],
    max_iter: int = 500,
) -> dict[str, Any]:
    """Run seven configurations sequentially with per-seed crash-safe evidence."""
    cache = load_verified_cache(cache_dir)
    x_train = cache["train_embeddings"]
    x_test = cache["test_embeddings"]
    y_train = np.asarray(cache["y_train"], dtype=np.int64)
    y_test = np.asarray(cache["y_test"], dtype=np.int64)
    signature_hash = hashlib.sha256(json.dumps(_jsonable(experiment_signature), sort_keys=True).encode()).hexdigest()

    for method in EXTENSION_METHODS:
        method_failed = False
        for seed in SEEDS:
            checkpoint_path = Path(paths["checkpoints"]) / f"{method}_seed{seed}.json"
            if checkpoint_path.exists():
                previous = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                prediction_path = Path(paths["predictions"]) / f"{method}_seed{seed}.npz"
                if previous.get("status") == "complete" and previous.get("experiment_signature_sha256") == signature_hash and prediction_path.exists():
                    print(f"RESUME: {method}, seed={seed} already complete")
                    continue
            stage = "initialization"
            sampler = None
            try:
                print(f"RUN: {method}, seed={seed}")
                stage = "sampling"
                sampling_started = time.perf_counter()
                sampler = _make_sampler(method, seed)
                if sampler is None:
                    x_resampled, y_resampled = x_train, y_train
                else:
                    x_resampled, y_resampled = sampler.fit_resample(x_train, y_train)
                sampling_seconds = time.perf_counter() - sampling_started

                stage = "classifier_training"
                classifier = LogisticRegression(
                    max_iter=max_iter,
                    random_state=seed,
                    class_weight="balanced" if method == "Class_Weighted" else None,
                )
                training_started = time.perf_counter()
                classifier.fit(x_resampled, y_resampled)
                training_seconds = time.perf_counter() - training_started

                stage = "inference"
                inference_started = time.perf_counter()
                y_pred = classifier.predict(x_test)
                y_probability = classifier.predict_proba(x_test)
                inference_seconds = time.perf_counter() - inference_started
                if not np.array_equal(classifier.classes_, np.asarray(LABELS)):
                    raise RuntimeError(f"Unexpected classifier class order: {classifier.classes_}")

                stage = "metric_calculation"
                metrics, report, matrix = calculate_metrics(y_test, y_pred)
                prediction_path = Path(paths["predictions"]) / f"{method}_seed{seed}.npz"
                report_path = Path(paths["reports"]) / f"{method}_seed{seed}.json"
                matrix_path = Path(paths["confusion_matrices"]) / f"{method}_seed{seed}.npy"
                atomic_npz_save(
                    prediction_path,
                    y_true=y_test.astype(np.int64),
                    y_pred=np.asarray(y_pred, dtype=np.int64),
                    y_probability=np.asarray(y_probability, dtype=np.float64),
                    train_indices=np.asarray(cache["train_indices"], dtype=np.int64),
                    test_indices=np.asarray(cache["test_indices"], dtype=np.int64),
                )
                atomic_json_dump(report, report_path)
                atomic_npy_save(matrix.astype(np.int64), matrix_path)
                record = {
                    "status": "complete",
                    "method": method,
                    "seed": seed,
                    "experiment_signature_sha256": signature_hash,
                    "metrics": metrics,
                    "sampling_time_seconds": sampling_seconds,
                    "training_time_seconds": training_seconds,
                    "inference_time_seconds": inference_seconds,
                    "sampler": None if sampler is None else {"class": type(sampler).__name__, "parameters": sampler.get_params(deep=True)},
                    "classifier": {"class": type(classifier).__name__, "parameters": classifier.get_params(deep=True)},
                    "train_shape_before": list(x_train.shape),
                    "train_shape_after": list(x_resampled.shape),
                    "train_distribution_before": distribution(y_train),
                    "train_distribution_after": distribution(y_resampled),
                    "test_shape": list(x_test.shape),
                    "test_distribution": distribution(y_test),
                    "prediction_path": str(prediction_path.resolve()),
                    "prediction_sha256": sha256_file(prediction_path),
                    "classification_report_path": str(report_path.resolve()),
                    "confusion_matrix_path": str(matrix_path.resolve()),
                    "memory_after": memory_snapshot(),
                }
                atomic_json_dump(record, checkpoint_path)
                del x_resampled, y_resampled, classifier, y_pred, y_probability
            except Exception as error:
                failure = {
                    "status": "failed",
                    "method": method,
                    "seed": seed,
                    "experiment_signature_sha256": signature_hash,
                    "execution_stage": stage,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "memory_report": memory_snapshot(),
                }
                atomic_json_dump(failure, checkpoint_path)
                print(f"FAILED: {method}, seed={seed}, stage={stage}: {error}")
                method_failed = True
                break
            finally:
                collect_extension_results(paths["checkpoints"], paths["results"], experiment_signature)
        if method_failed:
            print(f"Stopped remaining seeds for failed configuration: {method}")
    return collect_extension_results(paths["checkpoints"], paths["results"], experiment_signature)


def merge_verified_results(existing_validation: Mapping[str, Any], extension: Mapping[str, Any], output_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    existing = existing_validation["payload"]["Dataset_A"]
    for balancing_name, models in existing.items():
        for model_name, record in models.items():
            rows.append(
                {
                    "family": "TF-IDF",
                    "model": model_name,
                    "balancing": balancing_name,
                    "configuration": f"{model_name} + {balancing_name}",
                    "metrics": {metric: {"mean": float(record[metric]["mean"]), "std": float(record[metric]["std"]), "runs": record[metric]["runs"]} for metric in REQUIRED_EXISTING_METRICS},
                }
            )
    for method, record in extension["configurations"].items():
        summary = record["summary"]
        if summary["completed_runs"] != len(SEEDS):
            raise RuntimeError(f"Cannot merge incomplete IndoBERTweet configuration {method}: {summary}")
        rows.append(
            {
                "family": "Frozen IndoBERTweet",
                "model": "Logistic Regression",
                "balancing": method,
                "configuration": f"IndoBERTweet + {method}",
                "metrics": summary["metrics"],
            }
        )
    if len(rows) != 49:
        raise RuntimeError(f"Expected 49 combined configurations, found {len(rows)}")
    rankings = {}
    for metric in ("f1", "f1_macro", "f1_negative", "f1_neutral", "f1_positive"):
        rankings[metric] = [row["configuration"] for row in sorted(rows, key=lambda item: item["metrics"][metric]["mean"], reverse=True)]
    payload = {
        "configuration_count": len(rows),
        "existing_configuration_count": 42,
        "indobertweet_configuration_count": 7,
        "existing_results_sha256": existing_validation["sha256"],
        "configurations": rows,
        "rankings": rankings,
        "interpretation_constraints": [
            "No statistical significance or equivalence claim is made.",
            "Weighted F1 is interpreted together with macro-F1 and classwise F1.",
            "Conclusions are limited to this dataset, historical split, and evaluated configurations.",
        ],
    }
    atomic_json_dump(payload, Path(output_path))
    return payload


def combined_frame(combined: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for record in combined["configurations"]:
        row = {
            "family": record["family"],
            "configuration": record["configuration"],
            "model": record["model"],
            "balancing": record["balancing"],
        }
        for metric, summary in record["metrics"].items():
            row[f"{metric}_mean"] = summary["mean"]
            row[f"{metric}_std"] = summary["std"]
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def paper_evidence_summary(combined: Mapping[str, Any]) -> dict[str, Any]:
    frame = combined_frame(combined)
    evidence: dict[str, Any] = {}
    for metric in ("f1", "f1_macro", "f1_negative", "f1_neutral", "f1_positive"):
        row = frame.loc[frame[f"{metric}_mean"].idxmax()]
        evidence[f"best_{metric}"] = {
            "configuration": row["configuration"],
            "mean": float(row[f"{metric}_mean"]),
            "std": float(row[f"{metric}_std"]),
        }
    indo = frame[frame["family"] == "Frozen IndoBERTweet"]
    tfidf = frame[frame["family"] == "TF-IDF"]
    best_indo = indo.loc[indo["f1_mean"].idxmax()]
    best_tfidf = tfidf.loc[tfidf["f1_mean"].idxmax()]
    evidence["best_indobertweet"] = {
        "configuration": best_indo["configuration"],
        "weighted_f1": float(best_indo["f1_mean"]),
        "macro_f1": float(best_indo["f1_macro_mean"]),
    }
    evidence["best_tfidf"] = {
        "configuration": best_tfidf["configuration"],
        "weighted_f1": float(best_tfidf["f1_mean"]),
        "macro_f1": float(best_tfidf["f1_macro_mean"]),
    }
    evidence["weighted_f1_difference_percentage_points_indobertweet_minus_tfidf"] = float(
        (best_indo["f1_mean"] - best_tfidf["f1_mean"]) * 100
    )
    return evidence


def generate_result_artifacts(
    combined: Mapping[str, Any],
    extension: Mapping[str, Any],
    paths: Mapping[str, Path],
    project_root: Path,
) -> dict[str, Any]:
    """Generate the requested tables and focused figures after all runs exist."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    figures_dir = Path(paths["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    frame = combined_frame(combined)
    indo = frame[frame["family"] == "Frozen IndoBERTweet"].copy()
    artifacts: dict[str, Any] = {"tables": {}, "figures": {}, "limitations": []}

    tables = {
        "indobertweet_7_results.csv": indo.sort_values("f1_mean", ascending=False),
        "all_49_weighted_f1.csv": frame[["family", "configuration", "f1_mean", "f1_std"]].sort_values("f1_mean", ascending=False),
        "all_49_macro_f1.csv": frame[["family", "configuration", "f1_macro_mean", "f1_macro_std"]].sort_values("f1_macro_mean", ascending=False),
        "all_49_classwise_f1.csv": frame[["family", "configuration", "f1_negative_mean", "f1_neutral_mean", "f1_positive_mean"]],
    }
    for filename, table in tables.items():
        destination = figures_dir / filename
        table.to_csv(destination, index=False)
        artifacts["tables"][filename] = str(destination.resolve())

    sns.set_theme(style="whitegrid")
    for metric, filename, title in (
        ("f1_mean", "ranked_weighted_f1_49.png", "Weighted F1 across 49 evaluated configurations"),
        ("f1_macro_mean", "ranked_macro_f1_49.png", "Macro-F1 across 49 evaluated configurations"),
    ):
        ranked = frame.sort_values(metric, ascending=True)
        fig, axis = plt.subplots(figsize=(11, 15))
        colors = ["#d97706" if family == "Frozen IndoBERTweet" else "#2563eb" for family in ranked["family"]]
        axis.barh(ranked["configuration"], ranked[metric], color=colors)
        axis.set_xlabel(metric.replace("_mean", "").replace("_", " ").upper())
        axis.set_title(title)
        axis.set_xlim(0, max(1.0, float(ranked[metric].max()) * 1.05))
        fig.tight_layout()
        destination = figures_dir / filename
        fig.savefig(destination, dpi=300, bbox_inches="tight")
        plt.close(fig)
        artifacts["figures"][filename] = str(destination.resolve())

    best_indo_row = indo.loc[indo["f1_mean"].idxmax()]
    best_method = str(best_indo_row["balancing"])
    matrices = []
    for seed in SEEDS:
        matrix_path = Path(paths["confusion_matrices"]) / f"{best_method}_seed{seed}.npy"
        if not matrix_path.exists():
            raise FileNotFoundError(f"Missing confusion-matrix evidence: {matrix_path}")
        matrices.append(np.load(matrix_path, allow_pickle=False))
    aggregated = np.sum(matrices, axis=0)
    fig, axis = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(aggregated, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=axis)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(f"Best frozen IndoBERTweet: {best_method}\nAggregated over three seeds")
    fig.tight_layout()
    destination = figures_dir / "best_indobertweet_confusion_matrix.png"
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)
    artifacts["figures"][destination.name] = str(destination.resolve())

    best_overall = frame.loc[frame["f1_mean"].idxmax()]
    if best_overall["family"] == "Frozen IndoBERTweet":
        destination_overall = figures_dir / "best_overall_confusion_matrix.png"
        destination_overall.write_bytes(destination.read_bytes())
        artifacts["figures"][destination_overall.name] = str(destination_overall.resolve())
    else:
        source = (
            Path(project_root)
            / "results"
            / "dataset_kesepakatan"
            / _safe_name(str(best_overall["model"]))
            / f"{best_overall['balancing']}_cm.png"
        )
        if source.exists():
            destination_overall = figures_dir / "best_overall_confusion_matrix_existing_evidence.png"
            destination_overall.write_bytes(source.read_bytes())
            artifacts["figures"][destination_overall.name] = str(destination_overall.resolve())
            artifacts["limitations"].append(
                "The best-overall TF-IDF confusion matrix is copied from existing saved evidence; per-seed predictions were not stored by the historical pipeline."
            )
        else:
            artifacts["limitations"].append(f"Best-overall confusion matrix source was unavailable: {source}")

    comparison_names = [
        str(best_indo_row["configuration"]),
        "Random Forest + ROS",
        "Random Forest + No_Balancing",
    ]
    comparison = frame[frame["configuration"].isin(comparison_names)].copy()
    if len(comparison) != 3:
        raise RuntimeError(f"Classwise comparison rows are incomplete: {comparison_names}")
    long_rows = []
    for _, row in comparison.iterrows():
        for class_name in LABEL_NAMES:
            long_rows.append(
                {
                    "configuration": row["configuration"],
                    "class": class_name,
                    "F1": row[f"f1_{class_name}_mean"],
                }
            )
    fig, axis = plt.subplots(figsize=(11, 6))
    sns.barplot(data=pd.DataFrame(long_rows), x="class", y="F1", hue="configuration", ax=axis)
    axis.set_ylim(0, 1)
    axis.set_title("Classwise F1 for the best IndoBERTweet and key Random Forest baselines")
    fig.tight_layout()
    destination = figures_dir / "classwise_f1_key_configurations.png"
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)
    artifacts["figures"][destination.name] = str(destination.resolve())

    stability_rows = []
    timing_rows = []
    for method, record in extension["configurations"].items():
        for run in record["runs"]:
            if run.get("status") != "complete":
                continue
            stability_rows.append({"method": method, "seed": run["seed"], "weighted_f1": run["metrics"]["f1"], "macro_f1": run["metrics"]["f1_macro"]})
            timing_rows.append({"method": method, "seed": run["seed"], "seconds": run["sampling_time_seconds"] + run["training_time_seconds"]})
    stability = pd.DataFrame(stability_rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False)
    sns.pointplot(data=stability, x="method", y="weighted_f1", hue="seed", errorbar=None, ax=axes[0])
    sns.pointplot(data=stability, x="method", y="macro_f1", hue="seed", errorbar=None, ax=axes[1])
    for axis, title in zip(axes, ("Weighted F1 stability", "Macro-F1 stability")):
        axis.tick_params(axis="x", rotation=35)
        axis.set_title(title)
        axis.set_ylim(0, 1)
    fig.tight_layout()
    destination = figures_dir / "indobertweet_three_seed_stability.png"
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)
    artifacts["figures"][destination.name] = str(destination.resolve())

    timing = pd.DataFrame(timing_rows)
    fig, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(data=timing, x="method", y="seconds", errorbar="sd", ax=axis)
    axis.tick_params(axis="x", rotation=35)
    axis.set_ylabel("Sampling + classifier training seconds")
    axis.set_title("Frozen IndoBERTweet linear-head training time (encoder extraction excluded)")
    fig.tight_layout()
    destination = figures_dir / "indobertweet_training_time.png"
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)
    artifacts["figures"][destination.name] = str(destination.resolve())

    atomic_json_dump(artifacts, figures_dir / "artifact_manifest.json")
    return artifacts
