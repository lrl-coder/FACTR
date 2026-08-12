#!/usr/bin/env python
"""Serve a FACTR checkpoint over a MessagePack/ZMQ REP socket."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import io
import json
import logging
import os
from pathlib import Path
import signal
import threading
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import msgpack
import numpy as np
import yaml
import zmq


MODEL_NAME = "FACTR"
ACTION_STEPS = 16
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "FACTR_CHECKPOINT",
        PROJECT_ROOT / "checkpoints" / "flip_box_force",
    )
)
STATE_KEY = "observation.state"
WRENCH_KEY = "observation.wrench_compensated"
WRIST_KEY = "observation.images.wrist"
THIRD_VIEW_KEY = "observation.images.third_view"
TASK_KEY = "task"
OBSERVATION_KEYS = (
    STATE_KEY,
    WRENCH_KEY,
    WRIST_KEY,
    THIRD_VIEW_KEY,
    TASK_KEY,
)
USED_OBSERVATION_KEYS = (THIRD_VIEW_KEY, WRIST_KEY, WRENCH_KEY)
ACTION_REPRESENTATION = (
    "7 absolute desired left-arm joint positions (dataset order) + "
    "1 normalized absolute left-gripper command [0,1]"
)


class ContractError(ValueError):
    """Raised when a wire request violates the public service contract."""


class ConfigurationError(ValueError):
    """Raised when checkpoint artifacts disagree with one another."""


class BackendOutputError(RuntimeError):
    """Raised when a policy backend returns an invalid action chunk."""


class NumpyMsgpackCodec:
    """MessagePack codec matching mock_client.py without pickle support."""

    NDARRAY_MARKER = "__ndarray_class__"
    NPY_KEY = "as_npy"

    @classmethod
    def dumps(cls, value: Any) -> bytes:
        return msgpack.packb(value, default=cls._encode, use_bin_type=True)

    @classmethod
    def loads(cls, payload: bytes) -> Any:
        if not isinstance(payload, bytes):
            raise TypeError("MessagePack payload must be bytes")
        return msgpack.unpackb(payload, object_hook=cls._decode, raw=False)

    @classmethod
    def _encode(cls, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("object-dtype arrays are forbidden")
            output = io.BytesIO()
            np.save(output, value, allow_pickle=False)
            return {
                cls.NDARRAY_MARKER: True,
                cls.NPY_KEY: output.getvalue(),
            }
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(
            "Cannot MessagePack-encode {}".format(type(value).__name__)
        )

    @classmethod
    def _decode(cls, value: Any) -> Any:
        if not isinstance(value, dict) or cls.NDARRAY_MARKER not in value:
            return value
        if value.get(cls.NDARRAY_MARKER) is not True:
            raise ValueError("invalid NumPy array marker")
        if set(value) != {cls.NDARRAY_MARKER, cls.NPY_KEY}:
            raise ValueError("invalid NumPy array envelope")
        npy = value.get(cls.NPY_KEY)
        if not isinstance(npy, bytes):
            raise TypeError("NumPy as_npy payload must be bytes")
        array = np.load(io.BytesIO(npy), allow_pickle=False)
        if not isinstance(array, np.ndarray) or array.dtype.hasobject:
            raise TypeError("decoded value is not a non-object NumPy array")
        return array


def validate_observation(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError("data.observation must be a dictionary")

    missing = sorted(set(OBSERVATION_KEYS) - set(value))
    extra = sorted(set(value) - set(OBSERVATION_KEYS))
    if missing:
        raise ContractError("missing observation fields: {}".format(", ".join(missing)))
    if extra:
        raise ContractError("unknown observation fields: {}".format(", ".join(extra)))

    expected_arrays = {
        STATE_KEY: ((8,), np.dtype(np.float32)),
        WRENCH_KEY: ((6,), np.dtype(np.float32)),
        WRIST_KEY: ((480, 640, 3), np.dtype(np.uint8)),
        THIRD_VIEW_KEY: ((480, 640, 3), np.dtype(np.uint8)),
    }
    validated: Dict[str, Any] = {}
    for key, (shape, dtype) in expected_arrays.items():
        array = value[key]
        if not isinstance(array, np.ndarray):
            raise ContractError("{} must be a NumPy array".format(key))
        if array.shape != shape:
            raise ContractError(
                "{} must have shape {}, got {}".format(key, shape, array.shape)
            )
        if array.dtype != dtype:
            raise ContractError(
                "{} must have dtype {}, got {}".format(key, dtype, array.dtype)
            )
        if np.issubdtype(dtype, np.floating) and not np.all(np.isfinite(array)):
            raise ContractError("{} must contain only finite values".format(key))
        validated[key] = np.ascontiguousarray(array)

    task = value[TASK_KEY]
    if not isinstance(task, str) or not task.strip():
        raise ContractError("task must be a non-empty string")
    validated[TASK_KEY] = task.strip()
    return validated


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ConfigurationError("{} must contain a YAML mapping".format(path))
    return value


def _checkpoint_sort_key(path: Path) -> Tuple[int, float]:
    stem = path.stem
    try:
        step = int(stem.rsplit("_", 1)[-1])
    except ValueError:
        step = -1
    return step, path.stat().st_mtime


def resolve_checkpoint_file(checkpoint: Path) -> Tuple[Path, Path]:
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.is_file():
        root = checkpoint.parent.parent if checkpoint.parent.name == "rollout" else checkpoint.parent
        return checkpoint, root
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)

    preferred = (
        checkpoint / "rollout" / "latest_ckpt.ckpt",
        checkpoint / "latest_ckpt.ckpt",
    )
    for candidate in preferred:
        if candidate.is_file():
            return candidate, checkpoint

    candidates = list(checkpoint.glob("ckpt_*.ckpt"))
    candidates.extend((checkpoint / "rollout").glob("ckpt_*.ckpt"))
    if not candidates:
        raise FileNotFoundError("no FACTR .ckpt file found under {}".format(checkpoint))
    return max(candidates, key=_checkpoint_sort_key), checkpoint


def resolve_config_files(
    checkpoint_root: Path,
    checkpoint_file: Path,
    config: Optional[Path],
) -> Tuple[Path, Path, Optional[Path]]:
    if config is None:
        config_dir = checkpoint_root / "rollout"
        if not config_dir.is_dir():
            config_dir = checkpoint_file.parent
        agent_path = config_dir / "agent_config.yaml"
    else:
        config_path = config.expanduser().resolve()
        if config_path.is_dir():
            config_dir = config_path / "rollout" if (config_path / "rollout").is_dir() else config_path
            agent_path = config_dir / "agent_config.yaml"
        else:
            agent_path = config_path
            config_dir = config_path.parent

    rollout_path = config_dir / "rollout_config.yaml"
    hydra_path = checkpoint_root / ".hydra" / "config.yaml"
    return agent_path, rollout_path, hydra_path if hydra_path.is_file() else None


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,):
        raise ConfigurationError("{} must have shape ({},), got {}".format(name, size, array.shape))
    if not np.all(np.isfinite(array)):
        raise ConfigurationError("{} contains non-finite values".format(name))
    return array


def _read_action_names(rollout: Mapping[str, Any], action_dim: int) -> Tuple[str, ...]:
    configured_names = rollout.get("action_config", {}).get("names")
    if configured_names is not None:
        if not isinstance(configured_names, list) or len(configured_names) != action_dim:
            raise ConfigurationError(
                "checkpoint action_config.names must contain {} entries".format(action_dim)
            )
        names = tuple(str(name) for name in configured_names)
        if any(not name for name in names):
            raise ConfigurationError("checkpoint action_config.names contains an empty name")
        return names

    discovered = []
    for source in rollout.get("source_datasets", []):
        if not isinstance(source, dict) or "path" not in source:
            continue
        info_path = Path(str(source["path"])) / "meta" / "info.json"
        if not info_path.is_file():
            continue
        with info_path.open("r", encoding="utf-8") as stream:
            info = json.load(stream)
        action_key = rollout.get("action_config", {}).get("action_key", "action")
        names = info.get("features", {}).get(action_key, {}).get("names")
        if isinstance(names, list) and len(names) == action_dim:
            discovered.append(tuple(str(name) for name in names))
    if discovered and any(names != discovered[0] for names in discovered[1:]):
        raise ConfigurationError("source datasets disagree on action joint order")
    if discovered:
        return discovered[0]
    return tuple("action_{}".format(index) for index in range(action_dim))


@dataclass(frozen=True)
class PolicySpec:
    checkpoint_file: Path
    checkpoint_root: Path
    agent_config_path: Path
    rollout_config_path: Path
    hydra_config_path: Optional[Path]
    action_dim: int
    action_chunk: int
    obs_dim: int
    camera_keys: Tuple[str, ...]
    image_history: int
    image_size: int
    model_image_size: int
    jpeg_quality: int
    transform_name: str
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    action_names: Tuple[str, ...]

    @classmethod
    def load(cls, checkpoint: Path, config: Optional[Path] = None) -> "PolicySpec":
        checkpoint_file, checkpoint_root = resolve_checkpoint_file(checkpoint)
        agent_path, rollout_path, hydra_path = resolve_config_files(
            checkpoint_root, checkpoint_file, config
        )
        agent = _load_yaml(agent_path)
        rollout = _load_yaml(rollout_path)
        hydra_config = _load_yaml(hydra_path) if hydra_path is not None else {}

        target = agent.get("_target_")
        if target != "factr.models.action_transformer.TransformerAgent":
            raise ConfigurationError("unsupported FACTR agent target: {!r}".format(target))

        action_dim = int(agent["ac_dim"])
        action_chunk = int(agent["ac_chunk"])
        obs_dim = int(agent["odim"])
        n_cams = int(agent["n_cams"])
        image_history = int(agent.get("imgs_per_cam", 1))
        camera_keys = tuple(rollout.get("obs_config", {}).get("camera_keys", []))
        obs_keys = tuple(rollout.get("obs_config", {}).get("obs_keys", []))
        rollout_action_dim = int(rollout.get("action_config", {}).get("action_dim", -1))
        state_layout = rollout.get("obs_config", {}).get("state_layout", {})

        if action_dim != rollout_action_dim:
            raise ConfigurationError(
                "agent ac_dim {} != rollout action_dim {}".format(action_dim, rollout_action_dim)
            )
        if n_cams != len(camera_keys):
            raise ConfigurationError(
                "agent n_cams {} != rollout camera count {}".format(n_cams, len(camera_keys))
            )
        if camera_keys != (THIRD_VIEW_KEY, WRIST_KEY):
            raise ConfigurationError("unexpected camera order: {}".format(camera_keys))
        if obs_keys != (WRENCH_KEY,):
            raise ConfigurationError("this service requires force-only obs_keys, got {}".format(obs_keys))
        if state_layout.get("force") != [0, 6] or obs_dim != 6:
            raise ConfigurationError("checkpoint does not use a 6D wrench state token")
        if image_history != 1:
            raise ConfigurationError(
                "checkpoint requires {} image frames per camera; this service contract supplies one".format(
                    image_history
                )
            )
        if not bool(rollout.get("normalized", False)):
            raise ConfigurationError("checkpoint rollout_config must contain normalized training statistics")

        norm_stats = rollout.get("norm_stats", {})
        state_mean = _finite_vector(norm_stats.get("state", {}).get("mean"), obs_dim, "state mean")
        state_std = _finite_vector(norm_stats.get("state", {}).get("std"), obs_dim, "state std")
        action_mean = _finite_vector(norm_stats.get("action", {}).get("mean"), action_dim, "action mean")
        action_std = _finite_vector(norm_stats.get("action", {}).get("std"), action_dim, "action std")
        if np.any(state_std <= 0) or np.any(action_std <= 0):
            raise ConfigurationError("normalization standard deviations must be positive")

        model_image_size = int(agent.get("features", {}).get("model", {}).get("img_size", 224))
        image_size = int(rollout.get("image_size", 256))
        transform_name = str(hydra_config.get("test_transform", "preproc"))
        if transform_name != "preproc":
            raise ConfigurationError(
                "deterministic serving requires the checkpoint test_transform=preproc, got {!r}".format(
                    transform_name
                )
            )

        return cls(
            checkpoint_file=checkpoint_file,
            checkpoint_root=checkpoint_root,
            agent_config_path=agent_path,
            rollout_config_path=rollout_path,
            hydra_config_path=hydra_path,
            action_dim=action_dim,
            action_chunk=action_chunk,
            obs_dim=obs_dim,
            camera_keys=camera_keys,
            image_history=image_history,
            image_size=image_size,
            model_image_size=model_image_size,
            jpeg_quality=90,
            transform_name=transform_name,
            state_mean=state_mean,
            state_std=state_std,
            action_mean=action_mean,
            action_std=action_std,
            action_names=_read_action_names(rollout, action_dim),
        )


class FakePolicyBackend:
    """Deterministic backend used by --dry-run and protocol tests."""

    def __init__(
        self,
        action_dim: int = 8,
        action_steps: int = ACTION_STEPS,
        checkpoint: str = "dry-run",
        action_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.action_dim = int(action_dim)
        self.action_steps = int(action_steps)
        self.checkpoint = checkpoint
        self.action_names = tuple(action_names or ["action_{}".format(i) for i in range(action_dim)])
        self.dry_run = True
        self.episode_state = []
        self.reset_count = 0

    def predict(self, observation: Mapping[str, Any]) -> np.ndarray:
        self.episode_state.append(observation[TASK_KEY])
        return np.linspace(
            0.0,
            1.0,
            self.action_steps * self.action_dim,
            dtype=np.float32,
        ).reshape(self.action_steps, self.action_dim)

    def reset(self) -> None:
        self.episode_state.clear()
        self.reset_count += 1


class FactrPolicyBackend:
    """Single-load FACTR inference backend using checkpoint-native transforms."""

    def __init__(self, spec: PolicySpec, device: str, dtype: str) -> None:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import torch
        import hydra
        from omegaconf import OmegaConf

        from factr import misc
        from factr.replay_buffer import _img_to_tensor
        from factr.transforms import get_transform_by_name
        from robobuf.buffers import ObsWrapper
        from scripts.convert_lerobot21_force_to_factr import encode_bgr_frame

        self._torch = torch
        self._img_to_tensor = _img_to_tensor
        self._ObsWrapper = ObsWrapper
        self._encode_bgr_frame = encode_bgr_frame
        self._transform = get_transform_by_name(
            spec.transform_name, size=spec.model_image_size
        )
        self.spec = spec
        self.action_dim = spec.action_dim
        self.action_steps = spec.action_chunk
        self.checkpoint = str(spec.checkpoint_file)
        self.action_names = spec.action_names
        self.dry_run = False
        self.device = self._resolve_device(device)
        self.dtype_name, self._autocast_dtype = self._resolve_dtype(dtype)

        agent_config = OmegaConf.load(spec.agent_config_path)
        model = hydra.utils.instantiate(agent_config)
        try:
            checkpoint = torch.load(
                spec.checkpoint_file,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except TypeError:
            checkpoint = torch.load(
                spec.checkpoint_file,
                map_location="cpu",
                weights_only=False,
            )
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ConfigurationError("checkpoint is missing the model state dict")
        model.load_state_dict(checkpoint["model"], strict=True)
        global_step = int(checkpoint.get("global_step", -1))
        if global_step < 0:
            raise ConfigurationError("checkpoint is missing a valid global_step")
        max_step = int(agent_config.curriculum.max_step)
        if global_step > max_step:
            raise ConfigurationError(
                "checkpoint global_step {} exceeds curriculum max_step {}".format(
                    global_step, max_step
                )
            )
        misc.GLOBAL_STEP = global_step
        self.global_step = global_step

        del checkpoint
        gc.collect()
        self.model = model.eval().to(self.device)

    def _resolve_device(self, value: str) -> Any:
        torch = self._torch
        if value == "auto":
            value = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = torch.device(value)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ConfigurationError("CUDA was requested but is not available")
        return device

    def _resolve_dtype(self, value: str) -> Tuple[str, Optional[Any]]:
        aliases = {
            "float32": ("float32", None),
            "fp32": ("float32", None),
            "float16": ("float16", self._torch.float16),
            "fp16": ("float16", self._torch.float16),
            "bfloat16": ("bfloat16", self._torch.bfloat16),
            "bf16": ("bfloat16", self._torch.bfloat16),
        }
        if value not in aliases:
            raise ConfigurationError("unsupported dtype {!r}".format(value))
        name, autocast_dtype = aliases[value]
        if self.device.type != "cuda" and autocast_dtype is not None:
            raise ConfigurationError("float16/bfloat16 autocast requires a CUDA device")
        return name, autocast_dtype

    def _prepare_images(self, observation: Mapping[str, Any]) -> Dict[str, Any]:
        encoded_obs: Dict[str, Any] = {"state": np.empty((self.spec.obs_dim,), dtype=np.float32)}
        for cam_index, key in enumerate(self.spec.camera_keys):
            rgb = observation[key]
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            encoded_obs["enc_cam_{}".format(cam_index)] = self._encode_bgr_frame(
                bgr,
                self.spec.image_size,
                self.spec.jpeg_quality,
            )

        wrapped = self._ObsWrapper(
            encoded_obs,
            H=self.spec.image_size,
            W=self.spec.image_size,
        )
        images = {}
        for cam_index in range(len(self.spec.camera_keys)):
            decoded_rgb = wrapped.image(cam_index)[None]
            tensor = self._img_to_tensor(decoded_rgb)
            tensor = self._transform(tensor)
            images["cam{}".format(cam_index)] = tensor.unsqueeze(0).to(self.device)
        return images

    def _autocast(self) -> Any:
        if self.device.type == "cuda" and self._autocast_dtype is not None:
            return self._torch.autocast(
                device_type="cuda",
                dtype=self._autocast_dtype,
            )
        return nullcontext()

    def predict(self, observation: Mapping[str, Any]) -> np.ndarray:
        wrench = observation[WRENCH_KEY]
        normalized_wrench = (
            (wrench - self.spec.state_mean) / self.spec.state_std
        ).astype(np.float32, copy=False)
        state_tensor = self._torch.from_numpy(normalized_wrench).unsqueeze(0).to(self.device)
        images = self._prepare_images(observation)

        with self._torch.inference_mode():
            with self._autocast():
                normalized_actions = self.model.get_actions(images, state_tensor)
        if normalized_actions.ndim != 3 or normalized_actions.shape[0] != 1:
            raise BackendOutputError(
                "FACTR returned shape {}, expected [1,T,D]".format(
                    tuple(normalized_actions.shape)
                )
            )
        normalized_np = normalized_actions[0].float().cpu().numpy()
        actions = normalized_np * self.spec.action_std + self.spec.action_mean
        return np.ascontiguousarray(actions, dtype=np.float32)

    def reset(self) -> None:
        # The selected TransformerAgent has no recurrent or temporal input state.
        return None


class PolicyApplication:
    """Endpoint router independent of the ZMQ transport."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.episode_request_count = 0

    @property
    def observation_usage(self) -> Dict[str, str]:
        return {
            THIRD_VIEW_KEY: "direct: current RGB frame -> FACTR cam0; no history",
            WRIST_KEY: "direct: current RGB frame -> FACTR cam1; no history",
            WRENCH_KEY: "combined_into_state: normalized -> sole 6D low-dimensional force token",
            STATE_KEY: "checkpoint_unused: validated but not passed to the model",
            TASK_KEY: "checkpoint_unused: validated; checkpoint has no language encoder",
        }

    def _validate_request(self, request: Any) -> str:
        if not isinstance(request, dict):
            raise ContractError("request must be a dictionary")
        extra = set(request) - {"endpoint", "data"}
        if extra:
            raise ContractError("unknown request fields: {}".format(", ".join(sorted(extra))))
        endpoint = request.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint:
            raise ContractError("endpoint must be a non-empty string")
        return endpoint

    def _validate_options_data(self, data: Any, required_observation: bool) -> Optional[Dict[str, Any]]:
        if not isinstance(data, dict):
            raise ContractError("data must be a dictionary")
        expected = {"options", "observation"} if required_observation else {"options"}
        missing = expected - set(data)
        extra = set(data) - expected
        if missing:
            raise ContractError("missing data fields: {}".format(", ".join(sorted(missing))))
        if extra:
            raise ContractError("unknown data fields: {}".format(", ".join(sorted(extra))))
        if data["options"] is not None:
            raise ContractError("data.options must be null")
        return validate_observation(data["observation"]) if required_observation else None

    def _format_actions(self, value: Any) -> np.ndarray:
        actions = np.asarray(value)
        if actions.ndim != 2:
            raise BackendOutputError(
                "policy returned rank-{} actions; expected [T,D]".format(actions.ndim)
            )
        if actions.shape[1] != self.backend.action_dim:
            raise BackendOutputError(
                "policy returned action_dim {}, expected {}".format(
                    actions.shape[1], self.backend.action_dim
                )
            )
        if actions.shape[0] < ACTION_STEPS:
            raise BackendOutputError(
                "policy returned {} action steps; at least {} are required".format(
                    actions.shape[0], ACTION_STEPS
                )
            )
        actions = np.ascontiguousarray(actions[:ACTION_STEPS], dtype=np.float32)
        if not np.all(np.isfinite(actions)):
            raise BackendOutputError("policy returned non-finite actions")
        return actions

    def handle(self, request: Any) -> Any:
        endpoint = self._validate_request(request)
        if endpoint == "ping":
            if set(request) != {"endpoint"}:
                raise ContractError("ping accepts only the endpoint field")
            return {
                "status": "ok",
                "model_name": MODEL_NAME,
                "checkpoint": self.backend.checkpoint,
                "action_shape": [ACTION_STEPS, self.backend.action_dim],
                "dry_run": bool(self.backend.dry_run),
            }

        if endpoint == "reset":
            if "data" in request:
                self._validate_options_data(request["data"], required_observation=False)
            self.backend.reset()
            self.episode_request_count = 0
            return {"status": "ok", "reset": True}

        if endpoint != "get_action":
            raise ContractError("unknown endpoint: {}".format(endpoint))
        if set(request) != {"endpoint", "data"}:
            raise ContractError("get_action requires endpoint and data")

        observation = self._validate_options_data(
            request["data"], required_observation=True
        )
        assert observation is not None
        started = time.perf_counter()
        actions = self._format_actions(self.backend.predict(observation))
        inference_ms = (time.perf_counter() - started) * 1000.0
        self.episode_request_count += 1
        info = {
            "model_name": MODEL_NAME,
            "inference_ms": float(inference_ms),
            "action_shape": list(actions.shape),
            "action_representation": ACTION_REPRESENTATION,
            "checkpoint": self.backend.checkpoint,
            "used_observation_keys": list(USED_OBSERVATION_KEYS),
            "observation_field_usage": self.observation_usage,
            "action_names": list(self.backend.action_names),
            "history": "none; current frame/wrench only",
            "episode_request_index": self.episode_request_count,
            "dry_run": bool(self.backend.dry_run),
        }
        return [actions, info]

    def warmup(self, count: int) -> None:
        observation = {
            STATE_KEY: np.zeros((8,), dtype=np.float32),
            WRENCH_KEY: np.zeros((6,), dtype=np.float32),
            WRIST_KEY: np.zeros((480, 640, 3), dtype=np.uint8),
            THIRD_VIEW_KEY: np.zeros((480, 640, 3), dtype=np.uint8),
            TASK_KEY: "FACTR startup warmup",
        }
        for _ in range(count):
            self.handle(
                {
                    "endpoint": "get_action",
                    "data": {"observation": observation, "options": None},
                }
            )
        self.backend.reset()
        self.episode_request_count = 0


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        value = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", "log"),
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            value.update(fields)
        if record.exc_info:
            value["exception"] = self.formatException(record.exc_info)
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def configure_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("factr_policy_server")
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, level: int, event: str, message: str, **fields: Any) -> None:
    logger.log(level, message, extra={"event": event, "fields": fields})


class ZmqPolicyServer:
    def __init__(
        self,
        application: PolicyApplication,
        host: str,
        port: int,
        logger: logging.Logger,
    ) -> None:
        self.application = application
        self.host = host
        self.port = port
        self.logger = logger
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.bound_endpoint: Optional[str] = None

    def stop(self) -> None:
        self.stop_event.set()

    def _error_response(self, exc: BaseException) -> Dict[str, str]:
        return {"error": "{}: {}".format(type(exc).__name__, exc)}

    def serve_forever(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 250)
        bind_endpoint = "tcp://{}:{}".format(
            self.host, "*" if self.port == 0 else self.port
        )
        try:
            socket.bind(bind_endpoint)
            self.bound_endpoint = socket.getsockopt_string(zmq.LAST_ENDPOINT)
            self.ready_event.set()
            log_event(
                self.logger,
                logging.INFO,
                "server_started",
                "FACTR policy server is ready",
                endpoint=self.bound_endpoint,
                checkpoint=self.application.backend.checkpoint,
                dry_run=bool(self.application.backend.dry_run),
            )
            while not self.stop_event.is_set():
                try:
                    payload = socket.recv()
                except zmq.Again:
                    continue
                except zmq.ZMQError:
                    if self.stop_event.is_set():
                        break
                    raise

                endpoint = "unknown"
                try:
                    request = NumpyMsgpackCodec.loads(payload)
                    if isinstance(request, dict):
                        endpoint = str(request.get("endpoint", "unknown"))
                    response = self.application.handle(request)
                    log_event(
                        self.logger,
                        logging.INFO,
                        "request_completed",
                        "request completed",
                        endpoint=endpoint,
                    )
                except Exception as exc:
                    response = self._error_response(exc)
                    log_event(
                        self.logger,
                        logging.WARNING,
                        "request_failed",
                        str(exc),
                        endpoint=endpoint,
                        error_type=type(exc).__name__,
                    )

                try:
                    socket.send(NumpyMsgpackCodec.dumps(response))
                except Exception as exc:
                    fallback = self._error_response(exc)
                    socket.send(NumpyMsgpackCodec.dumps(fallback))
                    log_event(
                        self.logger,
                        logging.ERROR,
                        "response_encoding_failed",
                        str(exc),
                        endpoint=endpoint,
                        error_type=type(exc).__name__,
                    )
        finally:
            self.ready_event.set()
            socket.close(linger=0)
            context.term()
            log_event(
                self.logger,
                logging.INFO,
                "server_stopped",
                "FACTR policy server stopped",
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve one FACTR checkpoint over MessagePack/ZMQ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="rollout config directory or resolved agent_config.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16", "fp32", "fp16", "bf16"),
        default="float16",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate protocol/config and return deterministic fake actions without loading torch",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.host.strip():
        raise ValueError("--host must not be empty")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in [1, 65535]")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    logger = configure_logging(args.log_level)

    spec = PolicySpec.load(args.checkpoint, args.config)
    if args.dry_run:
        backend: Any = FakePolicyBackend(
            action_dim=spec.action_dim,
            action_steps=spec.action_chunk,
            checkpoint=str(spec.checkpoint_file),
            action_names=spec.action_names,
        )
    else:
        backend = FactrPolicyBackend(spec, args.device, args.dtype)

    log_event(
        logger,
        logging.INFO,
        "policy_configured",
        "FACTR policy configuration validated",
        checkpoint=str(spec.checkpoint_file),
        action_dim=spec.action_dim,
        action_chunk=spec.action_chunk,
        obs_dim=spec.obs_dim,
        cameras=list(spec.camera_keys),
        image_history=spec.image_history,
        image_size=spec.image_size,
        model_image_size=spec.model_image_size,
        jpeg_quality=spec.jpeg_quality,
        transform=spec.transform_name,
        global_step=getattr(backend, "global_step", None),
        dry_run=bool(backend.dry_run),
    )
    if not args.dry_run and backend.global_step <= 1:
        log_event(
            logger,
            logging.WARNING,
            "smoke_checkpoint",
            "checkpoint has only one optimizer step and is not deployment-ready",
            global_step=backend.global_step,
            checkpoint=backend.checkpoint,
        )

    application = PolicyApplication(backend)
    if args.warmup:
        started = time.perf_counter()
        application.warmup(args.warmup)
        log_event(
            logger,
            logging.INFO,
            "warmup_completed",
            "startup warmup completed",
            iterations=args.warmup,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    server = ZmqPolicyServer(application, args.host, args.port, logger)

    def stop_server(signum: int, _frame: Any) -> None:
        log_event(
            logger,
            logging.INFO,
            "signal_received",
            "shutdown signal received",
            signal=signum,
        )
        server.stop()

    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, stop_server)
    try:
        server.serve_forever()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
