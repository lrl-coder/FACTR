#!/usr/bin/env python
"""Convert a LeRobot parquet dataset into FACTR's robobuf pickle format."""

import argparse
import json
import pickle
from io import BytesIO
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

try:
    import cv2
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install it in the FACTR env with:\n"
        "  python -m pip install pyarrow\n"
        "OpenCV is also required by FACTR/robobuf."
    ) from exc

from robobuf.buffers import ObsWrapper, ReplayBuffer, Transition


STATE_KEY = "observation.state"
ACTION_KEY = "action"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot parquet episodes to FACTR robobuf data."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(
            "/root/autodl-fs/force_vla_data/data_lerobot/"
            "flexiv_pump_1bottle_inputForce"
        ),
        help="LeRobot dataset directory containing meta/ and data/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/root/autodl-fs/force_vla_data/processed_factr/"
            "flexiv_pump_1bottle_inputForce_force"
        ),
        help="Directory where buf.pkl and rollout_config.yaml will be written.",
    )
    parser.add_argument(
        "--image-keys",
        nargs="+",
        default=["observation.image", "observation.wrist_image"],
        help="LeRobot image columns to export as FACTR cameras.",
    )
    parser.add_argument(
        "--episode-indices",
        nargs="*",
        type=int,
        default=None,
        help="Optional explicit episode indices to convert.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--obs-mode",
        choices=["force", "state", "proprio"],
        default="force",
        help=(
            "Low-dimensional observation exported to FACTR obs.state. "
            "'force' uses observation.state[7:13], matching FACTR's force token; "
            "'state' uses all 13 dims; 'proprio' uses observation.state[0:7]."
        ),
    )
    parser.add_argument("--no-normalize", action="store_true")
    return parser.parse_args()


def load_info(dataset_dir):
    info_path = dataset_dir / "meta" / "info.json"
    with info_path.open("r") as f:
        return json.load(f)


def episode_files(dataset_dir, info, episode_indices):
    if episode_indices is None:
        total = int(info["total_episodes"])
        episode_indices = list(range(total))

    data_pattern = info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    chunk_size = int(info.get("chunks_size", 1000))
    files = []
    for episode_index in episode_indices:
        rel = data_pattern.format(
            episode_chunk=episode_index // chunk_size,
            episode_index=episode_index,
        )
        path = dataset_dir / rel
        if not path.exists():
            raise FileNotFoundError(path)
        files.append(path)
    return files


def image_to_rgb_array(image_value):
    if isinstance(image_value, dict):
        if image_value.get("bytes") is not None:
            image_value = image_value["bytes"]
        elif image_value.get("path") is not None:
            image_value = Path(image_value["path"]).read_bytes()

    if isinstance(image_value, (bytes, bytearray, memoryview)):
        with Image.open(BytesIO(bytes(image_value))) as img:
            return np.asarray(img.convert("RGB"))

    arr = np.asarray(image_value)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image array, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def encode_factr_image(image_value, image_size, jpeg_quality):
    rgb = image_to_rgb_array(image_value)
    if rgb.shape[:2] != (image_size, image_size):
        rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    bgr = rgb[:, :, ::-1]
    ok, encoded = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    )
    if not ok:
        raise ValueError("Failed to encode image as JPEG.")
    return encoded


def gaussian_stats(arrays):
    data = np.concatenate(arrays, axis=0).astype(np.float32)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0] = 1e-17
    return mean, std


def stats_to_yaml(mean, std):
    return {
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
    }


def select_obs(states, obs_mode):
    if obs_mode == "force":
        return states[:, 7:13]
    if obs_mode == "proprio":
        return states[:, 0:7]
    if obs_mode == "state":
        return states
    raise ValueError(f"Unknown obs_mode: {obs_mode}")


def obs_layout(obs_mode):
    if obs_mode == "force":
        return {"force": [0, 6], "source_in_observation.state": [7, 13]}
    if obs_mode == "proprio":
        return {"proprio": [0, 7], "source_in_observation.state": [0, 7]}
    if obs_mode == "state":
        return {
            "proprio": [0, 7],
            "force": [7, 13],
            "source_in_observation.state": [0, 13],
        }
    raise ValueError(f"Unknown obs_mode: {obs_mode}")


def read_episode(path, image_keys, image_size, jpeg_quality, obs_mode):
    columns = [ACTION_KEY, STATE_KEY] + list(image_keys)
    table = pq.read_table(path, columns=columns)
    rows = table.to_pylist()

    states = np.asarray([row[STATE_KEY] for row in rows], dtype=np.float32)
    obs = select_obs(states, obs_mode)
    actions = np.asarray([row[ACTION_KEY] for row in rows], dtype=np.float32)

    enc_images_by_cam = []
    for image_key in image_keys:
        enc_images_by_cam.append(
            [
                encode_factr_image(row[image_key], image_size, jpeg_quality)
                for row in rows
            ]
        )
    return obs, actions, enc_images_by_cam


def build_buffer(episodes, state_mean, state_std, action_mean, action_std, normalize):
    buffer = ReplayBuffer()
    for episode in tqdm(episodes, desc="Building robobuf"):
        states, actions, enc_images_by_cam = episode
        if normalize:
            states = (states - state_mean) / state_std
            actions = (actions - action_mean) / action_std

        for step_idx in range(len(states)):
            obs = {"state": states[step_idx].astype(np.float32)}
            for cam_idx, enc_images in enumerate(enc_images_by_cam):
                obs[f"enc_cam_{cam_idx}"] = enc_images[step_idx]

            transition = Transition(
                obs=ObsWrapper(obs),
                action=actions[step_idx].astype(np.float32),
                reward=float(step_idx == len(states) - 1),
            )
            buffer.add(transition, is_first=(step_idx == 0))
    return buffer


def main():
    args = parse_args()
    info = load_info(args.dataset_dir)
    image_keys = list(args.image_keys)
    normalize = not args.no_normalize

    state_shape = info["features"][STATE_KEY]["shape"]
    action_shape = info["features"][ACTION_KEY]["shape"]
    if state_shape != [13]:
        raise ValueError(f"Expected {STATE_KEY} shape [13], got {state_shape}")
    if action_shape != [7]:
        raise ValueError(f"Expected {ACTION_KEY} shape [7], got {action_shape}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_files = episode_files(args.dataset_dir, info, args.episode_indices)

    episodes = []
    all_states = []
    all_actions = []
    for path in tqdm(parquet_files, desc="Reading LeRobot episodes"):
        obs, actions, enc_images_by_cam = read_episode(
            path, image_keys, args.image_size, args.jpeg_quality, args.obs_mode
        )
        episodes.append((obs, actions, enc_images_by_cam))
        all_states.append(obs)
        all_actions.append(actions)

    state_mean, state_std = gaussian_stats(all_states)
    action_mean, action_std = gaussian_stats(all_actions)
    if not normalize:
        state_mean = np.zeros_like(state_mean)
        state_std = np.ones_like(state_std)
        action_mean = np.zeros_like(action_mean)
        action_std = np.ones_like(action_std)

    buffer = build_buffer(
        episodes,
        state_mean,
        state_std,
        action_mean,
        action_std,
        normalize=normalize,
    )
    with (args.output_dir / "buf.pkl").open("wb") as f:
        pickle.dump(buffer.to_traj_list(), f)

    rollout_config = {
        "source_dataset": str(args.dataset_dir),
        "obs_config": {
            "state_key": STATE_KEY,
            "obs_mode": args.obs_mode,
            "camera_keys": image_keys,
            "state_layout": obs_layout(args.obs_mode),
        },
        "action_config": {
            "action_key": ACTION_KEY,
            "action_dim": 7,
        },
        "norm_stats": {
            "state": stats_to_yaml(state_mean, state_std),
            "action": stats_to_yaml(action_mean, action_std),
        },
        "normalized": normalize,
        "image_size": args.image_size,
        "num_episodes": len(episodes),
        "num_frames": int(sum(len(ep[0]) for ep in episodes)),
    }
    with (args.output_dir / "rollout_config.yaml").open("w") as f:
        yaml.safe_dump(rollout_config, f, sort_keys=False)

    print(f"Wrote {args.output_dir / 'buf.pkl'}")
    print(f"Wrote {args.output_dir / 'rollout_config.yaml'}")


if __name__ == "__main__":
    main()
