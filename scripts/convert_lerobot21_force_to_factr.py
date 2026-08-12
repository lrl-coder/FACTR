#!/usr/bin/env python
"""Convert LeRobot 2.1 video datasets to FACTR robobuf format.

The low-dimensional FACTR observation is the compensated 6D force/torque.
Proprioceptive state is intentionally excluded from the model input.
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

from robobuf.buffers import ObsWrapper, ReplayBuffer, Transition


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = Path(
    os.environ.get("FACTR_DATASET_ROOT", PROJECT_ROOT.parent / "dataset-8Hz")
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "processed_data" / "dataset_8hz_force"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot 2.1 datasets to FACTR using 6D force/torque."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="A LeRobot dataset directory, or a root containing task subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where buf.pkl and rollout_config.yaml will be written.",
    )
    parser.add_argument(
        "--obs-keys",
        nargs="+",
        default=["observation.wrench_compensated"],
        help="Low-dimensional force/torque observation columns to concatenate.",
    )
    parser.add_argument(
        "--action-key",
        default="action",
        help="Action column.",
    )
    parser.add_argument(
        "--image-keys",
        nargs="+",
        default=["observation.images.third_view", "observation.images.wrist"],
        help="Video feature keys exported as FACTR cameras in this order.",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="*",
        default=None,
        help="Optional subset of child dataset names when --dataset-dir is a root.",
    )
    parser.add_argument(
        "--episode-indices",
        nargs="*",
        type=int,
        default=None,
        help="Optional explicit episode indices, applied to each selected dataset.",
    )
    parser.add_argument(
        "--max-episodes-per-dataset",
        type=int,
        default=None,
        help="Optional cap for quick debugging.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--no-normalize", action="store_true")
    return parser.parse_args()


def load_info(dataset_dir):
    info_path = dataset_dir / "meta" / "info.json"
    with info_path.open("r") as f:
        return json.load(f)


def discover_dataset_dirs(dataset_dir, dataset_names):
    if (dataset_dir / "meta" / "info.json").exists():
        if dataset_names:
            raise ValueError("--dataset-names is only valid when --dataset-dir is a root.")
        return [dataset_dir]

    candidates = sorted(
        child for child in dataset_dir.iterdir() if (child / "meta" / "info.json").exists()
    )
    if dataset_names:
        wanted = set(dataset_names)
        candidates = [child for child in candidates if child.name in wanted]
        missing = wanted - {child.name for child in candidates}
        if missing:
            raise FileNotFoundError(f"Dataset names not found: {sorted(missing)}")

    if not candidates:
        raise FileNotFoundError(
            f"No LeRobot datasets found under {dataset_dir}. Expected meta/info.json."
        )
    return candidates


def episode_records(dataset_dir, info, episode_indices, max_episodes):
    if episode_indices is None:
        total = int(info["total_episodes"])
        episode_indices = list(range(total))
    if max_episodes is not None:
        episode_indices = episode_indices[:max_episodes]

    data_pattern = info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    video_pattern = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    chunk_size = int(info.get("chunks_size", 1000))

    records = []
    for episode_index in episode_indices:
        episode_chunk = episode_index // chunk_size
        parquet_path = dataset_dir / data_pattern.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )
        if not parquet_path.exists():
            raise FileNotFoundError(parquet_path)
        records.append(
            {
                "dataset_name": dataset_dir.name,
                "dataset_dir": dataset_dir,
                "episode_index": episode_index,
                "episode_chunk": episode_chunk,
                "parquet_path": parquet_path,
                "video_pattern": video_pattern,
            }
        )
    return records


def validate_features(info, obs_keys, action_key, image_keys):
    features = info["features"]
    for key in obs_keys + [action_key] + image_keys:
        if key not in features:
            raise KeyError(f"Missing feature {key}")

    for image_key in image_keys:
        dtype = features[image_key].get("dtype")
        if dtype != "video":
            raise ValueError(f"Expected {image_key} dtype video, got {dtype}")

    if obs_keys == ["observation.wrench_compensated"]:
        force_shape = features[obs_keys[0]].get("shape")
        if force_shape != [6]:
            raise ValueError(f"Expected 6D force/torque, got shape {force_shape}")
    action_shape = features[action_key].get("shape")
    if action_shape != [8]:
        raise ValueError(f"Expected 8D action, got shape {action_shape}")


def read_low_dim_episode(path, obs_keys, action_key):
    table = pq.read_table(path, columns=list(obs_keys) + [action_key])
    rows = table.to_pylist()
    obs_parts = [
        np.asarray([row[key] for row in rows], dtype=np.float32) for key in obs_keys
    ]
    obs = np.concatenate(obs_parts, axis=1)
    actions = np.asarray([row[action_key] for row in rows], dtype=np.float32)
    return obs, actions


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


def video_path_for_record(record, image_key):
    return record["dataset_dir"] / record["video_pattern"].format(
        episode_chunk=record["episode_chunk"],
        episode_index=record["episode_index"],
        video_key=image_key,
    )


def encode_bgr_frame(frame_bgr, image_size, jpeg_quality):
    if frame_bgr.shape[:2] != (image_size, image_size):
        frame_bgr = cv2.resize(frame_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(
        ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    )
    if not ok:
        raise ValueError("Failed to encode image as JPEG.")
    return encoded


def read_encoded_video_frames(video_path, expected_frames, image_size, jpeg_quality):
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    try:
        for frame_idx in range(expected_frames):
            ok, frame_bgr = cap.read()
            if not ok:
                raise RuntimeError(
                    f"Video ended early: {video_path} at frame {frame_idx}/{expected_frames}"
                )
            frames.append(encode_bgr_frame(frame_bgr, image_size, jpeg_quality))
    finally:
        cap.release()
    return frames


def collect_records(dataset_dirs, args):
    records = []
    dataset_summaries = []
    for dataset_dir in dataset_dirs:
        info = load_info(dataset_dir)
        validate_features(info, args.obs_keys, args.action_key, args.image_keys)
        records.extend(
            episode_records(
                dataset_dir,
                info,
                args.episode_indices,
                args.max_episodes_per_dataset,
            )
        )
        dataset_summaries.append(
            {
                "name": dataset_dir.name,
                "total_episodes": int(info["total_episodes"]),
                "total_frames": int(info["total_frames"]),
                "fps": info.get("fps"),
            }
        )
    return records, dataset_summaries


def build_buffer(records, args, obs_mean, obs_std, action_mean, action_std, normalize):
    buffer = ReplayBuffer()
    total_frames = 0

    for record in tqdm(records, desc="Building robobuf"):
        obs, actions = read_low_dim_episode(
            record["parquet_path"], args.obs_keys, args.action_key
        )
        if normalize:
            obs = (obs - obs_mean) / obs_std
            actions = (actions - action_mean) / action_std

        enc_images_by_cam = [
            read_encoded_video_frames(
                video_path_for_record(record, image_key),
                expected_frames=len(obs),
                image_size=args.image_size,
                jpeg_quality=args.jpeg_quality,
            )
            for image_key in args.image_keys
        ]

        for step_idx in range(len(obs)):
            step_obs = {"state": obs[step_idx].astype(np.float32)}
            for cam_idx, enc_images in enumerate(enc_images_by_cam):
                step_obs[f"enc_cam_{cam_idx}"] = enc_images[step_idx]

            transition = Transition(
                obs=ObsWrapper(step_obs),
                action=actions[step_idx].astype(np.float32),
                reward=float(step_idx == len(obs) - 1),
            )
            buffer.add(transition, is_first=(step_idx == 0))
        total_frames += len(obs)

    return buffer, total_frames


def main():
    args = parse_args()
    normalize = not args.no_normalize
    dataset_dirs = discover_dataset_dirs(args.dataset_dir, args.dataset_names)
    records, dataset_summaries = collect_records(dataset_dirs, args)

    action_names = None
    for dataset_dir in dataset_dirs:
        names = load_info(dataset_dir)["features"][args.action_key].get("names")
        if names is None:
            continue
        names = [str(name) for name in names]
        if action_names is not None and names != action_names:
            raise ValueError("Source datasets disagree on action joint order.")
        action_names = names

    all_obs, all_actions = [], []
    for record in tqdm(records, desc="Reading low-dimensional data"):
        obs, actions = read_low_dim_episode(
            record["parquet_path"], args.obs_keys, args.action_key
        )
        all_obs.append(obs)
        all_actions.append(actions)

    obs_mean, obs_std = gaussian_stats(all_obs)
    action_mean, action_std = gaussian_stats(all_actions)
    if not normalize:
        obs_mean = np.zeros_like(obs_mean)
        obs_std = np.ones_like(obs_std)
        action_mean = np.zeros_like(action_mean)
        action_std = np.ones_like(action_std)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    buffer, total_frames = build_buffer(
        records,
        args,
        obs_mean,
        obs_std,
        action_mean,
        action_std,
        normalize,
    )

    with (args.output_dir / "buf.pkl").open("wb") as f:
        pickle.dump(buffer.to_traj_list(), f)

    rollout_config = {
        "source_datasets": dataset_summaries,
        "obs_config": {
            "obs_keys": list(args.obs_keys),
            "obs_mode": "force",
            "camera_keys": list(args.image_keys),
            "ignored_keys": ["observation.state"],
            "state_layout": {
                "force": [0, int(obs_mean.shape[0])],
                "source_observation.wrench_compensated": [0, 6],
            },
        },
        "action_config": {
            "action_key": args.action_key,
            "action_dim": int(action_mean.shape[0]),
            "names": action_names,
        },
        "norm_stats": {
            "state": stats_to_yaml(obs_mean, obs_std),
            "action": stats_to_yaml(action_mean, action_std),
        },
        "normalized": normalize,
        "image_size": args.image_size,
        "num_episodes": len(records),
        "num_frames": int(total_frames),
    }
    with (args.output_dir / "rollout_config.yaml").open("w") as f:
        yaml.safe_dump(rollout_config, f, sort_keys=False)

    print(f"Wrote {args.output_dir / 'buf.pkl'}")
    print(f"Wrote {args.output_dir / 'rollout_config.yaml'}")
    print(f"obs_dim={obs_mean.shape[0]} action_dim={action_mean.shape[0]}")
    print(f"episodes={len(records)} frames={total_frames}")


if __name__ == "__main__":
    main()
