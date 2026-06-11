"""
Run a full mock inference pass through FACTR without loading pretrained weights.

This script constructs the default ViT + ACT-style TransformerAgent, creates
random camera and force/state inputs, and prints the predicted action chunk
shape. The actions are not meaningful because all model weights are randomly
initialized.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

from factr.models.action_transformer import TransformerAgent
from factr.models.vit import vit_base_patch16


def build_agent(
    obs_dim: int,
    action_dim: int,
    action_chunk: int,
    n_cams: int,
    image_chunk: int,
    image_size: int,
) -> TransformerAgent:
    """Create the FACTR policy with random weights and no checkpoint loading."""
    visual_encoder = vit_base_patch16(
        img_size=image_size,
        use_cls=True,
        drop_path_rate=0.0,
    )

    curriculum = SimpleNamespace(
        space="pixel",
        operator="blur",
        scheduler="no",  # returns scale=0, so no curriculum corruption at inference
        start_scale=0,
        stop_scale=0,
        max_step=1,
    )

    return TransformerAgent(
        features=visual_encoder,
        odim=obs_dim,
        n_cams=n_cams,
        ac_dim=action_dim,
        ac_chunk=action_chunk,
        use_obs="add_token",
        imgs_per_cam=image_chunk,
        dropout=0.0,
        img_dropout=0.0,
        share_cam_features=False,
        early_fusion=True,
        feat_norm="layer_norm",
        token_dim=512,
        curriculum=curriculum,
        transformer_kwargs={
            "d_model": 512,
            "dropout": 0.0,
            "nhead": 8,
            "num_encoder_layers": 4,
            "num_decoder_layers": 6,
            "dim_feedforward": 3200,
            "activation": "relu",
        },
    )


def make_mock_inputs(
    batch_size: int,
    n_cams: int,
    image_chunk: int,
    obs_dim: int,
    image_size: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Create mock inputs matching RobobufReplayBuffer output after DataLoader."""
    imgs = {
        f"cam{i}": torch.rand(
            batch_size,
            image_chunk,
            3,
            image_size,
            image_size,
            device=device,
        )
        for i in range(n_cams)
    }

    # In this codebase obs is the normalized low-dimensional state token.
    # For the provided config it includes Franka torque and gripper position.
    obs = torch.randn(batch_size, obs_dim, device=device)
    return imgs, obs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--obs-dim", type=int, default=7)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--action-chunk", type=int, default=100)
    parser.add_argument("--n-cams", type=int, default=1)
    parser.add_argument("--image-chunk", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    args = parser.parse_args()

    if args.image_chunk != 1:
        raise ValueError(
            "The default ViT expects 3-channel images. Keep --image-chunk 1 "
            "unless you also change the ViT input channel count."
        )

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    agent = build_agent(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        action_chunk=args.action_chunk,
        n_cams=args.n_cams,
        image_chunk=args.image_chunk,
        image_size=args.image_size,
    ).to(device)
    agent.eval()

    imgs, obs = make_mock_inputs(
        batch_size=args.batch_size,
        n_cams=args.n_cams,
        image_chunk=args.image_chunk,
        obs_dim=args.obs_dim,
        image_size=args.image_size,
        device=device,
    )

    with torch.no_grad():
        tokens = agent.tokenize_obs(imgs, obs)
        actions = agent.get_actions(imgs, obs)

    print("Mock FACTR inference completed.")
    print(f"image[cam0] shape: {tuple(imgs['cam0'].shape)}")
    print(f"obs shape:         {tuple(obs.shape)}")
    print(f"token shape:       {tuple(tokens.shape)}")
    print(f"actions shape:     {tuple(actions.shape)}")
    print(f"first action[0,0]: {actions[0, 0].detach().cpu().tolist()}")


if __name__ == "__main__":
    main()

# Mock FACTR inference completed.
# image[cam0] shape: (2, 1, 3, 224, 224)
# obs shape:         (2, 7)
# token shape:       (2, 2, 512)
# actions shape:     (2, 100, 7)
# first action[0,0]: [1.5846076011657715, -0.16965866088867188, -0.6481045484542847, 0.39112338423728943, 1.1579583883285522, -0.09304735064506531, 0.014471067115664482]
