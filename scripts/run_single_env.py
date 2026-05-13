# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Run a single RoboCasa environment with a GR00T policy checkpoint.
By default this opens the MuJoCo viewer GUI; pass --headless for SSH/server runs.

Uses the same environment wrappers as run_eval.py to ensure correct
observation formatting for the policy.

Example:
    python scripts/run_single_env.py \
        --model_path /path/to/checkpoint \
        --env_name RinseSinkBasin \
        --split target
"""

import argparse
import time

import gymnasium as gym
import numpy as np
import torch

import robocasa  # noqa: F401
import robosuite  # noqa: F401

from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from robocasa.utils.dataset_registry_utils import get_task_horizon


def _make_single_env(env_name, split, n_action_steps, max_episode_steps):
    """Create a single env with MultiStepWrapper, matching run_eval.py."""
    def _make():
        base_env = gym.make(f"robocasa/{env_name}", split=split, enable_render=True)
        wrapped = MultiStepWrapper(
            base_env,
            video_delta_indices=np.array([0]),
            state_delta_indices=np.array([0]),
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
        )
        return wrapped
    return _make


def _get_robosuite_env(vec_env):
    """Unwrap through all gymnasium wrappers to reach the robosuite env."""
    e = vec_env.envs[0]
    while hasattr(e, "env"):
        e = e.env
    return e


def main():
    parser = argparse.ArgumentParser(description="Run a single RoboCasa env with a GR00T policy")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--env_name", type=str, default="RinseSinkBasin")
    parser.add_argument("--split", type=str, default="target", choices=["pretrain", "target"])
    parser.add_argument("--data_config", type=str, default="panda_omron")
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment")
    parser.add_argument("--denoising_steps", type=int, default=4)
    parser.add_argument("--n_action_steps", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--n_episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30, help="Target frame rate for viewer")
    parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer GUI")
    args = parser.parse_args()

    max_steps = args.max_steps or get_task_horizon(args.env_name)

    # -- Build policy --
    print(f"Loading policy from {args.model_path} ...")
    data_config = DATA_CONFIG_MAP[args.data_config]
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # -- Create env with SyncVectorEnv (n=1) + MultiStepWrapper --
    mode = "headless" if args.headless else "viewer"
    print(f"Task: {args.env_name} | split: {args.split} | max_steps: {max_steps} | mode: {mode}")
    print("Creating environment ...")
    env_fn = _make_single_env(args.env_name, args.split, args.n_action_steps, max_steps)
    vec_env = gym.vector.SyncVectorEnv([env_fn])

    rs_env = _get_robosuite_env(vec_env)

    for ep in range(args.n_episodes):
        print(f"\n=== Episode {ep + 1}/{args.n_episodes} ===")
        obs, info = vec_env.reset()

        if not args.headless:
            # Re-initialize the MuJoCo viewer after reset (reset destroys it).
            rs_env.initialize_renderer()

        annotation = obs.get("annotation.human.task_description", ["N/A"])
        if isinstance(annotation, np.ndarray):
            annotation = annotation[0]
        print(f"Task instruction: {annotation}")

        success = False
        done = False

        while not done:
            start = time.time()

            action_dict = policy.get_action(obs)
            obs, reward, terminated, truncated, info = vec_env.step(action_dict)
            done = bool(terminated[0]) or bool(truncated[0])

            if not args.headless:
                # Update MuJoCo viewer (lazily launches the window on first call).
                rs_env.viewer.update()

            if not args.headless:
                elapsed = time.time() - start
                sleep_time = 1.0 / args.fps - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            success_flags = info.get("success", [[False]])
            if any(s for s in success_flags[0]):
                success = True
                print("  SUCCESS!")
                break

        status = "SUCCESS" if success else "FAILED"
        print(f"Episode {ep + 1} finished: {status}")

    vec_env.close()
    print("Done.")


if __name__ == "__main__":
    main()
