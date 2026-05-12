#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Agentic bridge server for RoboCasa + GR00T.

This script runs one simulator instance and exposes socket APIs via ZeroMQ so
external programs can control the environment.

Main endpoints:
  - status
  - reset
  - get_observation
  - step_policy
  - step_action
  - move                  (forward/backward/turn_left/turn_right in body frame)
  - list_skills
  - register_skills
  - call_skill

Example:
  python scripts/run_agentic_bridge.py \
      --model_path ../checkpoint-60000/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000 \
      --env_name RinseSinkBasin \
      --split target \
      --port 8010
"""

import argparse
import time
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch

import robocasa  # noqa: F401
import robosuite  # noqa: F401

from gr00t.eval.service import BaseInferenceServer
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from robocasa.models.scenes.scene_registry import (
    LAYOUT_GROUPS_TO_IDS,
    STYLE_GROUPS_TO_IDS,
)
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon


def _make_single_env(env_name: str, split: str, n_action_steps: int, max_episode_steps: int):
    def _make():
        env_kwargs = {"enable_render": True}
        split_arg = split
        if split == "custom":
            split_arg = None
        env_kwargs["split"] = split_arg
        base_env = gym.make(f"robocasa/{env_name}", **env_kwargs)
        return MultiStepWrapper(
            base_env,
            video_delta_indices=np.array([0]),
            state_delta_indices=np.array([0]),
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
        )

    return _make


def _get_robosuite_env(vec_env):
    env = vec_env.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env


class AgenticBridgeCore:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.env_name: Optional[str] = args.env_name
        self.split = args.split
        self.layout_id: Optional[int] = None
        self.style_id: Optional[int] = None
        self.max_steps = args.max_steps
        self.step_count = 0
        self.done = False
        self.success = False
        self.last_reward = 0.0
        self.last_info: Dict[str, Any] = {}
        self.last_obs: Optional[Dict[str, Any]] = None

        data_config = DATA_CONFIG_MAP[args.data_config]
        self.policy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        self.vec_env = None
        self.rs_env = None

        # Teammates can register additional names, but built-ins are executable now.
        self.skills: Dict[str, Dict[str, Any]] = {
            "move_forward": {"type": "builtin"},
            "move_backward": {"type": "builtin"},
            "turn_left": {"type": "builtin"},
            "turn_right": {"type": "builtin"},
            "policy_step": {"type": "builtin"},
        }

        # Optional startup env can be provided, but default behavior is lazy launch.
        if self.env_name:
            init_max_steps = self.max_steps or get_task_horizon(self.env_name)
            self._rebuild_env(self.env_name, self.split, init_max_steps)
            self.reset()

    def close(self):
        if self.vec_env is not None:
            try:
                self.vec_env.close()
            except Exception:
                pass

    def _rebuild_env(
        self,
        env_name: str,
        split: str,
        max_steps: int,
        layout_id: Optional[int] = None,
        style_id: Optional[int] = None,
    ):
        if not env_name:
            raise ValueError("env_name is required to launch simulator")
        if self.vec_env is not None:
            try:
                self.vec_env.close()
            except Exception:
                pass
        if split not in ("pretrain", "target", "custom"):
            raise ValueError("split must be pretrain, target, or custom")
        if split != "custom" and (layout_id is not None or style_id is not None):
            raise ValueError("layout/style overrides require split='custom'")
        self.env_name = env_name
        self.split = split
        self.max_steps = int(max_steps)
        self.layout_id = layout_id
        self.style_id = style_id
        env_fn = _make_single_env(self.env_name, self.split, self.args.n_action_steps, self.max_steps)
        self.vec_env = gym.vector.SyncVectorEnv([env_fn])
        base_env = self.vec_env.envs[0]
        if split == "custom":
            kwargs = {}
            if layout_id is not None:
                kwargs["layout_ids"] = [layout_id]
            if style_id is not None:
                kwargs["style_ids"] = [style_id]
            if kwargs:
                # Recreate env with explicit scene config when requested.
                self.vec_env.close()
                def _make_custom():
                    custom_kwargs = {"enable_render": True, "split": None, **kwargs}
                    custom_base = gym.make(f"robocasa/{self.env_name}", **custom_kwargs)
                    return MultiStepWrapper(
                        custom_base,
                        video_delta_indices=np.array([0]),
                        state_delta_indices=np.array([0]),
                        n_action_steps=self.args.n_action_steps,
                        max_episode_steps=self.max_steps,
                    )
                self.vec_env = gym.vector.SyncVectorEnv([_make_custom])
                base_env = self.vec_env.envs[0]
        self.rs_env = _get_robosuite_env(self.vec_env)

    def _ensure_env_ready(self):
        if self.vec_env is None or self.rs_env is None:
            raise RuntimeError("Simulator is not launched. Call configure_environment first.")

    def _render_update(self):
        if self.rs_env is None:
            return
        if self.rs_env is not None and hasattr(self.rs_env, "viewer") and self.rs_env.viewer is not None:
            self.rs_env.viewer.update()

    def _update_flags(self, reward, terminated, truncated, info):
        self.last_reward = float(reward[0] if isinstance(reward, np.ndarray) else reward)
        self.done = bool(terminated[0]) or bool(truncated[0])
        self.last_info = info
        self.step_count += 1
        success_flags = info.get("success", [[False]])
        self.success = bool(any(s for s in success_flags[0]))

    def _coerce_action(self, action: Dict[str, Any]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for key, value in action.items():
            arr = np.array(value, dtype=np.float32)
            # Expected vector-env action layout for n_envs=1 is (1, n_action_steps, action_dim).
            # Accept convenient forms and normalize them:
            #   (action_dim,) -> (1, n_action_steps, action_dim)
            #   (n_action_steps, action_dim) -> (1, n_action_steps, action_dim)
            #   (1, n_action_steps, action_dim) -> unchanged
            if arr.ndim == 1:
                arr = np.tile(arr[np.newaxis, :], (self.args.n_action_steps, 1))
                arr = arr[np.newaxis, ...]
            elif arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            elif arr.ndim == 3:
                pass
            else:
                raise ValueError(f"{key}: expected rank-1/2/3, got {arr.shape}")

            if arr.shape[0] != 1:
                raise ValueError(f"{key}: first dim must be env batch=1, got {arr.shape[0]}")
            if arr.shape[1] != self.args.n_action_steps:
                raise ValueError(
                    f"{key}: second dim must equal n_action_steps={self.args.n_action_steps}, got {arr.shape[1]}"
                )
            out[key] = np.clip(arr, -1.0, 1.0)
        return out

    def _fresh_base_action(self) -> Dict[str, np.ndarray]:
        # control_mode > 0 -> base mode in robocasa gym wrapper.
        return {
            "action.end_effector_position": np.zeros((self.args.n_action_steps, 3), dtype=np.float32),
            "action.end_effector_rotation": np.zeros((self.args.n_action_steps, 3), dtype=np.float32),
            "action.gripper_close": np.full((self.args.n_action_steps, 1), -1.0, dtype=np.float32),
            "action.base_motion": np.zeros((self.args.n_action_steps, 4), dtype=np.float32),
            "action.control_mode": np.ones((self.args.n_action_steps, 1), dtype=np.float32),
        }

    def status(self) -> Dict[str, Any]:
        out = {
            "env_name": self.env_name,
            "split": self.split,
            "env_ready": self.vec_env is not None,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "n_action_steps": self.args.n_action_steps,
            "layout_id": self.layout_id,
            "style_id": self.style_id,
            "done": self.done,
            "success": self.success,
            "last_reward": self.last_reward,
            "timestamp": time.time(),
        }
        if self.last_obs is not None:
            annotation = self.last_obs.get("annotation.human.task_description", ["N/A"])
            if isinstance(annotation, np.ndarray):
                annotation = annotation[0]
            out["task_instruction"] = annotation
        return out

    def reset(self) -> Dict[str, Any]:
        self._ensure_env_ready()
        self.step_count = 0
        self.done = False
        self.success = False
        self.last_reward = 0.0
        self.last_info = {}

        obs, _ = self.vec_env.reset()
        self.last_obs = obs
        self.rs_env.initialize_renderer()
        self._render_update()
        return self.status()

    def configure_environment(
        self,
        env_name: Optional[str] = None,
        split: Optional[str] = None,
        max_steps: Optional[int] = None,
        layout_id: Optional[int] = None,
        style_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        new_env_name = env_name or self.env_name
        if not new_env_name:
            raise ValueError("env_name must be specified at least once before launch")
        new_split = split or self.split
        if new_split not in ("pretrain", "target", "custom"):
            raise ValueError("split must be pretrain, target, or custom")
        new_max_steps = int(max_steps) if max_steps is not None else get_task_horizon(new_env_name)
        self._rebuild_env(new_env_name, new_split, new_max_steps, layout_id=layout_id, style_id=style_id)
        return self.reset()

    def list_atomic_tasks(self) -> Dict[str, Any]:
        return {"atomic_tasks": list(TASK_SET_REGISTRY.get("all_atomic_tasks", []))}

    def list_composite_tasks(self) -> Dict[str, Any]:
        return {"composite_tasks": list(TASK_SET_REGISTRY.get("all_composite_tasks", []))}

    def list_available_envs(self) -> Dict[str, Any]:
        return {
            "all_tasks": list(TASK_SET_REGISTRY.get("all_tasks", [])),
            "atomic_tasks": list(TASK_SET_REGISTRY.get("all_atomic_tasks", [])),
            "composite_tasks": list(TASK_SET_REGISTRY.get("all_composite_tasks", [])),
        }

    def list_scene_config(self) -> Dict[str, Any]:
        return {
            "layout_id_range": [1, 60],
            "style_id_range": [1, 60],
            "layout_groups": LAYOUT_GROUPS_TO_IDS,
            "style_groups": STYLE_GROUPS_TO_IDS,
            "split_options": ["pretrain", "target", "custom"],
            # Derived from robocasa scene registry groups:
            # test=-1 -> 1..10, train=-2 -> 11..60, all=-3 -> 1..60
            "split_layout_style_ranges": {
                "target": [1, 10],
                "pretrain": [11, 60],
                "custom": [1, 60],
            },
        }

    def list_policy_descriptions(self) -> Dict[str, Any]:
        candidates = []
        task_instruction = self.status().get("task_instruction")
        if task_instruction is not None:
            if isinstance(task_instruction, (tuple, list)) and len(task_instruction) > 0:
                candidates.append(str(task_instruction[0]))
            else:
                candidates.append(str(task_instruction))
        if self.env_name:
            candidates.append(str(self.env_name))
        deduped = list(dict.fromkeys([c for c in candidates if c]))
        return {"policy_descriptions": deduped}

    def snapshot(self) -> Dict[str, Any]:
        observation = None
        if self.vec_env is not None and self.last_obs is not None:
            observation = self.get_observation()
        return {
            "status": self.status(),
            "observation": observation,
            "skills": {"skills": list(self.skills.keys()) + list(TASK_SET_REGISTRY.get("all_atomic_tasks", []))},
        }

    def get_observation(self) -> Dict[str, Any]:
        self._ensure_env_ready()
        if self.last_obs is None:
            raise RuntimeError("No observation available. Call reset first.")
        return self.last_obs

    def step_action(self, action: Dict[str, Any], repeat: int = 1) -> Dict[str, Any]:
        self._ensure_env_ready()
        if self.done:
            return self.status()
        action_dict = self._coerce_action(action)
        for _ in range(repeat):
            if self.done:
                break
            obs, reward, terminated, truncated, info = self.vec_env.step(action_dict)
            self.last_obs = obs
            self._update_flags(reward, terminated, truncated, info)
            self._render_update()
        return self.status()

    def step_policy(self, repeat: int = 1, description: Optional[str] = None) -> Dict[str, Any]:
        self._ensure_env_ready()
        if self.last_obs is None:
            raise RuntimeError("No observation available. Call reset first.")
        if self.done:
            return self.status()
        for _ in range(repeat):
            if self.done:
                break
            policy_obs = self.last_obs
            if description is not None and description.strip():
                policy_obs = dict(self.last_obs)
                policy_obs["annotation.human.task_description"] = (description.strip(),)
            action_dict = self.policy.get_action(policy_obs)
            obs, reward, terminated, truncated, info = self.vec_env.step(action_dict)
            self.last_obs = obs
            self._update_flags(reward, terminated, truncated, info)
            self._render_update()
        return self.status()

    def move(self, command: str, magnitude: float = 0.5, repeat: int = 1) -> Dict[str, Any]:
        if magnitude < 0 or magnitude > 1:
            raise ValueError("magnitude must be in [0, 1]")

        action = self._fresh_base_action()
        base_motion = np.zeros(4, dtype=np.float32)
        if command == "forward":
            base_motion[0] = magnitude
        elif command == "backward":
            base_motion[0] = -magnitude
        elif command == "turn_left":
            base_motion[2] = magnitude
        elif command == "turn_right":
            base_motion[2] = -magnitude
        else:
            raise ValueError(f"Unknown command: {command}")

        action["action.base_motion"] = np.tile(base_motion[np.newaxis, :], (self.args.n_action_steps, 1))
        return self.step_action(action, repeat=repeat)

    def register_skills(self, names: list[str]) -> Dict[str, Any]:
        for name in names:
            if not isinstance(name, str):
                raise ValueError("skill names must be strings")
            self.skills[name] = {"type": "external"}
        return {"skills": list(self.skills.keys())}

    def call_skill(self, name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        if name == "policy_step":
            return self.step_policy(
                repeat=int(params.get("repeat", 1)),
                description=params.get("description"),
            )
        if name == "move_forward":
            return self.move("forward", float(params.get("magnitude", 0.5)), int(params.get("repeat", 1)))
        if name == "move_backward":
            return self.move("backward", float(params.get("magnitude", 0.5)), int(params.get("repeat", 1)))
        if name == "turn_left":
            return self.move("turn_left", float(params.get("magnitude", 0.5)), int(params.get("repeat", 1)))
        if name == "turn_right":
            return self.move("turn_right", float(params.get("magnitude", 0.5)), int(params.get("repeat", 1)))
        # Allow atomic task names as "potential skill list":
        # selecting one switches environment and uses that task's default language condition.
        if name in TASK_SET_REGISTRY.get("all_atomic_tasks", []):
            split = str(params.get("split", self.split))
            max_steps = params.get("max_steps", None)
            return self.configure_environment(env_name=name, split=split, max_steps=max_steps)
        raise ValueError(f"Skill '{name}' not implemented in bridge yet")


class AgenticBridgeServer(BaseInferenceServer):
    def __init__(self, core: AgenticBridgeCore, host: str = "*", port: int = 8010):
        super().__init__(host=host, port=port)
        self.core = core
        self.register_endpoint("status", self._status, requires_input=False)
        self.register_endpoint("get_snapshot", self._get_snapshot, requires_input=False)
        self.register_endpoint("reset", self._reset, requires_input=False)
        self.register_endpoint("get_observation", self._get_observation, requires_input=False)
        self.register_endpoint("step_policy", self._step_policy, requires_input=True)
        self.register_endpoint("step_action", self._step_action, requires_input=True)
        self.register_endpoint("move", self._move, requires_input=True)
        self.register_endpoint("list_skills", self._list_skills, requires_input=False)
        self.register_endpoint("list_available_envs", self._list_available_envs, requires_input=False)
        self.register_endpoint("list_scene_config", self._list_scene_config, requires_input=False)
        self.register_endpoint("list_policy_descriptions", self._list_policy_descriptions, requires_input=False)
        self.register_endpoint("list_atomic_tasks", self._list_atomic_tasks, requires_input=False)
        self.register_endpoint("list_composite_tasks", self._list_composite_tasks, requires_input=False)
        self.register_endpoint("configure_environment", self._configure_environment, requires_input=True)
        self.register_endpoint("register_skills", self._register_skills, requires_input=True)
        self.register_endpoint("call_skill", self._call_skill, requires_input=True)

    def _status(self):
        return self.core.status()

    def _reset(self):
        return self.core.reset()

    def _get_observation(self):
        return self.core.get_observation()

    def _step_policy(self, data: Dict[str, Any]):
        repeat = int(data.get("repeat", 1))
        description = data.get("description")
        return self.core.step_policy(repeat=repeat, description=description)

    def _step_action(self, data: Dict[str, Any]):
        if "action" not in data:
            raise ValueError("Missing 'action' in request")
        repeat = int(data.get("repeat", 1))
        return self.core.step_action(action=data["action"], repeat=repeat)

    def _move(self, data: Dict[str, Any]):
        command = data.get("command")
        if command is None:
            raise ValueError("Missing 'command' in request")
        magnitude = float(data.get("magnitude", 0.5))
        repeat = int(data.get("repeat", 1))
        return self.core.move(command=command, magnitude=magnitude, repeat=repeat)

    def _list_skills(self):
        return {"skills": list(self.core.skills.keys()) + list(TASK_SET_REGISTRY.get("all_atomic_tasks", []))}

    def _list_available_envs(self):
        return self.core.list_available_envs()

    def _list_scene_config(self):
        return self.core.list_scene_config()

    def _list_policy_descriptions(self):
        return self.core.list_policy_descriptions()

    def _list_atomic_tasks(self):
        return self.core.list_atomic_tasks()

    def _list_composite_tasks(self):
        return self.core.list_composite_tasks()

    def _configure_environment(self, data: Dict[str, Any]):
        layout_id = data.get("layout_id", None)
        style_id = data.get("style_id", None)
        return self.core.configure_environment(
            env_name=data.get("env_name"),
            split=data.get("split"),
            max_steps=data.get("max_steps"),
            layout_id=(None if layout_id is None else int(layout_id)),
            style_id=(None if style_id is None else int(style_id)),
        )

    def _get_snapshot(self):
        return self.core.snapshot()

    def _register_skills(self, data: Dict[str, Any]):
        names = data.get("skills", [])
        if not isinstance(names, list):
            raise ValueError("'skills' must be a list of strings")
        return self.core.register_skills(names)

    def _call_skill(self, data: Dict[str, Any]):
        name = data.get("name")
        if not isinstance(name, str):
            raise ValueError("Missing 'name' in request")
        params = data.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("'params' must be a dict")
        return self.core.call_skill(name=name, params=params)


def parse_args():
    parser = argparse.ArgumentParser(description="RoboCasa + GR00T bridge server for agentic workflow")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--env_name", type=str, default=None, help="Optional startup env; if omitted, simulator launches lazily from UI/API")
    parser.add_argument("--split", type=str, default="target", choices=["pretrain", "target", "custom"])
    parser.add_argument("--data_config", type=str, default="panda_omron")
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment")
    parser.add_argument("--denoising_steps", type=int, default=4)
    parser.add_argument("--n_action_steps", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--host", type=str, default="*")
    parser.add_argument("--port", type=int, default=8010)
    return parser.parse_args()


def print_examples(host: str, port: int):
    print("\nBridge endpoint examples (Python client using BaseInferenceClient):")
    print("  client.call_endpoint('status', requires_input=False)")
    print("  client.call_endpoint('list_available_envs', requires_input=False)")
    print("  client.call_endpoint('list_scene_config', requires_input=False)")
    print("  client.call_endpoint('get_snapshot', requires_input=False)")
    print("  client.call_endpoint('reset', requires_input=False)")
    print("  client.call_endpoint('get_observation', requires_input=False)")
    print("  client.call_endpoint('list_atomic_tasks', requires_input=False)")
    print("  client.call_endpoint('configure_environment', {'env_name': 'OpenDrawer', 'split': 'target'})")
    print("  client.call_endpoint('move', {'command': 'forward', 'magnitude': 0.5, 'repeat': 1})")
    print("  client.call_endpoint('step_policy', {'repeat': 1})")
    print("  client.call_endpoint('call_skill', {'name': 'turn_left', 'params': {'magnitude': 0.5}})")
    print(f"\nServer binding: tcp://{host}:{port}\n")


def main():
    args = parse_args()
    print(f"Loading policy from {args.model_path}")
    print(f"Startup env: {args.env_name} | split={args.split} | n_action_steps={args.n_action_steps}")
    core = AgenticBridgeCore(args)
    server = AgenticBridgeServer(core=core, host=args.host, port=args.port)
    print_examples(args.host, args.port)
    try:
        server.run()
    finally:
        core.close()


if __name__ == "__main__":
    main()
