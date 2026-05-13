#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Streamlit UI for run_agentic_bridge.py.

Run:
    conda activate robocasa
    streamlit run scripts/demo_agentic_bridge.py -- --host localhost --port 8010
"""

import argparse
import sys
import time
from typing import Any, Dict

import numpy as np
import streamlit as st
import zmq
from streamlit.runtime.scriptrunner import get_script_run_ctx

from gr00t.eval.service import BaseInferenceClient


def _parse_streamlit_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--split", type=str, default="pretrain")
    parser.add_argument("--env_name", type=str, default="")
    parser.add_argument("--layout_id", type=str, default="")
    parser.add_argument("--style_id", type=str, default="")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def _to_jsonable(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _pick_frame(video_array: np.ndarray) -> np.ndarray:
    arr = np.array(video_array)
    if arr.ndim == 5:  # (B, T, H, W, C)
        frame = arr[0, 0]
    elif arr.ndim == 4:  # (T, H, W, C)
        frame = arr[0]
    elif arr.ndim == 3:  # (H, W, C)
        frame = arr
    else:
        raise ValueError(f"Unsupported video shape: {arr.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _bridge_call(
    host: str,
    port: int,
    endpoint: str,
    payload: Dict[str, Any] | None = None,
    requires_input: bool = False,
    timeout_ms: int = 1000,
):
    payload = payload or {}
    client = BaseInferenceClient(host=host, port=port)
    try:
        client.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        client.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        client.socket.setsockopt(zmq.LINGER, 0)
        if requires_input:
            return client.call_endpoint(endpoint, payload, requires_input=True)
        return client.call_endpoint(endpoint, requires_input=False)
    finally:
        try:
            client.socket.close(0)
        except Exception:
            pass
        try:
            client.context.term()
        except Exception:
            pass


def _send_command(
    host: str,
    port: int,
    timeout_ms: int,
    endpoint: str,
    payload: Dict[str, Any] | None = None,
):
    payload = payload or {}
    try:
        out = _bridge_call(
            host=host,
            port=port,
            endpoint=endpoint,
            payload=payload,
            requires_input=bool(payload),
            timeout_ms=timeout_ms,
        )
        st.session_state["last_command_result"] = {"ok": True, "endpoint": endpoint, "response": out}
        return True
    except Exception as exc:
        st.session_state["last_command_result"] = {"ok": False, "endpoint": endpoint, "error": str(exc)}
        return False


def _fetch_snapshot(host: str, port: int, timeout_ms: int):
    return _bridge_call(host, port, "get_snapshot", requires_input=False, timeout_ms=timeout_ms)


def _apply_observation_filter(snapshot: Dict[str, Any], obs_mode: str, step_interval: int):
    status = snapshot.get("status", {})
    step_count = int(status.get("step_count", 0))
    env_ready = bool(status.get("env_ready", False))

    if "display_snapshot" not in st.session_state:
        st.session_state["display_snapshot"] = snapshot
        st.session_state["last_seen_step"] = step_count
        return

    if not env_ready:
        st.session_state["display_snapshot"] = snapshot
        st.session_state["last_seen_step"] = step_count
        return

    if obs_mode == "Get obs once action end":
        # Every action click triggers rerun, so just show newest snapshot.
        st.session_state["display_snapshot"] = snapshot
        st.session_state["last_seen_step"] = step_count
        return

    # Interval mode: update displayed observation by env step count.
    last_seen = int(st.session_state.get("last_seen_step", -1))
    if step_count != last_seen:
        if step_interval == 0 or (step_count % (step_interval + 1) == 0):
            st.session_state["display_snapshot"] = snapshot
        st.session_state["last_seen_step"] = step_count


def main():
    cli = _parse_streamlit_args()
    host = cli.host
    port = int(cli.port)
    snapshot_timeout_ms = 1500
    command_timeout_ms = 15000
    camera_columns = 3
    st.set_page_config(page_title="GR00T Bridge UI", layout="wide")
    st.title("GR00T Bridge UI")
    st.caption("API demo for launching a controlled RoboCasa scene, configuring the policy task description, and sending robot actions.")

    if "last_command_result" not in st.session_state:
        st.session_state["last_command_result"] = None

    with st.sidebar:
        st.header("Connection")
        st.write(f"`{host}:{port}`")

        st.header("Observation Mode")
        obs_mode = st.radio(
            "When to update observation?",
            options=["Get obs once action end", "Interval by env steps"],
            index=0,
            help=(
                "'Get obs once action end' updates the displayed observation after a clicked action finishes. "
                "'Interval by env steps' periodically refreshes while the environment step counter changes."
            ),
        )
        env_step_interval = st.slider(
            "Env step interval (0=every step, 1=every 2 steps)",
            0,
            10,
            0,
            1,
            disabled=(obs_mode == "Get obs once action end"),
            help="Controls how often the UI replaces the displayed observation in interval mode.",
        )
        poll_interval_s = st.slider(
            "UI poll interval (seconds)",
            0.5,
            5.0,
            1.0,
            0.1,
            disabled=(obs_mode == "Get obs once action end"),
            help="How often Streamlit asks the bridge for a fresh snapshot in interval mode.",
        )
        auto_refresh = st.checkbox(
            "Auto refresh",
            value=True,
            disabled=(obs_mode == "Get obs once action end"),
            help="When enabled, the page reruns automatically to fetch new bridge state.",
        )

        st.header("Run Settings")
        max_steps = st.number_input(
            "Max steps",
            min_value=-1,
            value=-1,
            step=1,
            help="-1 means no episode step limit. 0 means use RoboCasa's default horizon for the selected task. Positive values set an explicit limit.",
        )

        st.header("Display")
        image_width = st.selectbox(
            "Image width",
            options=["stretch", "content"],
            index=0,
            help="'stretch' fills the available column width. 'content' keeps the image closer to its natural display size.",
        )

        if st.button("Refresh now", width="stretch", help="Immediately fetch and display the latest bridge snapshot."):
            st.rerun()

    # Load registry info
    envs_info = {"all_tasks": [], "atomic_tasks": [], "composite_tasks": []}
    scene_info = {"layout_id_range": [1, 60], "style_id_range": [1, 60], "split_options": ["target", "pretrain", "custom"]}
    try:
        envs_info = _bridge_call(host, port, "list_available_envs", requires_input=False, timeout_ms=snapshot_timeout_ms)
        scene_info = _bridge_call(host, port, "list_scene_config", requires_input=False, timeout_ms=snapshot_timeout_ms)
    except Exception:
        pass

    snapshot = None
    fetch_error = None
    try:
        snapshot = _fetch_snapshot(host, port, snapshot_timeout_ms)
    except Exception as exc:
        fetch_error = str(exc)

    if fetch_error:
        st.error(f"Bridge request failed: {fetch_error}")
        if st.session_state["last_command_result"] is not None:
            st.json(_to_jsonable(st.session_state["last_command_result"]))
        return

    _apply_observation_filter(snapshot, obs_mode, int(env_step_interval))
    display_snapshot = st.session_state.get("display_snapshot", snapshot)
    status = display_snapshot.get("status", {})
    obs = display_snapshot.get("observation")

    left, right = st.columns([0.9, 1.35], gap="large")

    with left:
        st.subheader("1) Launch / Configure Environment")
        all_tasks = envs_info.get("all_tasks", [])
        default_env = cli.env_name if cli.env_name else ("RinseSinkBasin" if "RinseSinkBasin" in all_tasks else (all_tasks[0] if all_tasks else ""))
        env_name = st.selectbox(
            "Task / Env Name",
            options=all_tasks or [default_env],
            index=(all_tasks.index(default_env) if default_env in all_tasks else 0),
            help="The RoboCasa task/environment to launch, such as RinseSinkBasin or OpenDrawer.",
        )
        split_options = scene_info.get("split_options", ["target", "pretrain", "custom"])
        split = st.selectbox(
            "Split",
            options=split_options,
            index=0 if cli.split not in split_options else split_options.index(cli.split),
            help=(
                "RoboCasa scene split. pretrain uses train/pretraining scenes and is usually easier for released policies. "
                "target uses held-out target/test scenes for generalization. custom lets you manually set layout/style IDs."
            ),
        )
        split_ranges = scene_info.get(
            "split_layout_style_ranges",
            {"target": [1, 10], "pretrain": [11, 60], "custom": [1, 60]},
        )
        range_for_split = split_ranges.get(split, [1, 60])
        layout_min, layout_max = int(range_for_split[0]), int(range_for_split[1])
        style_min, style_max = int(range_for_split[0]), int(range_for_split[1])
        split_is_custom = split == "custom"

        id_cols = st.columns(2)
        with id_cols[0]:
            use_layout_id = st.checkbox(
                "Set layout id",
                value=(bool(cli.layout_id.strip()) and split_is_custom),
                disabled=not split_is_custom,
                help="Enable this only with custom split to reproduce a specific kitchen layout.",
            )
            layout_id_value = st.number_input(
                f"Layout id ({layout_min}-{layout_max})",
                min_value=int(layout_min),
                max_value=int(layout_max),
                value=int(cli.layout_id) if cli.layout_id.strip() else int(layout_min),
                step=1,
                disabled=not use_layout_id,
                help="RoboCasa kitchen layout ID. The valid range depends on the selected split.",
            )
        with id_cols[1]:
            use_style_id = st.checkbox(
                "Set style id",
                value=(bool(cli.style_id.strip()) and split_is_custom),
                disabled=not split_is_custom,
                help="Enable this only with custom split to reproduce a specific kitchen visual/object style.",
            )
            style_id_value = st.number_input(
                f"Style id ({style_min}-{style_max})",
                min_value=int(style_min),
                max_value=int(style_max),
                value=int(cli.style_id) if cli.style_id.strip() else int(style_min),
                step=1,
                disabled=not use_style_id,
                help="RoboCasa kitchen style ID. The valid range depends on the selected split.",
            )

        c_launch, c_reset = st.columns(2)
        with c_launch:
            if st.button("Launch / Apply Environment", width="stretch", help="Create or recreate the simulator with the selected task, split, scene IDs, and run settings."):
                payload: Dict[str, Any] = {"env_name": env_name, "split": split}
                payload["max_steps"] = int(max_steps)
                if use_layout_id:
                    payload["layout_id"] = int(layout_id_value)
                if use_style_id:
                    payload["style_id"] = int(style_id_value)
                _send_command(host, port, command_timeout_ms, "configure_environment", payload)
                st.rerun()
        with c_reset:
            if st.button("Reset Episode", width="stretch", help="Reset the current simulator episode without changing the selected environment configuration."):
                _send_command(host, port, command_timeout_ms, "reset")
                st.rerun()

        st.subheader("2) Actions / Policy Configure")
        st.write("**Policy Configure**")
        policy_skills = []
        try:
            skill_resp = _bridge_call(
                host,
                port,
                "list_policy_skills",
                requires_input=False,
                timeout_ms=snapshot_timeout_ms,
            )
            policy_skills = skill_resp.get("skills", [])
        except Exception:
            policy_skills = []

        atomic_tasks = envs_info.get("atomic_tasks", [])
        skill_names = [skill.get("name", "") for skill in policy_skills if skill.get("name")]
        if not skill_names:
            skill_names = atomic_tasks
        default_atomic = status.get("env_name") if status.get("env_name") in skill_names else (skill_names[0] if skill_names else "")
        selected_skill = st.selectbox(
            "Skill / Atomic task",
            options=skill_names or [default_atomic],
            index=(skill_names.index(default_atomic) if default_atomic in skill_names else 0),
            help="Select the atomic RoboCasa skill/task. The bridge resolves this name to the task description sent to GR00T.",
        )
        resolved_skill = next((skill for skill in policy_skills if skill.get("name") == selected_skill), None)
        if resolved_skill is None and selected_skill:
            try:
                resolved_skill = _bridge_call(
                    host,
                    port,
                    "resolve_skill_description",
                    {"name": selected_skill},
                    requires_input=True,
                    timeout_ms=snapshot_timeout_ms,
                )
            except Exception:
                resolved_skill = {"name": selected_skill, "description": selected_skill, "source": "fallback_task_name", "is_template": False}
        policy_description = (resolved_skill or {}).get("description", selected_skill)
        st.text_area(
            "Resolved task description sent to GR00T",
            value=policy_description or "",
            height=90,
            disabled=True,
            help="Read-only. This is the language instruction that will be sent to GR00T when Execute Policy is clicked.",
        )
        st.caption(
            f"Description source: `{(resolved_skill or {}).get('source', 'unknown')}`"
            + (" | template contains unresolved choices" if (resolved_skill or {}).get("is_template") else "")
        )

        magnitude = st.slider(
            "Base move magnitude",
            0.0,
            1.0,
            0.5,
            0.05,
            help="Normalized base motion amount for forward/backward/turn actions.",
        )
        policy_repeat = st.number_input(
            "Policy repeat",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
            help="How many GR00T policy steps to execute with the configured task description.",
        )

        st.write("**Actions**")
        m1, m2 = st.columns(2)
        with m1:
            if st.button("Forward", width="stretch", help="Move the robot base forward in the current environment."):
                _send_command(host, port, command_timeout_ms, "move", {"command": "forward", "magnitude": float(magnitude), "repeat": 1})
                st.rerun()
            if st.button("Turn Left", width="stretch", help="Rotate the robot base left in the current environment."):
                _send_command(host, port, command_timeout_ms, "move", {"command": "turn_left", "magnitude": float(magnitude), "repeat": 1})
                st.rerun()
        with m2:
            if st.button("Backward", width="stretch", help="Move the robot base backward in the current environment."):
                _send_command(host, port, command_timeout_ms, "move", {"command": "backward", "magnitude": float(magnitude), "repeat": 1})
                st.rerun()
            if st.button("Turn Right", width="stretch", help="Rotate the robot base right in the current environment."):
                _send_command(host, port, command_timeout_ms, "move", {"command": "turn_right", "magnitude": float(magnitude), "repeat": 1})
                st.rerun()
        if st.button("Execute Policy", width="stretch", help="Run the GR00T policy using the task description configured above."):
            _send_command(
                host,
                port,
                command_timeout_ms,
                "call_skill",
                {"name": selected_skill, "params": {"repeat": int(policy_repeat)}},
            )
            st.rerun()

    with right:
        st.subheader("3) Status / Observation")
        if not status.get("env_ready", False):
            st.info("Simulator not launched yet. Configure environment on the left and click 'Launch / Apply Environment'.")
        elif obs is None:
            st.warning("No observation available yet.")
        else:
            cameras = obs.get("cameras", {}) if isinstance(obs, dict) else {}
            if cameras:
                video_items = sorted(cameras.items())
            else:
                video_items = sorted((k, obs[k]) for k in obs.keys() if k.startswith("video."))
            video_keys = [key for key, _ in video_items]
            cols = st.columns(max(1, min(camera_columns, len(video_keys) if len(video_keys) > 0 else 1)))
            for idx, (key, value) in enumerate(video_items):
                try:
                    frame = _pick_frame(value)
                    cols[idx % len(cols)].image(frame, caption=key, channels="RGB", width=image_width)
                except Exception as exc:
                    cols[idx % len(cols)].warning(f"{key}: {exc}")

        st.write("**Status**")
        st.json(_to_jsonable(status))
        if isinstance(obs, dict) and obs.get("robot_state") is not None:
            st.write("**Robot state**")
            st.json(_to_jsonable(obs["robot_state"]))
        if st.session_state["last_command_result"] is not None:
            st.write("**Last command result**")
            st.json(_to_jsonable(st.session_state["last_command_result"]))

    if auto_refresh and obs_mode == "Interval by env steps":
        time.sleep(poll_interval_s)
        st.rerun()


if __name__ == "__main__":
    if get_script_run_ctx() is None:
        print("This is a Streamlit app. Please run:")
        print("  streamlit run scripts/demo_agentic_bridge.py -- --host localhost --port 8010")
        raise SystemExit(1)
    main()
