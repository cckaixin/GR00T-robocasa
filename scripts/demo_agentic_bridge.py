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
    parser.add_argument("--split", type=str, default="target")
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
    st.caption("Launch env from UI, call skills/actions, and view latest observations.")

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
        )
        env_step_interval = st.slider(
            "Env step interval (0=every step, 1=every 2 steps)",
            0,
            10,
            0,
            1,
            disabled=(obs_mode == "Get obs once action end"),
        )
        poll_interval_s = st.slider(
            "UI poll interval (seconds)",
            0.5,
            5.0,
            1.0,
            0.1,
            disabled=(obs_mode == "Get obs once action end"),
        )
        auto_refresh = st.checkbox("Auto refresh", value=True, disabled=(obs_mode == "Get obs once action end"))

        st.header("Display")
        image_width = st.selectbox("Image width", options=["stretch", "content"], index=0)

        if st.button("Refresh now", width="stretch"):
            st.rerun()

    # Load registry info
    envs_info = {"all_tasks": [], "atomic_tasks": [], "composite_tasks": []}
    scene_info = {"layout_id_range": [1, 60], "style_id_range": [1, 60], "split_options": ["target", "pretrain", "custom"]}
    try:
        envs_info = _bridge_call(host, port, "list_available_envs", requires_input=False, timeout_ms=snapshot_timeout_ms)
        scene_info = _bridge_call(host, port, "list_scene_config", requires_input=False, timeout_ms=snapshot_timeout_ms)
    except Exception:
        pass

    # Environment launch panel
    st.subheader("1) Launch / Configure Environment")
    e1, e2, e3, e4 = st.columns([2, 1, 1, 1])
    all_tasks = envs_info.get("all_tasks", [])
    default_env = cli.env_name if cli.env_name else ("RinseSinkBasin" if "RinseSinkBasin" in all_tasks else (all_tasks[0] if all_tasks else ""))
    with e1:
        env_name = st.selectbox("Task / Env Name", options=all_tasks or [default_env], index=(all_tasks.index(default_env) if default_env in all_tasks else 0))
    with e2:
        split = st.selectbox("Split", options=scene_info.get("split_options", ["target", "pretrain", "custom"]), index=0 if cli.split not in scene_info.get("split_options", []) else scene_info.get("split_options", []).index(cli.split))
    split_ranges = scene_info.get(
        "split_layout_style_ranges",
        {"target": [1, 10], "pretrain": [11, 60], "custom": [1, 60]},
    )
    range_for_split = split_ranges.get(split, [1, 60])
    layout_min, layout_max = int(range_for_split[0]), int(range_for_split[1])
    style_min, style_max = int(range_for_split[0]), int(range_for_split[1])
    split_is_custom = split == "custom"
    with e3:
        use_layout_id = st.checkbox(
            "Set layout id",
            value=(bool(cli.layout_id.strip()) and split_is_custom),
            disabled=not split_is_custom,
        )
        layout_id_value = st.number_input(
            f"Layout id ({layout_min}-{layout_max})",
            min_value=int(layout_min),
            max_value=int(layout_max),
            value=int(cli.layout_id) if cli.layout_id.strip() else int(layout_min),
            step=1,
            disabled=not use_layout_id,
        )
    with e4:
        use_style_id = st.checkbox(
            "Set style id",
            value=(bool(cli.style_id.strip()) and split_is_custom),
            disabled=not split_is_custom,
        )
        style_id_value = st.number_input(
            f"Style id ({style_min}-{style_max})",
            min_value=int(style_min),
            max_value=int(style_max),
            value=int(cli.style_id) if cli.style_id.strip() else int(style_min),
            step=1,
            disabled=not use_style_id,
        )

    if split_is_custom:
        st.caption("If layout/style is not set, RoboCasa samples one automatically.")
    else:
        st.caption(f"For split `{split}`, RoboCasa uses its predefined scene split; explicit layout/style overrides are disabled.")

    max_steps = st.number_input("Max steps (0=task default)", min_value=0, value=0, step=1)
    c_launch, c_reset = st.columns(2)
    with c_launch:
        if st.button("Launch / Apply Environment", width="stretch"):
            payload: Dict[str, Any] = {"env_name": env_name, "split": split}
            if max_steps > 0:
                payload["max_steps"] = int(max_steps)
            if use_layout_id:
                payload["layout_id"] = int(layout_id_value)
            if use_style_id:
                payload["style_id"] = int(style_id_value)
            _send_command(host, port, command_timeout_ms, "configure_environment", payload)
            st.rerun()
    with c_reset:
        if st.button("Reset Episode", width="stretch"):
            _send_command(host, port, command_timeout_ms, "reset")
            st.rerun()

    # Command panel
    st.subheader("2) Actions / Skills")
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
    skills = display_snapshot.get("skills", {"skills": []})

    a1, a2 = st.columns([1, 1])
    with a1:
        st.write("**Base Move**")
        magnitude = st.slider("Magnitude", 0.0, 1.0, 0.5, 0.05)
        m1, m2 = st.columns(2)
        with m1:
            if st.button("Forward", width="stretch"):
                _send_command(host, port, command_timeout_ms, "move", {"command": "forward", "magnitude": float(magnitude), "repeat": 1})
                st.rerun()
            if st.button("Turn Left", width="stretch"):
                _send_command(host, port, command_timeout_ms, "move", {"command": "turn_left", "magnitude": float(magnitude), "repeat": 1})
                st.rerun()
        with m2:
            if st.button("Backward", width="stretch"):
                _send_command(host, port, command_timeout_ms, "move", {"command": "backward", "magnitude": float(magnitude), "repeat": 1})
                st.rerun()
            if st.button("Turn Right", width="stretch"):
                _send_command(host, port, command_timeout_ms, "move", {"command": "turn_right", "magnitude": float(magnitude), "repeat": 1})
                st.rerun()

    with a2:
        st.write("**Policy Call**")
        all_skills = skills.get("skills", []) or ["policy_step"]
        base_move_skills = {"move_forward", "move_backward", "turn_left", "turn_right"}
        policy_skills = [s for s in all_skills if s not in base_move_skills]
        if len(policy_skills) == 0:
            policy_skills = ["policy_step"]
        skill_name = st.selectbox("Policy skill", options=policy_skills)
        policy_description = None
        if skill_name == "policy_step":
            desc_options = []
            try:
                desc_resp = _bridge_call(
                    host,
                    port,
                    "list_policy_descriptions",
                    requires_input=False,
                    timeout_ms=snapshot_timeout_ms,
                )
                desc_options = desc_resp.get("policy_descriptions", [])
            except Exception:
                desc_options = []
            desc_source = st.selectbox(
                "Policy description source",
                options=["Use env instruction", "Use task name", "Custom text"],
                index=0,
            )
            if desc_source == "Custom text":
                policy_description = st.text_input("Custom description", value="")
            elif desc_source == "Use task name":
                policy_description = status.get("env_name")
            else:
                policy_description = desc_options[0] if len(desc_options) > 0 else None
        if st.button("Step Policy", width="stretch"):
            params: Dict[str, Any] = {}
            if skill_name == "policy_step" and policy_description is not None:
                params["description"] = policy_description
            _send_command(host, port, command_timeout_ms, "call_skill", {"name": skill_name, "params": params})
            st.rerun()

    st.subheader("3) Status / Observation")
    s1, s2 = st.columns([1, 2])
    with s2:
        if not status.get("env_ready", False):
            st.info("Simulator not launched yet. Configure environment above and click 'Launch / Apply Environment'.")
        elif obs is None:
            st.warning("No observation available yet.")
        else:
            video_keys = sorted([k for k in obs.keys() if k.startswith("video.")])
            cols = st.columns(max(1, min(camera_columns, len(video_keys) if len(video_keys) > 0 else 1)))
            for idx, key in enumerate(video_keys):
                try:
                    frame = _pick_frame(obs[key])
                    cols[idx % len(cols)].image(frame, caption=key, channels="RGB", width=image_width)
                except Exception as exc:
                    cols[idx % len(cols)].warning(f"{key}: {exc}")

    with s1:
        st.json(_to_jsonable(status))
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
