#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Streamlit demo UI for scripts/run_agentic_bridge.py.

Run:
    conda activate robocasa
    streamlit run scripts/demo_agentic_bridge.py
"""

import json
import time
from typing import Any, Dict

import numpy as np
import streamlit as st
import zmq
from streamlit.runtime.scriptrunner import get_script_run_ctx

from gr00t.eval.service import BaseInferenceClient


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


def _shape_dtype(x: Any) -> Dict[str, Any]:
    if isinstance(x, np.ndarray):
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    return {"type": str(type(x))}


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
    timeout_ms: int = 800,
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


def _send_command(host: str, port: int, timeout_ms: int, endpoint: str, payload: Dict[str, Any] | None = None):
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
    except Exception as exc:
        st.session_state["last_command_result"] = {"ok": False, "endpoint": endpoint, "error": str(exc)}


def main():
    st.set_page_config(page_title="GR00T Agentic Bridge Demo", layout="wide")
    st.title("GR00T Agentic Bridge Demo")
    st.caption("Reference dashboard for teammates to receive observation, send commands, and inspect status.")

    if "last_command_result" not in st.session_state:
        st.session_state["last_command_result"] = None

    # Sidebar: connection + polling
    with st.sidebar:
        st.header("Connection")
        host = st.text_input("Host", value="localhost")
        port = st.number_input("Port", value=8010, step=1)
        timeout_ms = st.slider("Socket timeout (ms)", 200, 5000, 800, 100)

        st.header("Environment")
        split_choice = st.selectbox("Split", options=["target", "pretrain", "custom"], index=0)
        layout_text = st.text_input("Layout id (custom split only)", value="")
        style_text = st.text_input("Style id (custom split only)", value="")
        max_steps_input = st.number_input("Max steps (0 = task default)", min_value=0, value=0, step=1)
        if st.button("Apply scene config", width="stretch"):
            payload: Dict[str, Any] = {"split": split_choice}
            if int(max_steps_input) > 0:
                payload["max_steps"] = int(max_steps_input)
            if layout_text.strip():
                payload["layout_id"] = int(layout_text.strip())
            if style_text.strip():
                payload["style_id"] = int(style_text.strip())
            _send_command(host, int(port), int(timeout_ms), "configure_environment", payload)

        st.header("Polling")
        auto_refresh = st.checkbox("Auto refresh", value=True)
        poll_interval = st.slider("Poll interval (seconds)", 0.5, 5.0, 1.0, 0.1)

        st.header("Layout / Style")
        camera_columns = st.slider("Camera columns", 1, 3, 3, 1)
        image_width_mode = st.selectbox("Image width", options=["stretch", "content"], index=0)
        show_obs_keys = st.checkbox("Show observation keys", value=True)
        show_state_meta = st.checkbox("Show state metadata", value=True)

        if st.button("Refresh now", width="stretch"):
            st.rerun()

    # Command panel
    status = None
    obs = None
    skills = {"skills": []}
    err = None
    try:
        snap = _bridge_call(host, int(port), "get_snapshot", requires_input=False, timeout_ms=int(timeout_ms))
        status = snap.get("status", {})
        obs = snap.get("observation", {})
        skills = snap.get("skills", {"skills": []})
    except Exception as exc:
        err = str(exc)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.subheader("Episode")
        if st.button("Reset", width="stretch"):
            _send_command(host, int(port), int(timeout_ms), "reset")
        policy_repeat = st.number_input("Policy repeat", value=1, step=1, min_value=1)
        if st.button("Step policy", width="stretch"):
            _send_command(host, int(port), int(timeout_ms), "step_policy", {"repeat": int(policy_repeat)})

    with c2:
        st.subheader("Move")
        magnitude = st.slider("Magnitude", 0.0, 1.0, 0.5, 0.05)
        repeat = st.number_input("Repeat", value=1, step=1, min_value=1)
        m1, m2 = st.columns(2)
        with m1:
            if st.button("Forward", width="stretch"):
                _send_command(host, int(port), int(timeout_ms), "move", {"command": "forward", "magnitude": float(magnitude), "repeat": int(repeat)})
            if st.button("Turn Left", width="stretch"):
                _send_command(host, int(port), int(timeout_ms), "move", {"command": "turn_left", "magnitude": float(magnitude), "repeat": int(repeat)})
        with m2:
            if st.button("Backward", width="stretch"):
                _send_command(host, int(port), int(timeout_ms), "move", {"command": "backward", "magnitude": float(magnitude), "repeat": int(repeat)})
            if st.button("Turn Right", width="stretch"):
                _send_command(host, int(port), int(timeout_ms), "move", {"command": "turn_right", "magnitude": float(magnitude), "repeat": int(repeat)})

    with c3:
        st.subheader("Skill Call")
        st.caption("Atomic tasks are included in this list; calling one switches the current task.")
        skill_options = skills.get("skills", [])
        if len(skill_options) == 0:
            skill_options = ["policy_step"]
        skill_name = st.selectbox("Skill name", options=skill_options)
        skill_params_text = st.text_area("Skill params (JSON)", value="{}", height=100)
        if st.button("Call skill", width="stretch"):
            try:
                skill_params = json.loads(skill_params_text) if skill_params_text.strip() else {}
                _send_command(
                    host,
                    int(port),
                    int(timeout_ms),
                    "call_skill",
                    {"name": skill_name, "params": skill_params},
                )
            except Exception as exc:
                st.session_state["last_command_result"] = {
                    "ok": False,
                    "endpoint": "call_skill",
                    "error": f"Invalid JSON params: {exc}",
                }

    if err:
        st.error(f"Bridge request failed: {err}")
    else:
        left, right = st.columns([1, 2])
        with left:
            st.subheader("Status")
            st.json(_to_jsonable(status))

            st.subheader("Skills")
            st.json(_to_jsonable(skills))

            st.subheader("Last command result")
            if st.session_state["last_command_result"] is None:
                st.write("No command sent yet.")
            else:
                st.json(_to_jsonable(st.session_state["last_command_result"]))

            if show_state_meta:
                state_dict = {k: _shape_dtype(v) for k, v in obs.items() if k.startswith("state.")}
                st.subheader("State keys")
                st.json(state_dict)

        with right:
            st.subheader("Latest observation (cameras)")
            video_keys = sorted([k for k in obs.keys() if k.startswith("video.")])
            cols = st.columns(max(1, min(camera_columns, len(video_keys) if len(video_keys) > 0 else 1)))
            for idx, key in enumerate(video_keys):
                try:
                    frame = _pick_frame(obs[key])
                    cols[idx % len(cols)].image(frame, caption=key, channels="RGB", width=image_width_mode)
                except Exception as exc:
                    cols[idx % len(cols)].warning(f"{key}: {exc}")

            if show_obs_keys:
                st.subheader("Observation keys")
                st.write(sorted(obs.keys()))

    # Controlled pull frequency:
    # The bridge is request-response. The client decides update frequency by polling.
    if auto_refresh:
        time.sleep(poll_interval)
        st.rerun()


if __name__ == "__main__":
    # Avoid noisy warnings when users run `python ...` instead of `streamlit run ...`.
    if get_script_run_ctx() is None:
        print("This is a Streamlit app. Please run:")
        print("  streamlit run scripts/demo_agentic_bridge.py")
        raise SystemExit(1)
    main()
