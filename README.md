## NVIDIA Isaac GR00T (for AI 617)

This is the NVIDIA Isaac GR00T fork repo for running RoboCasa benchmark experiments. This fork is based on the original [GR00T code](https://github.com/NVIDIA/Isaac-GR00T) from NVIDIA. Our fork supports training for **GR00T N1.5**.

## TODO
1. Simplify the system's launch and configuration.
2. Expose APIs to make it accessible to OpenCLAW.
   - Task Specification
   - Camera Observation Retrieval
   - Skill Stack List
   - etc.

## Recommended system specs

For inference we recommend a GPU with at least 8 Gb of memory. (5090 Recommended)

### Install
Step0: Create conda env and project folder
```bash
conda create -c conda-forge -n robocasa python=3.11
conda activate robocasa

mkdir -p ~/workbench/AI_617
```

Step1: Install robosuite and robocasa [Document](https://robocasa.ai/docs/build/html/introduction/installation.html)
```bash
cd ~/workbench/AI_617
git clone https://github.com/ARISE-Initiative/robosuite
cd robosuite
pip install -e .

cd ~/workbench/AI_617
git clone https://github.com/robocasa/robocasa
cd robocasa
pip install -e .

python -m robocasa.scripts.setup_macros              # Set up system variables.
python -m robocasa.scripts.download_kitchen_assets   # Caution: Assets to be downloaded are around 10GB.
```

Step2: Install GR00T-N1.5 
```bash
cd ~/workbench/AI_617
git clone git@github.com:cckaixin/GR00T-robocasa.git
cd GR00T-robocasa

pip uninstall torch torchvision torchaudio -y   # if installed other version. uninstall first.
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128   # This is for my 5050 pick one suits your GPU 
pip install -e .
pip install --no-build-isolation flash-attn==2.7.1.post4  # This will take 30-60mins

pip uninstall tensorflow -y
pip install --force-reinstall opencv-python-headless opencv-python
pip install numpy==2.2.5
pip install streamlit
```

## Download Policy CKPT
You need first install huggingface-cli and login with your token
```bash
cd ~/workbench/AI_617
hf download robocasa/robocasa365_checkpoints \
  --include "gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000/*" \
  --local-dir ./checkpoint-60000 \
  --repo-type model
```

## Run single env with mujogo UI
```bash
cd ~/workbench/AI_617/GR00T-robocasa

python scripts/run_single_env.py \
    --model_path ../checkpoint-60000/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000 \
    --env_name RinseSinkBasin \
    --split target
```

## Agentic bridge API (ZeroMQ)
Start a simulator bridge server so another repo/process can connect and control RoboCasa:
```bash
cd ~/workbench/AI_617/GR00T-robocasa
conda activate robocasa
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"   # recommended for flash-attn runtime

python scripts/run_agentic_bridge.py \
    --model_path ../checkpoint-60000/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000 \
    --env_name RinseSinkBasin \
    --split target \
    --port 8010
```

Available endpoints:
- `status`, `reset`, `get_observation`
- `step_policy`, `step_action`
- `move` (`forward` / `backward` / `turn_left` / `turn_right`)
- `configure_environment` (`split`, optional `env_name`, `max_steps`, `layout_id`, `style_id`)
- `get_snapshot` (status + observation + skills in one request)
- `list_atomic_tasks`, `list_composite_tasks`
- `list_skills`, `register_skills`, `call_skill`

Minimal client usage:
```python
from gr00t.eval.service import BaseInferenceClient

client = BaseInferenceClient(host="localhost", port=8010)
print(client.call_endpoint("status", requires_input=False))
obs = client.call_endpoint("get_observation", requires_input=False)
client.call_endpoint("move", {"command": "forward", "magnitude": 0.5, "repeat": 1})
```

## Demo bridge client
`scripts/demo_agentic_bridge.py` is a reference client for building agentic workflow. It:
- connects to `run_agentic_bridge.py`
- fetches status + observation (polling interval configurable)
- sends commands (`move`, `step_policy`, `call_skill`)
- treats atomic task names as skill calls (calling one switches current task)
- configures scene (`split`, `layout_id`, `style_id`, optional `max_steps`) from UI
- supports UI layout/style controls (camera columns, image width, shown panels)
- shows latest camera observations and status in Streamlit dashboard

Run it in a second terminal after the bridge server is up:
```bash
cd ~/workbench/AI_617/GR00T-robocasa
conda activate robocasa
streamlit run scripts/demo_agentic_bridge.py -- --host localhost --port 8010
```

Recommended Streamlit usage:
- **Task selection**: use `Skill Call` dropdown; atomic task names are included as skills.
- **Scene control**: in sidebar `Environment`, choose `split` and optional `layout_id` / `style_id`.
  - Use `split=custom` when setting explicit `layout_id` / `style_id`.
- **Refresh rate**: start with `poll interval = 1.0s`, then lower gradually if stable.
- **Move defaults**: UI move magnitude default is `0.5`.
- **After bridge code changes**: restart both bridge and Streamlit app.

Note:
- Do not run `python scripts/demo_agentic_bridge.py ...`
- Streamlit apps must be started with `streamlit run ...`

## Troubleshooting
If you see:
- `ImportError: ... libstdc++.so.6: version 'CXXABI_1.3.15' not found`
- or failure importing `flash_attn_2_cuda`

Install conda C++ runtime in your env:
```bash
conda activate robocasa
conda install -y -c conda-forge libstdcxx-ng libgcc-ng
```

## Relevant Document
1. Robocasa: https://robocasa.ai/docs/build/html/introduction/overview.html
2. RoboCLAW: https://arxiv.org/abs/2603.11558
3. RoboCLAW (code): https://github.com/MINT-SJTU/RoboClaw