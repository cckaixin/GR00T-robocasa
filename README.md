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
cd ~/workbench/AI_617/R00T-robocasa

python scripts/run_single_env.py \
    --model_path ../checkpoint-60000/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000 \
    --env_name RinseSinkBasin \
    --split target
```

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