# Learning to Select Visual In-Context Demonstrations

Official code for **Learning to Select Visual In-Context Demonstrations** (CVPR 2026 Findings).

Eugene Lee, Yu-Chi Lin, Jiajie Diao

[[Paper]](https://arxiv.org/abs/2603.26775)

## Overview

Multimodal LLMs can solve visual regression tasks from a few in-context demonstrations. The usual way to choose
those demonstrations is kNN retrieval on image embeddings, which tends to return near-duplicate examples. This
repo implements **LSD (Learning to Select Demonstrations)**, which treats demonstration selection as a sequential
decision problem: a reinforcement learning agent (a Dueling DQN with a query-centric Transformer decoder) picks
the demonstration set one example at a time, rewarded by the MLLM's prediction quality.

Across five visual regression benchmarks, LSD clearly outperforms the baselines on objective, factual regression
tasks (such as age prediction and image quality), while kNN remains the strongest choice on subjective preference
tasks (such as aesthetics and facial beauty).

## Setup

```bash
git clone https://github.com/Yuchi017/Learning-to-Select-Demonstrations.git
cd Learning-to-Select-Demonstrations
conda create -n LSD python=3.10 -y
conda activate LSD
pip install -r requirements.txt
```

Training and evaluation run the MLLM through [vLLM](https://github.com/vllm-project/vllm), so a CUDA GPU is required.

## Data

Download the five datasets and place them where `datasets.yaml` expects them (or edit the paths there):

| Task (`--task_type`) | Dataset | Expected location |
|---|---|---|
| `age_prediction` | [UTKFace](https://susanqq.github.io/UTKFace/) | `data/utk/UTKFace` |
| `aesthetic_score` | [AVA](https://www.kaggle.com/datasets/nicolacarrassi/ava-aesthetic-visual-assessment) | `data/ava/images` |
| `facial_beauty` | [SCUT-FBP5500 v2](https://www.kaggle.com/datasets/pranavchandane/scut-fbp5500-v2-facial-beauty-scores) | `data/fbp5500_v2/Image` |
| `wild_image_quality` | [KonIQ-10k](https://database.mmsp-kn.de/koniq-10k-database.html) | `data/koniq/512x384` |
| `modified_image_quality` | [KADID-10k](https://database.mmsp-kn.de/kadid-10k-database.html) | `data/kadid/images` |

Then build the train/test splits and SigLIP embeddings for every task in `datasets.yaml`:

```bash
bash scripts/prepare_data.sh
```

## Training

```bash
bash scripts/train_dqn.sh
```

This trains the LSD agent for one task and one MLLM (default: `age_prediction` with `google/gemma-3-4b-it`).
The script contains commented-out commands for the other tasks and for Qwen2.5-VL-7B, InternVL2-8B and
Phi-3.5-vision. Results and checkpoints are written to `train_res/<task>/<model>/run_<id>/`.

## Evaluation

Set `CHECKPOINT` in `scripts/eval_dqn.sh` to a trained checkpoint, then run:

```bash
bash scripts/eval_dqn.sh
```

This compares LSD against random and kNN selection at the given number of shots and writes results to `eval_res/`.

Both scripts use GPU 0 by default; override with `CUDA_VISIBLE_DEVICES=<id> bash scripts/...`.

## Citation

```bibtex
@inproceedings{lee2026learning,
  title     = {Learning to Select Visual In-Context Demonstrations},
  author    = {Lee, Eugene and Lin, Yu-Chi and Diao, Jiajie},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  pages     = {9455--9465},
  year      = {2026}
}
```
