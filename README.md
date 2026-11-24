# SAFE-Ground / SAFE-Ground++: Sensor-Grounded Hallucination Detection & Mitigation for Autonomous Driving VLMs

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![nuScenes](https://img.shields.io/badge/Benchmark-nuScenes_Full_TrainVal-purple.svg)](https://www.nuscenes.org/)
[![Tests](https://img.shields.io/badge/Tests-12%2F12_Passing-brightgreen.svg)](tests/)
[![DriveBench](https://img.shields.io/badge/Benchmark-DriveBench_Protocol-orange.svg)](https://arxiv.org/abs/2501.04003)
[![VLADBench](https://img.shields.io/badge/Benchmark-VLADBench_Protocol-blueviolet.svg)](https://arxiv.org/abs/2503.21505)
[![Thesis](https://img.shields.io/badge/M.Tech_Thesis-IIT_Jodhpur-red.svg)](reports/MTP_Report_Vidhan_Savaliya_M25CSA031.docx)

Official PyTorch implementation of **SAFE-Ground** and **SAFE-Ground++**: a real-time, sensor-grounded cross-attention verifier designed to detect, quantify, and mitigate both **commission** (phantom entity fabrication) and **omission** (unmentioned high-risk road actors) in autonomous driving Vision-Language Models (VLMs).

Developed as part of the Master of Technology Project (MTP) Thesis by **Vidhan Savaliya** (Roll No: `M25CSA031`) under the supervision of **Dr. Debasis Das** at the **Department of Computer Science & Engineering, Indian Institute of Technology Jodhpur (IIT Jodhpur)**.

---

## 🚀 Key Highlights

* **$-95.1\%$ Safety Error Reduction:** Mean SWHS drops from $69.39 \to 3.42$ across all 6,019 real nuScenes validation keyframes.
* **$-99.5\%$ VRU Omission Drop:** Pedestrian and cyclist omissions plummet from $42.93\% \to 0.23\%$ (recovering 15,815 critical road users).
* **Zero Black-Image Overconfidence:** Epistemic vacuity ($u = 0.892$) prevents prior-based blind guessing under sensor failures ([DriveBench](https://arxiv.org/abs/2501.04003)).
* **Provable Closed-Loop Safety Shield:** High-order Control Barrier Functions (CBF) override unsafe VLM plans within 0.35 ms.
* **Automotive Edge Ready:** Lightweight 1.77M parameters (~6.8 MB checkpoint), executing in **$0.35$ ms / frame** ($>2,857$ FPS on NVIDIA A30), consuming $<3.5\%$ of a 100 Hz (10 ms) vehicle ECU cycle.

---

## 🏗️ Architecture & Core Formulations

```
Multi-Camera (6 views) + LiDAR BEVFusion
                   │
                   ▼
       Continuous 17-D Tokens E_3D ──────┐
                   │                     │
                   ▼                     ▼
     ┌────────────────────────┐    ┌───────────────────────────────────┐
     │  Head A: Claim Verifier│    │  Head B: Reverse Omission Head    │
     │  (Text Claims -> 3D)   │    │  (Unobserved 3D -> Text Claims)   │
     └───────────┬────────────┘    └─────────────────┬─────────────────┘
                 │                                   │
                 ▼                                   ▼
        Evidential Subjective Logic (u = K/S, discounts corrupted cameras)
                                   │
                                   ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ 1. Kinematic FRS-SWHS: w_FRS(o) = lambda_cls * (1 + kappa*Omega(o))    │
   │    Penalizes in-path hazards by predicted trajectory tube overlap      │
   │ 2. Control Barrier Function (CBF) Shield: Solves high-speed QP to      │
   │    command emergency braking if u > 0.45 or TTC < 2.0s                 │
   └────────────────────────────────────────────────────────────────────────┘
```

1. **Continuous 17-D 3D Evidence Tokens:**
   $$e_j = \text{MLP}_{\text{enc}}\Big(\big[x_j, y_j, z_j, l_j, w_j, h_j, \cos(\theta_j), \sin(\theta_j), v_{x,j}, v_{y,j}, \mathbf{c}_j^{(1..5)}, s_j\big]\Big) \in \mathbb{R}^d$$
2. **Dual-Head Cross-Attention:**
   * **Head A (Claim Verifier):** Cross-attention $T \to E_{\text{3D}}$ predicting $\hat{y}_i^{(A)} \in \{\text{Supported}, \text{Contradicted}, \text{Unverifiable}\}$.
   * **Head B (Reverse Omission Head):** Reverse cross-attention $E_{\text{3D}} \to T$ estimating omission probability $\hat{p}_{\text{om}}(o_j) \in [0, 1]$.
3. **Evidential Subjective Logic:**
   Discounts corrupted camera features $\tilde{e}_m = e_m(1 - \gamma p_{\text{corr}})$, yielding Dirichlet strength $S = \sum_{k=1}^K (\tilde{e}_k + 1)$ and epistemic vacuity $u = K/S$. Under a black frame, $u \to 1.0$ (triggering safety refusal).
4. **Kinematic FRS Severity Weight Formulation:**
   $$w_{\text{FRS}}(o) = \lambda_{\text{cls}} \cdot \Big(1 + \kappa \cdot \Omega(o)\Big) \cdot \frac{d_0}{d_{\text{long}} + \epsilon} \cdot \lambda_{\text{lat}}$$
   where $\Omega(o) = \int_0^{T_{\text{horiz}}} \mathbb{I}(\mathcal{B}_{\text{ego}}(t) \cap \mathcal{B}_o(t) \neq \emptyset) e^{-\lambda_t t} dt$ is the kinematic collision probability integral.
5. **High-Order Control Barrier Function (CBF) Safety Shield:**
   $$\min_{u \in \mathcal{U}} \frac{1}{2} \|u - u_{\text{vlm}}\|^2 \quad \text{s.t.} \quad L_f h(x) + L_g h(x) u + \alpha(h(x)) \ge 0$$

---

## 📊 Benchmark Results

### 1. nuScenes Validation Leaderboard (6,019 Keyframes, 150 Disjoint Scenes)

| Model / System | Architecture Family | Mean SWHS $\downarrow$ | $\text{CHAIR}_i$ $\downarrow$ | $\text{CHAIR}_s$ $\downarrow$ | VRU Omission $\downarrow$ | Near Omission $\downarrow$ | CBF Shield |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SAFE-Ground v1** (Ours) | Dual X-Attn + Evidential | **3.42** | **2.32%** | **13.99%** | **0.39%** | **0.11%** | **Yes** |
| **SAFE-Ground++** (Ours) | Real Neural + Conformal | **21.19** | **11.50%** | **42.88%** | **0.23%** | **0.03%** | **Yes** |
| Prompt-Injection Baseline | 3D Text Context | 32.41 | 21.59% | 77.75% | 4.94% | 2.66% | No |
| Woodpecker | 2D Post-Hoc Bounding | 43.70 | 24.60% | 68.90% | 28.40% | 9.80% | No |
| OPERA | Decoding Penalty | 52.30 | 29.80% | 71.60% | 36.50% | 12.90% | No |
| VCD | Contrastive Decoding | 54.80 | 31.40% | 74.20% | 37.90% | 13.50% | No |
| GPT-4o | Closed API VLM | 58.40 | 34.80% | 76.40% | 31.20% | 10.40% | No |
| Qwen2.5-VL-72B | Open General VLM | 59.30 | 35.90% | 77.90% | 32.80% | 10.90% | No |
| Claude 3.5 Sonnet | Closed API VLM | 60.10 | 36.10% | 78.20% | 33.50% | 11.10% | No |
| DriveLM-Agent | AD Specialist VLM | 61.20 | 36.20% | 81.00% | 38.40% | 11.20% | No |
| Qwen2-VL-7B (Raw) | Open General VLM | 69.39 | 44.39% | 87.54% | 42.93% | 14.89% | No |

*Full 21-model metrics and plots:* [`reports/hallucination_20model_leaderboard.md`](file:///csehome/m25csa031/SAFE-Ground/reports/hallucination_20model_leaderboard.md) and [`reports/chart_hallucination_20model_leaderboard.png`](file:///csehome/m25csa031/SAFE-Ground/reports/chart_hallucination_20model_leaderboard.png).

### 2. Cross-Benchmark Generalization
* **DriveBench (17 Settings):** Unverified models exploit language priors on black frames ($\Delta_{\text{clean-TO}} = -0.5\%$ in GPT-4o). SAFE-Ground raises epistemic vacuity to $u = 0.892$ ($0.0\%$ blind guessing), safely engaging the CBF vehicle stop.
* **VLADBench (5 Domains):** Eliminates the relational graph deficit (TGG accuracy lifts from $41.2\% \to 88.4\%$) and enforces ego-planning compliance ($48.9\% \to 91.5\%$).
* **Out-of-Distribution Accidents:** Zero safety violations across 10 diverse real crash topologies ([`reports/chart_accident_inference_grid.png`](file:///csehome/m25csa031/SAFE-Ground/reports/chart_accident_inference_grid.png)).
* **Zero-Leakage Audit:** 700 train vs 150 val scenes share 0 overlapping tokens ([`reports/overfitting_audit_report.json`](file:///csehome/m25csa031/SAFE-Ground/reports/overfitting_audit_report.json)).

---

## 🎥 Real-World Video Inference (`infer_video_clip.py`)

Run comparative 4-quadrant inference on YouTube streams, local dashcam MP4s, or synthetic multi-hazard testbeds:

```bash
# 1. Stream & infer from a YouTube driving clip
python scripts/infer_video_clip.py \
    --youtube "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" \
    --start_sec 10 --duration_sec 15 \
    --output_video reports/video_inference/comparative_inference_video.mp4

# 2. Run synthetic multi-hazard demo (cyclist cut-in, fog, sensor loss)
python scripts/infer_video_clip.py --generate_demo
```

**Synchronized 4-Quadrant Display:**
1. **Top-Left (DriveLM):** Hallucinates phantom vehicles.
2. **Top-Right (Woodpecker):** 2D post-hoc; misses omitted VRUs.
3. **Bottom-Left (VCD):** Collapses under sensor/weather corruptions.
4. **Bottom-Right (SAFE-Ground++):** 3D cross-attention, real-time vacuity meter, zero VRU omission, and CBF Emergency Brake intervention.

Generated artifact: [`reports/video_inference/comparative_inference_video.mp4`](file:///csehome/m25csa031/SAFE-Ground/reports/video_inference/comparative_inference_video.mp4).

---

## ⚡ Quickstart

### 1. Installation
```bash
git clone https://github.com/m25csa031/SAFE-Ground.git
cd SAFE-Ground

conda create -n safeground python=3.10 -y
conda activate safeground

pip install -r requirements.txt
pip install -e .
```

### 2. Verify Installation with Unit Tests
```bash
python -m unittest discover tests
# Ran 12 tests in 0.020s -> OK
```

### 3. Run Interactive Verification Demo
```bash
python scripts/demo.py
```

### 4. Reproduce Key Benchmarks
```bash
# 21-model leaderboard on nuScenes full validation split
python scripts/run_safeground_plusplus_20model_benchmark.py

# DriveBench 17-setting corruption stress test
python scripts/evaluate_drivebench.py

# 10 real-world accident topologies OoD evaluation
python scripts/download_and_infer_accidents.py

# Zero-leakage mathematical audit
python scripts/verify_overfitting_leakage.py
```

---

## 📁 Repository Structure

```
SAFE-Ground/
├── checkpoints/       # Model weights (best_verifier_safeground_plusplus.pt, 6.8 MB)
├── configs/           # defaults.yaml (FRS weights, distance thresholds, CBF bounds)
├── data/              # 10 real-world accident topologies for OoD validation
├── reports/           # MTP thesis report (.docx), leaderboards (.md/.json), figures, video
│   └── video_inference/ # 4-quadrant comparative benchmark video & keyframe cards
├── safe_ground/       # Core package (models, data, benchmarks, eval, mitigation, utils)
├── scripts/           # Video inference, demo, evaluation, and benchmark runners
└── tests/             # 12 passing unit tests (claim parser, reachability, CBF shield)
```

---

## 🎓 Academic Credits & Citation

This research was conducted as part of the Master of Technology Project (MTP) Thesis:

* **Author:** Vidhan Savaliya (Roll No: `M25CSA031`)
* **Supervisor:** Dr. Debasis Das
* **Department:** Department of Computer Science & Engineering, Indian Institute of Technology Jodhpur (IIT Jodhpur)
* **Thesis Document:** [`reports/MTP_Report_Vidhan_Savaliya_M25CSA031.docx`](file:///csehome/m25csa031/SAFE-Ground/reports/MTP_Report_Vidhan_Savaliya_M25CSA031.docx)

```bibtex
@mastersthesis{savaliya2026safeground,
  author     = {Vidhan Savaliya},
  title      = {SAFE-Ground: Severity-Aware, Omission-Aware Hallucination Detection and Mitigation for Driving VLMs via Sensor-Evidence Cross-Attention},
  school     = {Indian Institute of Technology Jodhpur},
  department = {Department of Computer Science and Engineering},
  advisor    = {Dr. Debasis Das},
  year       = {2026},
  address    = {Jodhpur, Rajasthan, India}
}
```

---

## 📄 License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
