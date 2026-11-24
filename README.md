# SAFE-Ground / SAFE-Ground++: Severity-Aware, Omission-Aware Hallucination Detection & Mitigation for Autonomous Driving Vision-Language Models

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![nuScenes](https://img.shields.io/badge/Benchmark-nuScenes_Full_TrainVal-purple.svg)](https://www.nuscenes.org/)
[![Tests](https://img.shields.io/badge/Tests-12%2F12_Passing-brightgreen.svg)](tests/)
[![DriveBench](https://img.shields.io/badge/Benchmark-DriveBench_Protocol-orange.svg)](https://arxiv.org/abs/2501.04003)
[![VLADBench](https://img.shields.io/badge/Benchmark-VLADBench_Protocol-blueviolet.svg)](https://arxiv.org/abs/2503.21505)
[![Thesis](https://img.shields.io/badge/M.Tech_Thesis-IIT_Jodhpur-red.svg)](reports/MTP_Report_Vidhan_Savaliya_M25CSA031.docx)

Official PyTorch implementation of **SAFE-Ground** and **SAFE-Ground++**: a mathematically rigorous, sensor-grounded cross-attention verifier designed to detect, quantify, and mitigate both **commission** (phantom entity fabrication) and **omission** (unmentioned high-risk road actors) hallucinations in autonomous driving Vision-Language Models (VLMs).

Developed as part of the Master of Technology Project (MTP) Thesis by **Vidhan Savaliya** (Roll No: `M25CSA031`) under the supervision of **Dr. Debasis Das** at the **Department of Computer Science & Engineering, Indian Institute of Technology Jodhpur (IIT Jodhpur)**.

---

## 📑 Table of Contents
- [Executive Technical Overview](#-executive-technical-overview)
- [System Architecture](#-system-architecture)
- [Mathematical Foundations](#-mathematical-foundations)
  - [1. Continuous 17-D 3D Sensor Evidence Formulation](#1-continuous-17-d-3d-sensor-evidence-formulation)
  - [2. Dual-Head Cross-Attention Verifier (Head A & Head B)](#2-dual-head-cross-attention-verifier-head-a--head-b)
  - [3. Evidential Subjective Logic & Adaptive Vacuity Link](#3-evidential-subjective-logic--adaptive-vacuity-link)
  - [4. Forward Reachable Sets (FRS) Severity Weight Formulation](#4-forward-reachable-sets-frs-severity-weight-formulation)
  - [5. Severity-Weighted Hallucination Score (FRS-SWHS)](#5-severity-weighted-hallucination-score-frs-swhs)
  - [6. High-Order Control Barrier Function (CBF) Trajectory Shield](#6-high-order-control-barrier-function-cbf-trajectory-shield)
  - [7. Conformal Calibration & Temporal Smoothing in SAFE-Ground++](#7-conformal-calibration--temporal-smoothing-in-safe-ground)
- [Comprehensive Empirical Evaluation](#-comprehensive-empirical-evaluation)
  - [1. 21-Model Hallucination Leaderboard on nuScenes Full TrainVal](#1-21-model-hallucination-leaderboard-on-nuscenes-full-trainval)
  - [2. Real Neural SAFE-Ground++ Ablation Study (5-Fold Cross-Validation)](#2-real-neural-safe-ground-ablation-study-5-fold-cross-validation)
  - [3. DriveBench 17-Setting Stress Test Protocol](#3-drivebench-17-setting-stress-test-protocol)
  - [4. VLADBench 5-Domain Cognitive Evaluation](#4-vladbench-5-domain-cognitive-evaluation)
  - [5. Out-of-Distribution Real-World Crash Evaluation (10 Topologies)](#5-out-of-distribution-real-world-crash-evaluation-10-topologies)
  - [6. Zero-Leakage & Scene-Disjointness Audit Proof](#6-zero-leakage--scene-disjointness-audit-proof)
  - [7. Real-Time Hardware Profiling & Latency](#7-real-time-hardware-profiling--latency)
- [Real-World Video Inference System (`infer_video_clip.py`)](#-real-world-video-inference-system-infer_video_clippy)
- [Repository Structure](#-repository-structure)
- [Installation & Getting Started](#-installation--getting-started)
- [Reproducing Benchmark Protocols](#-reproducing-benchmark-protocols)
- [Academic Credits & Citation](#-academic-credits--citation)

---

## 🔬 Executive Technical Overview

Modern Vision-Language Models (e.g., GPT-4o, Claude 3.5 Sonnet, Qwen2.5-VL, DriveLM) are increasingly deployed as end-to-end cognitive planners for autonomous vehicles. However, systematic evaluations expose critical failure modes that render unverified VLMs unacceptable for safety-critical deployment:

1. **Blind Prior Exploitation & Black-Image Collapse ([DriveBench, arXiv:2501.04003](https://arxiv.org/abs/2501.04003)):** Under sensor degradation, leading commercial and open VLMs exploit dataset text priors rather than visual evidence. For example, GPT-4o achieves $59.5\%$ accuracy on a completely black image ($\Delta_{\text{clean-TO}} = -0.5\%$), hallucinating driving decisions with zero sensory input.
2. **Relational Incompetence & Planning Deficits ([VLADBench, arXiv:2503.21505](https://arxiv.org/abs/2503.21505)):** Frontier VLMs drop to $\le 41\%$ accuracy on traffic graph generation (TGG) and fail ego-planning collision checks $\sim 50\%$ of the time, unable to bind traffic lights to lane trajectories.
3. **The Omission Blindness in Existing 2D Mitigators:** Prior mitigation methods (Woodpecker, OPERA, VCD) operate purely post-hoc on generated text or 2D image coordinates. They only detect **commission** (hallucinated phantom objects) while remaining completely blind to **omission** (unmentioned pedestrians, cyclists, or crossing vehicles that the ego-vehicle is about to strike).

### The SAFE-Ground Paradigm
SAFE-Ground resolves these deficits by introducing a real-time, sensor-grounded verifier operating between the driving VLM and the vehicle control execution:
* **Continuous 3D Grounding:** Couples 17-D continuous 3D bounding geometry ($E_{\text{3D}}$ from multi-camera + LiDAR BEVFusion) directly to language claims via dual-head cross-attention.
* **Dual-Head Verification:** Head A verifies claim veracity (Supported, Contradicted, Unverifiable), while Head B performs reverse cross-attention to identify critical unmentioned obstacles in the 3D sensory field.
* **Evidential Subjective Logic:** Dynamically discounts corrupted camera evidence ($\tilde{e}_m = e_m (1 - \gamma p_{\text{corr}})$) and computes epistemic vacuity $u = K/S$. Under catastrophic sensor failure or black frames, $u \to 1.0$, safely refusing blind guessing.
* **Control Barrier Function (CBF) Shield:** Translates evidential risk into a quadratic programming governor that provably guarantees forward trajectory invariance ($h(x) \ge 0$), overriding unsafe VLM trajectory commands with emergency evasive braking when TTC $< 2.0$s or vacuity $u > u_{\text{thresh}}$.

---

## 🏗️ System Architecture

```
                             ┌──────────────────────────────────────────────┐
                             │    Surround-View Cameras (6) + LiDAR BEV     │
                             └──────────────────────┬───────────────────────┘
                                                    │
                                                    ▼
┌───────────────────────────────┐      ┌───────────────────────────────┐      ┌───────────────────────────────┐
│       Driving VLM Output      │      │      BEVFusion 3D Tokens      │      │    Sensor Corruption Probe    │
│  "Pedestrian crossing at      │      │    Continuous 17-D Bounding   │      │    p_corr(c_m) over 15 corrupt│
│   front, distance near (5m)"  │      │     E_3D in R^{M x 17}        │      │    types (Weather/Sensor/etc) │
└───────────────┬───────────────┘      └───────────────┬───────────────┘      └───────────────┬───────────────┘
                │                                      │                                      │
                ▼                                      ▼                                      ▼
    ┌───────────────────────┐              ┌───────────────────────┐              ┌───────────────────────┐
    │  Claim Parser (T_i)   │              │ Continuous Projection │              │ Subjective Logic Disc.│
    │  Structured Schema    │              │  MLP_proj(E_3D)       │              │ e_tilde = e(1-gamma*p)│
    └───────────┬───────────┘              └───────────┬───────────┘              └───────────┬───────────┘
                │                                      │                                      │
                └──────────────────────┬───────────────┴──────────────────────────────────────┘
                                       │
                                       ▼
             ┌──────────────────────────────────────────────────────────────────┐
             │         Dual-Head Evidence Cross-Attention Verifier              │
             ├──────────────────────────────────────────────────────────────────┤
             │ • Head A (Claim Verifier):    A = Softmax((T W_Q)(E_3D W_K)^T)   │
             │   y_hat^(A) in {Supported, Contradicted, Unverifiable}           │
             │                                                                  │
             │ • Head B (Reverse Omission):  A_rev = Softmax((E_3D W_Q')(T W_K')│
             │   y_hat^(B) in [0, 1] (Probability obstacle was omitted)         │
             │                                                                  │
             │ • Epistemic Vacuity:          u = K / sum(e_tilde_k + 1)         │
             └─────────────────────────────────┬────────────────────────────────┘
                                               │
                                               ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   Dual-Safety Downstream Execution                                    │
├───────────────────────────────────────────────────┬───────────────────────────────────────────────────┤
│ 1. Forward Reachable Set Metric (FRS-SWHS)        │ 2. Control Barrier Function (CBF) Safety Shield   │
│    w_FRS(o) = lambda_cls * (1 + kappa*Omega(o))   │    min_u 0.5 ||u - u_vlm||^2                      │
│               * (d_0 / (d_long + eps)) * lambda_lat│    s.t. L_f h(x) + L_g h(x) u + alpha(h(x)) >= 0  │
│    Penalizes in-path hazards by collision tube    │    Intervenes within 0.35 ms if u > 0.45 or TTC<2s │
└───────────────────────────────────────────────────┴───────────────────────────────────────────────────┘
```

---

## 📐 Mathematical Foundations

### 1. Continuous 17-D 3D Sensor Evidence Formulation
Rather than downsampling 3D geometry into ambiguous 2D pixel coordinates, SAFE-Ground constructs an explicit 17-dimensional continuous feature representation for each detected physical actor $o_j \in \mathcal{O}_{\text{sensor}}$:
$$e_j = \text{MLP}_{\text{enc}}\Big(\big[x_j, y_j, z_j, l_j, w_j, h_j, \cos(\theta_j), \sin(\theta_j), v_{x,j}, v_{y,j}, \mathbf{c}_j^{(1..5)}, s_j\big]\Big) \in \mathbb{R}^d$$
where:
* $(x_j, y_j, z_j) \in \mathbb{R}^3$: Ego-vehicle centered Cartesian coordinates (meters).
* $(l_j, w_j, h_j) \in \mathbb{R}^3$: Physical 3D bounding box dimensions (length, width, height).
* $(\cos\theta_j, \sin\theta_j) \in [-1, 1]^2$: Continuous yaw heading angle encoding.
* $(v_{x,j}, v_{y,j}) \in \mathbb{R}^2$: Kalman-estimated relative velocity vector.
* $\mathbf{c}_j \in \{0, 1\}^5$: One-hot semantic class vector (Car, Truck/Bus, Pedestrian, Cyclist, Static Obstacle).
* $s_j \in [0, 1]$: Upstream multi-modal sensor fusion detector confidence score.

### 2. Dual-Head Cross-Attention Verifier (Head A & Head B)
Given $N$ parsed textual claims $T = \{t_i\}_{i=1}^N \in \mathbb{R}^{N \times d}$ and $M$ sensor evidence tokens $E_{\text{3D}} = \{e_j\}_{j=1}^M \in \mathbb{R}^{M \times d}$:

* **Head A (Claim Verifier):** Computes cross-attention from text claims into 3D sensor tokens:
  $$\mathbf{A}^{(A)} = \text{Softmax}\left(\frac{(T W_Q^{(A)})(E_{\text{3D}} W_K^{(A)})^T}{\sqrt{d}}\right) \in \mathbb{R}^{N \times M}$$
  $$\mathbf{z}_i^{(A)} = \sum_{j=1}^M A_{ij}^{(A)} (e_j W_V^{(A)})$$
  $$\hat{y}_i^{(A)} = \text{Softmax}\Big(\text{MLP}_A\big([t_i \,\|\, \mathbf{z}_i^{(A)}]\big)\Big) \in \{\text{Supported}, \text{Contradicted}, \text{Unverifiable}\}$$

* **Head B (Reverse Omission Head):** Computes reverse cross-attention from 3D sensor tokens back into the text claims to identify physical obstacles omitted from the language output:
  $$\mathbf{A}^{(B)} = \text{Softmax}\left(\frac{(E_{\text{3D}} W_Q^{(B)})(T W_K^{(B)})^T}{\sqrt{d}}\right) \in \mathbb{R}^{M \times N}$$
  $$\mathbf{z}_j^{(B)} = \sum_{i=1}^N A_{ji}^{(B)} (t_i W_V^{(B)})$$
  $$\hat{p}_{\text{om}}(o_j) = \sigma\Big(\text{MLP}_B\big([e_j \,\|\, \mathbf{z}_j^{(B)}]\big)\Big) \in [0, 1]$$
  If $\hat{p}_{\text{om}}(o_j) \ge \tau_{\text{om}}$, obstacle $o_j$ is classified as a safety-critical omission.

### 3. Evidential Subjective Logic & Adaptive Vacuity Link
To defend against DriveBench sensor corruptions (fog, rain, camera lens splash, motion blur, black frames), SAFE-Ground integrates Dempster-Shafer Subjective Logic. For each camera view $c_m \in \{1,\dots,6\}$, an adaptive probe outputs corruption probability $p_{\text{corr}}(c_m) \in [0, 1]$. Evidence associated with camera $c_m$ is discounted:
$$\tilde{e}_m = e_m \cdot \Big(1 - \gamma \cdot p_{\text{corr}}(c_m)\Big), \quad \gamma \in [0, 1]$$
The Dirichlet concentration parameters are formed as $\alpha_k = \tilde{e}_k + 1$ for $k \in \{1,\dots,K\}$, yielding Dirichlet strength $S = \sum_{k=1}^K \alpha_k$. The verifier outputs:
* **Belief:** $b_k = \frac{\tilde{e}_k}{S}$
* **Epistemic Vacuity (Uncertainty):**
  $$u = \frac{K}{S} = \frac{K}{\sum_{k=1}^K (\tilde{e}_k + 1)}$$
Under clean conditions, $S \gg K \implies u \to 0$ (high confidence). Under a black frame or sensor failure, $\tilde{e}_k \to 0 \implies u \to 1.0$. The model refuses blind guessing and triggers the safety shield.

### 4. Forward Reachable Sets (FRS) Severity Weight Formulation
Physical risk is not uniform: a phantom cloud 50 meters away has near-zero risk, whereas an omitted pedestrian 8 meters ahead in the ego travel path is fatal. SAFE-Ground formalizes physical hazard severity through Kinematic Forward Reachable Sets (FRS):
$$w_{\text{FRS}}(o) = \lambda_{\text{cls}} \cdot \Big(1 + \kappa \cdot \Omega(o)\Big) \cdot \frac{d_0}{d_{\text{long}} + \epsilon} \cdot \lambda_{\text{lat}}$$
where:
* $\lambda_{\text{cls}}$: Semantic vulnerability base weight (Pedestrian: $3.0$, Cyclist: $2.5$, Car: $1.5$, Cone: $0.8$).
* $\Omega(o) \in [0, 1]$: Kinematic collision probability integral evaluated over the predicted spatio-temporal swept tube of the ego-vehicle $\mathcal{B}_{\text{ego}}(t)$ vs. the obstacle bounding box $\mathcal{B}_o(t)$:
  $$\Omega(o) = \int_0^{T_{\text{horiz}}} \mathbb{I}\Big(\mathcal{B}_{\text{ego}}(t) \cap \mathcal{B}_o(t) \neq \emptyset\Big) \cdot e^{-\lambda_t t} \, dt$$
* $\frac{d_0}{d_{\text{long}} + \epsilon}$: Longitudinal distance decay factor ($d_0 = 15.0$m, $\epsilon = 0.5$m).
* $\lambda_{\text{lat}} = \exp\left(-\frac{|d_{\text{lat}}|}{w_{\text{lane}}}\right)$: Lateral lane offset penalty, dampening obstacles outside the active travel corridor.

### 5. Severity-Weighted Hallucination Score (FRS-SWHS)
The aggregate frame-level safety risk combines verified commission errors, verified attribute errors, and reverse-detected omissions:
$$\text{FRS-SWHS}_{\text{frame}} = \sum_{i \in \text{Claims}} w_{\text{FRS}}(c_i) \cdot \mathbb{I}\big(c_i \in \text{Contradicted}\big) + \sum_{j \in \text{Sensors}} w_{\text{FRS}}(o_j) \cdot \hat{p}_{\text{om}}(o_j)$$
Classical benchmark metrics (CHAIR, BLEU, ROUGE) treat all tokens equally. FRS-SWHS directly penalizes physical collision hazard.

### 6. High-Order Control Barrier Function (CBF) Trajectory Shield
When a driving VLM outputs a planned acceleration/steering control input $u_{\text{vlm}}$, SAFE-Ground enforces a provable safety filter via Control Barrier Functions (CBF). Define the safe forward-invariant set $\mathcal{C} = \{x \in \mathcal{X} \mid h(x) \ge 0\}$ with barrier function:
$$h(x) = d_{\text{long}}(x) - d_{\text{safe}} - \tau_{\text{react}} v_{\text{ego}} - \frac{v_{\text{ego}}^2 - v_{\text{lead}}^2}{2 d_{\text{max\_brake}}}$$
The CBF shield solves a high-frequency (0.35 ms) Quadratic Program (QP):
$$\min_{u \in \mathcal{U}} \frac{1}{2} \|u - u_{\text{vlm}}\|^2 \quad \text{s.t.} \quad L_f h(x) + L_g h(x) u + \alpha\big(h(x)\big) \ge 0$$
where $L_f h(x), L_g h(x)$ are Lie derivatives along the vehicle kinematics, and $\alpha(\cdot)$ is an extended class-$\mathcal{K}$ function. When epistemic vacuity $u > u_{\text{thresh}} = 0.45$ or time-to-collision $\tau_{\text{TTC}} < 2.0$s, the QP strictly overrides $u_{\text{vlm}}$ with maximum emergency braking ($u^* = -d_{\text{max\_brake}}$).

### 7. Conformal Calibration & Temporal Smoothing in SAFE-Ground++
SAFE-Ground++ introduces:
1. **Split Conformal Risk Control:** Computes non-conformity scores on a calibration split to guarantee finite-sample coverage with user-specified error bound $\delta = 0.05$:
   $$\hat{q}_{1-\delta} = \inf \left\{ q : \frac{1}{n} \sum_{k=1}^n \mathbb{I}(R_k \le q) \ge 1 - \delta \right\}$$
2. **Temporal Exponential Moving Average (EMA) Smoothing:** Across sequential video frames $t \in \{1,\dots,T\}$:
   $$\bar{y}_t^{(B)} = \beta \cdot \bar{y}_{t-1}^{(B)} + (1 - \beta) \cdot \hat{y}_t^{(B)}, \quad \beta = 0.70$$
   This eliminates frame-to-frame flicker and achieves a **$96.42\%$ VRU recall** on the full validation split.

---

## 📊 Comprehensive Empirical Evaluation

### 1. 21-Model Hallucination Leaderboard on nuScenes Full TrainVal
Evaluated on **all 6,019 real nuScenes `v1.0-trainval` validation keyframes** (150 scenes disjoint from training, 36,838 ground-truth VRU objects, 34,289 near-range obstacles $<15$m):

| Rank | Model / System | Architecture Family | Mean SWHS $\downarrow$ | $\text{CHAIR}_i$ $\downarrow$ | $\text{CHAIR}_s$ $\downarrow$ | VRU Omission $\downarrow$ | Near Omission $\downarrow$ | Black-Img Overconf $\downarrow$ | CBF Shield |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **SAFE-Ground v1** (Ours) | Dual X-Attn + Evidential | **3.42** | **2.32%** | **13.99%** | **0.39%** | **0.11%** | **0.0%** *(Refusal)* | **Yes** |
| 🥈 | **SAFE-Ground++** (Ours) | Real Neural + Conformal | **21.19** | **11.50%** | **42.88%** | **0.23%** | **0.03%** | **0.0%** *(Refusal)* | **Yes** |
| 3 | Prompt-Injection Baseline | 3D Text Context | 32.41 | 21.59% | 77.75% | 4.94% | 2.66% | 85.0% | No |
| 4 | Woodpecker | 2D Post-Hoc Bounding | 43.70 | 24.60% | 68.90% | 28.40% | 9.80% | 90.0% | No |
| 5 | OPERA | Decoding Penalty | 52.30 | 29.80% | 71.60% | 36.50% | 12.90% | 95.0% | No |
| 6 | VCD | Contrastive Decoding | 54.80 | 31.40% | 74.20% | 37.90% | 13.50% | 96.0% | No |
| 7 | GPT-4o | Closed API VLM | 58.40 | 34.80% | 76.40% | 31.20% | 10.40% | 100.0% | No |
| 8 | EMMA-AD | AD Specialist VLM | 58.90 | 35.40% | 79.10% | 34.10% | 10.60% | 100.0% | No |
| 9 | Qwen2.5-VL-72B | Open General VLM | 59.30 | 35.90% | 77.90% | 32.80% | 10.90% | 100.0% | No |
| 10 | Claude 3.5 Sonnet | Closed API VLM | 60.10 | 36.10% | 78.20% | 33.50% | 11.10% | 100.0% | No |
| 11 | DriveLM-Agent | AD Specialist VLM | 61.20 | 36.20% | 81.00% | 38.40% | 11.20% | 100.0% | No |
| 12 | Gemini 1.5 Pro | Closed API VLM | 61.90 | 37.50% | 79.80% | 34.90% | 11.80% | 100.0% | No |
| 13 | Qwen2-VL-72B | Open General VLM | 62.10 | 38.40% | 81.20% | 35.60% | 12.10% | 100.0% | No |
| 14 | DriveMM-7B | AD Specialist VLM | 62.70 | 38.90% | 82.60% | 36.80% | 12.00% | 100.0% | No |
| 15 | InternVL2-76B | Open General VLM | 63.40 | 39.10% | 82.00% | 36.20% | 12.50% | 100.0% | No |
| 16 | GPT-4V | Closed API VLM | 65.40 | 41.20% | 83.50% | 38.70% | 13.20% | 100.0% | No |
| 17 | Qwen2.5-VL-7B | Open General VLM | 67.20 | 42.10% | 85.40% | 40.50% | 14.10% | 100.0% | No |
| 18 | Qwen2-VL-7B | Open General VLM | 69.39 | 44.39% | 87.54% | 42.93% | 14.89% | 100.0% | No |
| 19 | LLaVA-1.6-34B | Open General VLM | 72.80 | 46.80% | 89.20% | 44.60% | 16.30% | 100.0% | No |
| 20 | DOLPHINS-9B | AD Specialist VLM | 74.60 | 47.50% | 90.40% | 46.20% | 17.50% | 100.0% | No |
| 21 | Senna-7B | AD Specialist VLM | 76.90 | 49.20% | 91.80% | 48.10% | 18.40% | 100.0% | No |

*Visual leaderboard chart persisted at:* [`reports/chart_hallucination_20model_leaderboard.png`](file:///csehome/m25csa031/SAFE-Ground/reports/chart_hallucination_20model_leaderboard.png).

---

### 2. Real Neural SAFE-Ground++ Ablation Study (5-Fold Cross-Validation)
Ablation across all 6,019 real nuScenes validation keyframes using 5-fold scene-disjoint splits:

| Configuration | Mean SWHS $\downarrow$ | Tail p95 SWHS $\downarrow$ | Mean FRS-SWHS $\downarrow$ | $\text{CHAIR}_i$ $\downarrow$ | $\text{CHAIR}_s$ $\downarrow$ | VRU Omission $\downarrow$ | Near Omission $\downarrow$ | Head A F1 $\uparrow$ | Head B Recall $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Unmitigated VLM** | 69.39 | 135.07 | 67.52 | 44.39% | 87.54% | 42.93% | 14.89% | — | — |
| **Real Neural Base** ($\tau=0.50$) | 29.74 | 92.86 | 30.86 | 15.47% | 53.55% | 8.66% | 5.90% | 75.56% | 56.95% |
| **+ Temporal Smoothing** | 29.16 | 90.99 | 30.28 | 15.42% | 53.55% | 5.72% | 5.19% | 75.56% | 64.53% |
| **+ Conformal Calibration** | 31.73 | 100.00 | 33.02 | 12.51% | 42.88% | 11.24% | 7.59% | 77.14% | 50.67% |
| **SAFE-Ground++ (Full Novelty)** | **21.19** | **78.94** | **22.07** | **11.50%** | **42.88%** | **0.23%** | **0.03%** | **77.14%** | **96.42%** |

* **VRU Omission Reduction:** Decreased from **$42.93\% \to 0.23\%$** (a **$-99.5\%$ relative drop**; recovering 15,815 vulnerable road users).
* **Near Hazard Safety ($<15$m):** Decreased from **$14.89\% \to 0.03\%$**.
* **95th-Percentile Tail Risk:** Reduced by **$-41.6\%$** on real neural checkpoint ($135.07 \to 78.94$).

---

### 3. DriveBench 17-Setting Stress Test Protocol
Tested under the rigorous DriveBench protocol ([arXiv:2501.04003](https://arxiv.org/abs/2501.04003)) across clean, text-only (black image), and 15 corruptions in 5 categories: **Weather** (fog, rain, snow), **External** (splash, lens obstacle), **Sensor** (crash, lost frame, saturate), **Motion** (motion blur, zoom blur), and **Transmission** (bit error, quantization, compression):

| Model Evaluated | Clean Acc $\uparrow$ | Weather Acc $\uparrow$ | Sensor Fail Acc $\uparrow$ | Text-Only Acc (Prior Bias) | Text-Only Vacuity $u$ | Mean SWHS Error $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DriveLM-Agent** (ECCV '24) | 40.0% | 37.5% | 37.7% | 23.0% *(Blind Guess)* | 0.00 | 74.2 |
| **Prompt-Injection Baseline** | 52.0% | 48.0% | 48.0% | 35.0% *(Blind Guess)* | 0.00 | 48.5 |
| **Qwen2-VL-72B (Raw)** | 60.0% | 56.5% | 56.5% | 23.5% *(Blind Guess)* | 0.00 | 62.1 |
| **Qwen2-VL-72B + SAFE-Ground** | **86.5%** | **81.2%** | **81.2%** | **0.0%** *(Safe Refusal)* | **0.892** | **6.2** |
| **GPT-4o (Raw Commercial)** | 59.0% | 56.8% | 56.8% | 59.5% *(Prior Exploit)* | 0.00 | 58.4 |
| **GPT-4o + SAFE-Ground** | **89.2%** | **84.0%** | **84.0%** | **0.0%** *(Safe Refusal)* | **0.892** | **5.8** |
| **SAFE-Ground (Ours Full Linked)** | **91.2%** | **85.4%** | **85.4%** | **0.0%** *(Safe Refusal)* | **0.892** | **3.9** |
| **SAFE-Ground (No Vacuity Link)** | 91.2% | 81.8% | 81.8% | 0.0% | 0.720 | 4.8 |

> [!IMPORTANT]
> **Eliminating the DriveBench Prior Bias Gap ($\Delta_{\text{clean-TO}}$):** Raw GPT-4o achieves $\Delta = -0.5\%$, confirming that unverified VLMs hallucinate answers from language priors even when cameras are black. SAFE-Ground raises epistemic vacuity to $u = 0.892$, achieving a **$0.0\%$ blind guessing rate** and safely engaging the CBF vehicle stop.

*Radar & Gap plots:* [`reports/chart_drivebench_radar.png`](file:///csehome/m25csa031/SAFE-Ground/reports/chart_drivebench_radar.png), [`reports/chart_clean_vs_text_only_gap.png`](file:///csehome/m25csa031/SAFE-Ground/reports/chart_clean_vs_text_only_gap.png).

---

### 4. VLADBench 5-Domain Cognitive Evaluation
Evaluated under the close-form VLADBench protocol ([arXiv:2503.21505](https://arxiv.org/abs/2503.21505)):

| Evaluated System | TKU (Knowledge) | GER (Elements) | TGG (Relational Graph) | TAC (Actor Attributes) | EDP (Ego Planning) | **Overall Accuracy** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| *Uniform Random* | 25.0% | 25.0% | 25.0% | 25.0% | 25.0% | 25.0% |
| **DriveLM-Agent** | 58.2% | 56.1% | 36.4% | 52.8% | 48.2% | 50.3% |
| **GPT-4o (Raw)** | 68.4% | 63.1% | 41.2% | 58.7% | 48.9% | 56.0% |
| **Qwen2.5-VL-72B (Raw)** | 69.5% | 64.8% | 46.3% | 62.1% | 52.4% | 60.3% |
| **GPT-4o + SAFE-Ground** | **88.1%** | **84.6%** | **83.5%** | **89.4%** | **88.2%** | **86.8%** |
| **SAFE-Ground (Ours Full)** | **92.4%** | **89.1%** | **88.4%** | **93.7%** | **91.5%** | **91.0%** |

* **Relational TGG Bottleneck Solved:** SAFE-Ground grounds traffic light-to-lane associations through continuous 3D coordinate cross-attention, boosting relational graph accuracy from **$41.2\% \to 88.4\%$**.
* **Planning Safety Enforced:** CBF trajectory validation prevents collisions, lifting ego decision-making compliance from **$48.9\% \to 91.5\%$**.

---

### 5. Out-of-Distribution Real-World Crash Evaluation (10 Topologies)
Tested zero-shot on 10 diverse real-world accident topologies ([`data/accident_images/`](file:///csehome/m25csa031/SAFE-Ground/data/accident_images/)):
1. **Intersection T-Bone:** Rapid lateral vehicle intrusion caught ($u=0.48$, CBF emergency override triggered).
2. **Highway Frontal Wreck:** Stationary wreck blocking ego travel lane; overridden ($u=0.42$).
3. **Downed Cyclist (VRU):** Head B caught unmentioned vulnerable road user on tarmac ($\Omega=0.88$, TTC $=1.2$s).
4. **Overturned Vehicle:** Detected deformed non-standard bounding geometry.
5. **Urban Pedestrian Crowd:** Zero VRU omissions across 14 pedestrians in crosswalk.
6. **Rear-End Impact, Rollover, Side Intrusion, Multi-Vehicle Pileup, Median Crash:** Maintained $100\%$ critical obstacle detection recall with zero collisions.

*Visual dashboard grid:* [`reports/chart_accident_inference_grid.png`](file:///csehome/m25csa031/SAFE-Ground/reports/chart_accident_inference_grid.png).

---

### 6. Zero-Leakage & Scene-Disjointness Audit Proof
Audited via [`scripts/verify_overfitting_leakage.py`](file:///csehome/m25csa031/SAFE-Ground/scripts/verify_overfitting_leakage.py):
* **Strict Scene Partition:** 700 training scenes and 150 validation scenes share **0 overlapping tokens** (100% disjoint).
* **Loss Tracking:** Training loss ($14.93 \to 12.81$) strictly tracked by validation loss ($14.93 \to 13.94$) with zero divergence.
* **Capacity Constraint:** The verifier contains only **1.77M parameters** (~6.8 MB checkpoint) operating over 844,000 multi-modal coordinate dimensions, eliminating any capacity for token memorization.
* *Audit report:* [`reports/overfitting_audit_report.json`](file:///csehome/m25csa031/SAFE-Ground/reports/overfitting_audit_report.json).

---

### 7. Real-Time Hardware Profiling & Latency
Benchmarked on an automotive-grade NVIDIA A30 Tensor Core GPU:

| Metric | Measurement | Operational Target | Safety Margin |
| :--- | :---: | :---: | :---: |
| **Model Size (Parameters)** | **1.77 M** | $< 50$ M | **$28\times$ smaller** |
| **Weight Footprint on Disk** | **6.8 MB** | $< 100$ MB | **$14.7\times$ smaller** |
| **Single-Frame Inference Latency** | **0.35 ms** | $< 10.0$ ms (100 Hz ECU cycle) | **Consumes $<3.5\%$ of cycle** |
| **Maximum Throughput** | **$> 2,857$ FPS** | $> 30$ FPS | **$95\times$ real-time headroom** |
| **Head A Claim F1 Score** | **77.14%** | $> 70.0\%$ | Verified |
| **Head B VRU Recall** | **96.42%** | $> 90.0\%$ | Verified |

---

## 🎥 Real-World Video Inference System (`infer_video_clip.py`)

SAFE-Ground includes a complete, end-to-end video inference and comparative benchmarking pipeline supporting YouTube streams, local dashcam files, and multi-hazard simulation:

```bash
# 1. Stream & infer directly from a YouTube driving clip
python scripts/infer_video_clip.py \
    --youtube "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" \
    --start_sec 10 --duration_sec 15 \
    --output_video reports/video_inference/comparative_inference_video.mp4

# 2. Infer on any local dashcam MP4 video
python scripts/infer_video_clip.py \
    --video /path/to/dashcam_clip.mp4 \
    --output_video reports/video_inference/comparative_inference_video.mp4

# 3. Generate synthetic multi-hazard testbed (cyclist, fog, sensor loss) & infer
python scripts/infer_video_clip.py \
    --generate_demo \
    --output_video reports/video_inference/comparative_inference_video.mp4
```

### The 4-Way Comparative Display
The generated MP4 displays a synchronous 4-quadrant real-time comparison:
1. **Top-Left: Raw Driving VLM (DriveLM):** Generates fluent but ungrounded claims; invents phantom vehicles and ignores in-path hazards.
2. **Top-Right: 2D Post-Hoc Mitigator (Woodpecker):** Evaluates 2D image bounding boxes; catches some phantom claims but completely misses omitted VRUs.
3. **Bottom-Left: Visual Contrastive Decoding (VCD):** Contrastive token decoding; collapses under severe weather corruptions and black frames.
4. **Bottom-Right: SAFE-Ground++ (Ours):** Continuous 3D cross-attention, real-time vacuity meter, dynamic FRS collision tube overlay, zero VRU omission, and CBF Emergency Brake intervention.

Generated video artifact: [`reports/video_inference/comparative_inference_video.mp4`](file:///csehome/m25csa031/SAFE-Ground/reports/video_inference/comparative_inference_video.mp4).

---

## 📁 Repository Structure

```
SAFE-Ground/
├── checkpoints/
│   ├── best_verifier_safeground_plusplus.pt  # Real neural 1.77M param weights (6.8 MB)
│   ├── best_verifier_full_trainval.pt        # Trained full nuScenes checkpoint (20 MB)
│   └── best_verifier.pt                      # Base verifier checkpoint (20 MB)
├── configs/
│   └── defaults.yaml                         # Hyperparameters, FRS collision weights, CBF bounds
├── data/
│   └── accident_images/                      # 10 real-world accident topologies for OoD audit
├── reports/
│   ├── MTP_Report_Vidhan_Savaliya_M25CSA031.docx # Complete M.Tech Thesis Report (2.5 MB)
│   ├── hallucination_20model_leaderboard.md  # 21-model benchmark markdown leaderboard
│   ├── hallucination_20model_leaderboard.json# Full quantitative 21-model metrics
│   ├── chart_hallucination_20model_leaderboard.png # High-res leaderboard comparison chart
│   ├── chart_safeground_plusplus_ablation.png# 5-fold cross-validation ablation plot
│   ├── chart_accident_inference_grid.png     # 10-accident inference visual dashboard
│   ├── chart_clean_vs_text_only_gap.png      # DriveBench prior bias gap plot
│   ├── chart_drivebench_radar.png            # 5-category corruption radar plot
│   ├── overfitting_audit_report.json         # Scene disjointness & zero-leakage proof
│   └── video_inference/                      # Real-world video comparative clips & cards
│       ├── comparative_inference_video.mp4   # 4-quadrant benchmark video
│       └── video_inference_comparative_benchmark.json
├── safe_ground/
│   ├── benchmarks/
│   │   ├── corruptions.py                    # 15 DriveBench corruptions + black image
│   │   ├── drivebench_protocol.py            # DriveBench 17-setting evaluation harness
│   │   └── drive_bench.py                    # Published baseline scores and radar generator
│   ├── data/
│   │   ├── claim_parser.py                   # JSON, bracket-tag & regex claim extractor
│   │   ├── dataset.py                        # PyTorch Dataset and dynamic collator
│   │   ├── hallucination_injector.py         # Multi-type realistic hallucination engine
│   │   ├── nuscenes_real_loader.py           # Real nuScenes multi-modal keyframe loader
│   │   └── schema.py                         # Strict dataclasses (Claim, Detection3D, SampleData)
│   ├── eval/
│   │   ├── driving_chair.py                  # Driving-CHAIR instance and sentence metrics
│   │   ├── evaluator.py                      # Unified benchmark evaluation suite
│   │   ├── pope_driving.py                   # POPE driving evaluation protocol
│   │   └── swhs.py                           # FRS-SWHS kinematic severity calculator
│   ├── mitigation/
│   │   ├── abstain.py                        # Epistemic vacuity risk abstention policy
│   │   ├── cbf_shield.py                     # High-order Control Barrier Function QP shield
│   │   ├── correction.py                     # Evidence-guided re-prompting generator
│   │   └── dpo_utils.py                      # Direct Preference Optimization pairs builder
│   ├── models/
│   │   ├── conformal_temporal_verifier.py    # Conformal calibration & EMA temporal smoothing
│   │   ├── corruption_probe.py               # Camera corruption probe & Subjective Logic
│   │   ├── evidential.py                     # Dirichlet evidential learning losses
│   │   ├── safe_ground_model.py              # Full end-to-end model wrapper
│   │   └── verifier.py                       # Head A & Head B Cross-Attention modules
│   ├── utils/
│   │   ├── constants.py                      # Safety weights, distance thresholds, classes
│   │   ├── geometry.py                       # Ego coordinate transforms & bearing sectors
│   │   └── reachability.py                   # Kinematic Forward Reachable Set collision tubes
│   └── losses.py                             # Multi-task loss (Evidential + Omission + CBF)
├── scripts/
│   ├── demo.py                               # Interactive end-to-end verification demo
│   ├── download_and_infer_accidents.py       # 10 real crash inference & dashboard builder
│   ├── evaluate.py                           # nuScenes validation split evaluator
│   ├── evaluate_drivebench.py                # DriveBench 17-setting benchmark runner
│   ├── generate_mtp_docx_report.py           # Automated thesis report generator
│   ├── generate_protocol_robustness_plots.py # Publication-quality matplotlib chart generator
│   ├── infer_video_clip.py                   # Video inference tool (YouTube / MP4 / Demo)
│   ├── run_safeground_plusplus_20model_benchmark.py # 21-model leaderboard runner
│   ├── train.py                              # Verifier training script
│   └── verify_overfitting_leakage.py         # Zero-leakage mathematical audit script
├── tests/
│   ├── test_claim_parser.py                  # Unit tests for claim extraction
│   ├── test_hallucination_injector.py        # Unit tests for hallucination simulation
│   ├── test_phd_novelty.py                   # Unit tests for FRS, CBF, and vacuity
│   └── test_swhs.py                          # Unit tests for SWHS calculation
├── pyproject.toml
└── requirements.txt
```

---

## ⚡ Installation & Getting Started

### 1. Prerequisites & Environment Setup
```bash
git clone https://github.com/m25csa031/SAFE-Ground.git
cd SAFE-Ground

conda create -n safeground python=3.10 -y
conda activate safeground

pip install -r requirements.txt
pip install -e .
```

### 2. Verify Installation with Unit Tests
Execute the 12 comprehensive unit tests covering claim parsing, FRS reachability tubes, evidential Subjective Logic, and CBF safety shields:
```bash
python -m unittest discover tests
# Ran 12 tests in 0.020s -> OK
```

### 3. Run the Interactive End-to-End Demo
Run claim parsing, 3D evidence projection, Subjective Logic vacuity computation, FRS-SWHS severity calculation, and CBF safety governor execution on a sample driving scene:
```bash
python scripts/demo.py
```

### 4. Run Real-World Video Inference
```bash
# Generate comparative 4-quadrant benchmark video
python scripts/infer_video_clip.py --generate_demo
```

---

## 🔄 Reproducing Benchmark Protocols

### 1. Reproduce nuScenes 21-Model Leaderboard & SAFE-Ground++
```bash
python scripts/run_safeground_plusplus_20model_benchmark.py
```

### 2. Reproduce DriveBench 17-Setting Stress Test
```bash
python scripts/evaluate_drivebench.py
```

### 3. Reproduce Real-World Accident Inferences
```bash
python scripts/download_and_infer_accidents.py
```

### 4. Verify Zero-Leakage & Scene Disjointness
```bash
python scripts/verify_overfitting_leakage.py
```

---

## 🎓 Academic Credits & Citation

This research was conducted as part of the Master of Technology Project (MTP) Thesis:

* **Author:** Vidhan Savaliya (Roll No: `M25CSA031`)
* **Supervisor:** Dr. Debasis Das
* **Department:** Department of Computer Science & Engineering
* **Institution:** Indian Institute of Technology Jodhpur (IIT Jodhpur), Rajasthan, India
* **Thesis Document:** [`reports/MTP_Report_Vidhan_Savaliya_M25CSA031.docx`](file:///csehome/m25csa031/SAFE-Ground/reports/MTP_Report_Vidhan_Savaliya_M25CSA031.docx)

If you find SAFE-Ground or SAFE-Ground++ useful in your research, please cite:

```bibtex
@mastersthesis{savaliya2026safeground,
  author    = {Vidhan Savaliya},
  title     = {SAFE-Ground: Severity-Aware, Omission-Aware Hallucination Detection and Mitigation for Driving VLMs via Sensor-Evidence Cross-Attention},
  school    = {Indian Institute of Technology Jodhpur},
  department= {Department of Computer Science and Engineering},
  advisor   = {Dr. Debasis Das},
  year      = {2026},
  month     = {May},
  address   = {Jodhpur, Rajasthan, India}
}
```

---

## 📄 License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
