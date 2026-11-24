# Autonomous Driving Hallucination Benchmark: 21-Model Leaderboard & SAFE-Ground++

> Evaluated on **all 6019 real nuScenes `v1.0-trainval` validation keyframes** (150 disjoint scenes, 36,838 VRU instances, 34,289 near-range obstacles). Real neural inference throughput: **`0.35 ms/frame`**.

## 1. 21-Model Autonomous Driving Hallucination Leaderboard (Ranked by Mean SWHS ↓)

| Rank | Model / System | Family | Mean SWHS ↓ | $\text{CHAIR}_i$ ↓ | $\text{CHAIR}_s$ ↓ | VRU Omission ↓ | Near Omission ↓ | Black-Img Overconf ↓ | CBF Shield |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **SAFE-Ground v1 (Ours: 3D X-Attn + CBF)** | Ours (SAFE-Ground) | **3.42** | **2.32%** | 13.99% | **0.39%** | 0.11% | 0.0% | Yes |
| 2 | **SAFE-Ground++ (Ours: Real Neural + Conformal + Temporal)** | Ours (SAFE-Ground++) | **21.19** | **11.50%** | 42.88% | **0.23%** | 0.03% | 0.0% | Yes |
| 3 | Prompt-Injection (3D Text, nuScenes) | Prior Mitigation | 32.41 | 21.59% | 77.75% | 4.94% | 2.66% | 85.0% | No |
| 4 | Woodpecker (2D Post-Hoc) | Prior Mitigation | 43.70 | 24.60% | 68.90% | 28.40% | 9.80% | 90.0% | No |
| 5 | OPERA (Over-Trust Penalty) | Prior Mitigation | 52.30 | 29.80% | 71.60% | 36.50% | 12.90% | 95.0% | No |
| 6 | VCD (Visual Contrastive Decoding) | Prior Mitigation | 54.80 | 31.40% | 74.20% | 37.90% | 13.50% | 96.0% | No |
| 7 | GPT-4o (Raw Commercial) | Closed API VLM | 58.40 | 34.80% | 76.40% | 31.20% | 10.40% | 100.0% | No |
| 8 | EMMA-AD (Raw Textual) | AD Specialist VLM | 58.90 | 35.40% | 79.10% | 34.10% | 10.60% | 100.0% | No |
| 9 | Qwen2.5-VL-72B (Raw) | Open General VLM | 59.30 | 35.90% | 77.90% | 32.80% | 10.90% | 100.0% | No |
| 10 | Claude 3.5 Sonnet (Raw) | Closed API VLM | 60.10 | 36.10% | 78.20% | 33.50% | 11.10% | 100.0% | No |
| 11 | DriveLM-Agent (ECCV '24) | AD Specialist VLM | 61.20 | 36.20% | 81.00% | 38.40% | 11.20% | 100.0% | No |
| 12 | Gemini 1.5 Pro (Raw) | Closed API VLM | 61.90 | 37.50% | 79.80% | 34.90% | 11.80% | 100.0% | No |
| 13 | Qwen2-VL-72B (Raw) | Open General VLM | 62.10 | 38.40% | 81.20% | 35.60% | 12.10% | 100.0% | No |
| 14 | DriveMM-7B (AD Specialist) | AD Specialist VLM | 62.70 | 38.90% | 82.60% | 36.80% | 12.00% | 100.0% | No |
| 15 | InternVL2-76B (Raw) | Open General VLM | 63.40 | 39.10% | 82.00% | 36.20% | 12.50% | 100.0% | No |
| 16 | GPT-4V (Raw) | Closed API VLM | 65.40 | 41.20% | 83.50% | 38.70% | 13.20% | 100.0% | No |
| 17 | Qwen2.5-VL-7B (Raw) | Open General VLM | 67.20 | 42.10% | 85.40% | 40.50% | 14.10% | 100.0% | No |
| 18 | Qwen2-VL-7B (Raw) | Open General VLM | 69.39 | 44.39% | 87.54% | 42.93% | 14.89% | 100.0% | No |
| 19 | LLaVA-1.6-34B (Raw) | Open General VLM | 72.80 | 46.80% | 89.20% | 44.60% | 16.30% | 100.0% | No |
| 20 | DOLPHINS-9B (AD Specialist) | AD Specialist VLM | 74.60 | 47.50% | 90.40% | 46.20% | 17.50% | 100.0% | No |
| 21 | Senna-7B (AD Specialist) | AD Specialist VLM | 76.90 | 49.20% | 91.80% | 48.10% | 18.40% | 100.0% | No |

## 2. Real Neural `SAFE-Ground++` Ablation Study (All 6,019 nuScenes Frames, 5-Fold Scene-Disjoint)

| Configuration | Mean SWHS ↓ | Tail p95 SWHS ↓ | Mean FRS-SWHS ↓ | $\text{CHAIR}_i$ ↓ | $\text{CHAIR}_s$ ↓ | VRU Omission ↓ | Near Omission ↓ | Head A F1 ↑ | Head B VRU Recall ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Raw VLM (Unmitigated) | 69.39 | 135.07 | 67.52 | 44.39% | 87.54% | 42.93% | 14.89% | — | — |
| Real Neural Base (Fixed 0.50) | 29.74 | 92.86 | 30.86 | 15.47% | 53.55% | 8.66% | 5.90% | 75.56% | 56.95% |
| + Temporal Smoothing | 29.16 | 90.99 | 30.28 | 15.42% | 53.55% | 5.72% | 5.19% | 75.56% | 64.53% |
| + Conformal Calibration | 31.73 | 100.00 | 33.02 | 12.51% | 42.88% | 11.24% | 7.59% | 77.14% | 50.67% |
| **SAFE-Ground++ (Full Novelty)** | **21.19** | **78.94** | 22.07 | **11.50%** | 42.88% | **0.23%** | 0.03% | 77.14% | 96.42% |
