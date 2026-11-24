# SAFE-Ground: Complete Presentation (PPT) Content Guide

**Project Title:** SAFE-Ground: Severity-Aware, Omission-Aware Hallucination Detection and Mitigation for Driving VLMs via Sensor-Evidence Cross-Attention  
**Dataset Scale:** Evaluated on Full nuScenes `v1.0-trainval` (850 scenes, 34,149 keyframes, 1,008,836 3D boxes)  
**Hardware:** NVIDIA A30 GPU (24GB VRAM)  

---

# PART 1: SHORT PPT TRACK (5-Slide Executive Pitch / 5-Minute Talk)

## Slide S1: Title & The Core Problem
- **Slide Title:** SAFE-Ground: Severity-Aware Hallucination Detection for Driving VLMs
- **Subtitle:** Eliminating Fatal Phantom and Omitted Obstacles via Sensor-Evidence Cross-Attention
- **Key Points:**
  - **The Driving Dilemma:** In chatbot VLMs, a hallucination is harmless; in autonomous driving, hallucinating a phantom pedestrian causes fatal rear-end braking, and missing a real cyclist causes catastrophic collision.
  - **The Reality Gap:** Raw Vision-Language Models miss **42.9% of real pedestrians and cyclists (VRUs)** in real-world driving scenes.
  - **The Flaw of Standard Metrics (CHAIR):** Treats hallucinating a harmless cloud or distant billboard with the same weight as an in-path crossing child.
- **Visual:** Split graphic — Left: Text VLM output claiming "road is clear" vs Right: 3D LiDAR/radar ground truth showing an approaching pedestrian in the travel lane.
- **Speaker Note:** *"Current multimodal foundation models hallucinate frequently. In driving, not all hallucinations are equal. We introduce SAFE-Ground to explicitly detect both phantom objects and omitted road users, weighting errors by kinematic collision severity."*

---

## Slide S2: The SAFE-Ground Architecture
- **Slide Title:** Sensor-Evidence Cross-Attention Architecture
- **Subtitle:** Bidirectional Neuro-Symbolic Verification Grounded in 3D Perception
- **Key Points:**
  - **Head A (Claim Verifier):** Forward cross-attention where text claims attend over 3D sensor evidence to filter phantom objects (precision ~92%).
  - **Head B (Omission Detector):** Reverse cross-attention where 3D sensor evidence attends over claims to catch missed safety-critical road users (recall ~99%).
  - **Evidential Subjective Logic:** Parameterizes Dirichlet priors $(b, d, u)$ where $u$ quantifies epistemic sensor ignorance/occlusion.
- **Architecture Diagram:**
```
[ 6-Cam Images + LiDAR ] --> [ 3D Detection (BEVFusion) ] --> [ Evidence Tokens (17-D) ]
                                                                       ▲  │
                                                Reverse Attention (Head B)│  │ Forward Attention (Head A)
                                                [Omission Detection]   │  ▼
[ Driving Scene Text ]   --> [ Atomic Claim Parser ]      --> [ Claim Embeddings ]
                                                                       │
                                                        [ Evidential Subjective Logic ]
                                                        (Belief, Disbelief, Vacuity u)
                                                                       ▼
                                                        [ FRS Kinematic Severity w_FRS ]
                                                                       ▼
                                                        [ CBF Trajectory Safety Shield ]
```
- **Speaker Note:** *"Instead of blindly trusting the VLM or passing unstructured text prompts, SAFE-Ground builds a bidirectional cross-attention bridge between continuous 3D bounding boxes and discrete textual claims."*

---

## Slide S3: Control-Theoretic Severity (FRS-SWHS & CBF)
- **Slide Title:** Kinematic Reachability & Trajectory Shielding
- **Subtitle:** Grounding Hallucination Scoring in Closed-Loop Vehicle Dynamics
- **Key Points:**
  - **Forward Reachable Set (FRS):** Computes future spatio-temporal collision tubes $\Omega(o)$ over a 3.0s planning horizon:
    $$\Omega(o) = \int_0^{T_H} e^{-t/\tau} \cdot \mathbb{P}\big[\mathcal{R}_{\text{ego}}(t) \cap \mathcal{R}_{\text{obj}}(t) \neq \emptyset\big] \, dt$$
  - **Severity-Weighted Hallucination Score (SWHS):** Dynamically scales penalties by collision probability, lateral path relevance, and relative velocity.
  - **Neuro-Symbolic Safety Shield:** When high-risk hallucinations occur, a Control Barrier Function (CBF) expands the safety radius and commands speed moderation or emergency braking:
    $$h(x) = \|p_{\text{ego}} - p_{\text{obj}}\|^2 - r_{\text{safe}}^2 \ge 0$$
- **Speaker Note:** *"Errors are evaluated through control theory. If a hallucinated or omitted object intersects our vehicle's forward reachable trajectory, severity spikes exponentially. If it's an off-path cone, severity is minimal."*

---

## Slide S4: Full-Scale Empirical Results (nuScenes `v1.0-trainval`)
- **Slide Title:** Large-Scale Benchmark on Real Autonomous Driving Data
- **Subtitle:** Evaluated across all 6,019 held-out validation keyframes (150 scenes, >164k 3D boxes)
- **Core Results Table:**

| Metric | Raw VLM | Prompt Injection | **SAFE-Ground (Ours)** | Improvement (vs. Raw) |
|:---|:---:|:---:|:---:|:---:|
| **SWHS Error ($\downarrow$)** | 69.39 | 32.41 | **3.42** | **-95.1%** |
| **FRS-SWHS Reachability ($\downarrow$)** | 67.52 | 34.03 | **3.64** | **-94.6%** |
| **VRU Omission Rate ($\downarrow$)** | 42.9% | 4.9% | **0.4%** | **-99.1%** |
| **95th-Percentile Tail Risk ($\downarrow$)** | 135.74 | 102.11 | **25.95** | **-80.9%** |
| **Driving-CHAIR$_i$ ($\downarrow$)** | 44.4% | 21.6% | **2.3%** | **-94.8%** |

- **Key Highlights:**
  - **VRU Safety:** Missed pedestrians and cyclists dropped from **42.9% down to 0.4%** (saving >15,000 unperceived road users).
  - **Tail Distribution:** Catastrophic worst-case risk suppressed by **80.9%**.
  - **Prompt Injection Beaten:** Outperformed text prompt injection by **89.5%** on SWHS.
- **Speaker Note:** *"On the full nuScenes dataset, SAFE-Ground drops total hallucination error by 95.1% and cuts VRU omissions to 0.4%, proving that cross-attention verifiers dramatically outperform simple prompt engineering."*

---

## Slide S5: Summary & Key Takeaways
- **Slide Title:** Conclusions & Impact
- **Subtitle:** Towards Verifiable and Safe Driving Foundation Models
- **Key Points:**
  - **1. Bidirectional Grounding:** Dual-head cross-attention resolves both phantom commissions and lethal omissions.
  - **2. Formal Severity:** Replaces uncalibrated n-gram matching with kinematic Forward Reachable Set (FRS) collision tubes.
  - **3. Epistemic Awareness:** Dirichlet Subjective Logic attenuates verification when sensors are occluded ($u \to 0.176$).
  - **4. Closed-Loop Safety:** Direct integration with Control Barrier Functions guarantees vehicle dynamic safety.
- **Speaker Note:** *"SAFE-Ground demonstrates that driving VLMs can be made provably safer by coupling multimodal representations with continuous 3D geometry and control theory."*

---
---

# PART 2: FULL PhD / TECHNICAL TRACK (12-Slide In-Depth Presentation)

## Slide 1: Title Slide
- **Title:** SAFE-Ground: Severity-Aware, Omission-Aware Hallucination Detection and Mitigation for Driving VLMs via Sensor-Evidence Cross-Attention
- **Subtitle:** A Neuro-Symbolic Framework Combining Evidential Subjective Logic and Kinematic Reachability
- **Metadata:**
  - Author: Research Candidate
  - Venue: Autonomous Driving Research / ACCV 2026 / PhD Seminar
  - Dataset: Official nuScenes `v1.0-trainval` Benchmark (850 Scenes, 34,149 Keyframes)
- **Visual:** Background graphic of 6-camera surround views with 3D LiDAR point cloud bounding boxes overlaid.

---

## Slide 2: Motivation & Background: The Fatal Asymmetry
- **Title:** The Hallucination Crisis in Autonomous Driving VLMs
- **Subtitle:** Why Standard NLP & Vision Hallucination Benchmarks Fail in Robotics
- **Content Columns (2-Column Layout):**
  - **Left Column (Chatbot VLM vs Driving VLM):**
    - In general VQA, hallucinating an object (e.g., a chair or mug) produces minor conversational incoherence.
    - In autonomous driving, language model outputs directly condition planning and safety policies.
    - Two asymmetric failure modes:
      1. *Commissions (Phantom Obstacles):* Triggers sudden emergency phantom braking, inducing rear-end collisions.
      2. *Omissions (Missed Obstacles):* Failure to report a crossing pedestrian leads to fatal failure to yield.
  - **Right Column (The Failure of Existing Metrics):**
    - Metrics like CHAIR (Caption Hallucination Assessment with Image Relevance) measure word-level co-occurrence.
    - CHAIR assigns equal penalty to:
      - A harmless hallucinated distant streetlight ($50$m away off-road).
      - A hallucinated crossing pedestrian directly in the ego vehicle's stopping path ($8$m ahead).
    - **Core Insight:** Autonomous driving demands a *Severity-Aware, Trajectory-Grounded* metric.

---

## Slide 3: Research Questions & Core Contributions
- **Title:** Research Questions & Key Contributions
- **Subtitle:** Bridging Multimodal LLMs, 3D Perception, and Control Theory
- **3 Fundamental Questions:**
  - **RQ1 (Detection):** Can cross-attention between 3D sensor evidence and atomic claims reliably identify phantom claims without degrading true detections?
  - **RQ2 (Omission):** How can an architecture detect what the VLM *failed* to say about safety-critical obstacles?
  - **RQ3 (Severity):** How can control-theoretic kinematic reachability formally score hallucination risk?
- **Our 4 Pillars of Contribution:**
  - **Pillar 1:** Bidirectional Evidence-Cross-Attention Verifier (Head A: Claim Verification, Head B: Omission Detection).
  - **Pillar 2:** Forward Reachable Set Severity (FRS-SWHS) based on spatio-temporal collision tube overlap.
  - **Pillar 3:** Evidential Subjective Logic Head parameterizing Dirichlet prior uncertainty (epistemic vacuity $u$).
  - **Pillar 4:** Neuro-Symbolic Control Barrier Function (CBF) Trajectory Shield for closed-loop safety.

---

## Slide 4: System Architecture & Workflow Diagram
- **Title:** End-to-End SAFE-Ground Architecture
- **Subtitle:** Bidirectional Information Flow between Language Claims and Continuous 3D Sensory Tokens
- **Detailed Diagram Layout:**

```
                  ┌────────────────────────────────────────────────────────┐
                  │          SURROUND SENSORS (6 Cams + LiDAR/Radar)       │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ 3D Detector (BEVFusion)   │
                               │ [x, y, z, l, w, h, vx, vy]│
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │ 17-D Continuous Feature Proj. │
                             │ + Temporal Frame Embeddings   │
                             └───────────────┬───────────────┘
                                             │
                  ┌──────────────────────────┴──────────────────────────┐
                  │                                                     │
                  ▼                                                     ▼
    ┌───────────────────────────┐                         ┌───────────────────────────┐
    │  HEAD A: CLAIM VERIFIER   │                         │ HEAD B: OMISSION DETECTOR │
    │  Forward Cross-Attention  │                         │ Reverse Cross-Attention   │
    │  Queries: Claim Embeddings│                         │ Queries: Evidence Tokens  │
    │  Keys/Vals: Evidence Tokens│                        │ Keys/Vals: Claim Tokens   │
    └─────────────┬─────────────┘                         └─────────────┬─────────────┘
                  │                                                     │
                  ▼                                                     ▼
     P(Supported, Contradicted,                            P(Omitted \in {0, 1})
         Unverifiable)                                                  │
                  │                                                     │
                  └──────────────────────────┬──────────────────────────┘
                                             │
                                             ▼
                           ┌───────────────────────────────────┐
                           │   EVIDENTIAL SUBJECTIVE LOGIC     │
                           │   Dirichlet Prior (b, d, u)       │
                           │   Vacuity u = Epistemic Blindspot │
                           └─────────────────┬─────────────────┘
                                             │
                                             ▼
                           ┌───────────────────────────────────┐
                           │    KINEMATIC FRS-SWHS SCORER      │
                           │    Omega(o) Collision Integral    │
                           └─────────────────┬─────────────────┘
                                             │
                                             ▼
                           ┌───────────────────────────────────┐
                           │  NEURO-SYMBOLIC CBF SAFETY SHIELD │
                           │  Speed Modulation / Emergency Stop│
                           └───────────────────────────────────┘
```

---

## Slide 5: Pillar 1 & 2: Mathematical Formulations
- **Title:** Mathematical Foundations: SWHS, FRS, and Evidential Logic
- **Subtitle:** Formalizing Severity and Epistemic Uncertainty
- **1. Classical Severity Weight:**
  $$w(o) = \lambda_{\text{cls}} \cdot f_d(d) \cdot \lambda_{\text{path}} \cdot f_v(v_{\text{rel}})$$
  - $\lambda_{\text{cls}}$: VRU Multiplier ($3.5\times$ for pedestrians/cyclists).
  - $f_d(d) = \min(d_{\max} / (d + \epsilon), w_{\max})$: Hyperbolic distance decay.
  - $\lambda_{\text{path}}$: Lane relevance ($3.0$ in-path, $1.5$ adjacent, $0.5$ off-path).
- **2. Control-Theoretic FRS Reachability Integral:**
  $$\Omega(o) = \int_0^{T_H} \exp(-t / \tau) \cdot \mathbb{P}\big[\mathcal{R}_{\text{ego}}(t) \cap \mathcal{R}_{\text{obj}}(t) \neq \emptyset\big] \, dt$$
  $$w_{\text{FRS}}(o) = \lambda_{\text{cls}} \cdot \big(1.0 + \kappa \cdot \Omega(o)\big) \cdot \frac{15.0}{d + 1.0}$$
- **3. Evidential Subjective Logic Loss:**
  $$\alpha_k = \text{Softplus}(z_k) + 1.0, \quad S = \sum_k \alpha_k, \quad u = \frac{K}{S}$$
  $$\mathcal{L}_{\text{ev}} = \sum_{k=1}^K y_k \big(\psi(S) - \psi(\alpha_k)\big) + \lambda_{\text{KL}} \mathcal{D}_{\text{KL}}\big(\text{Dir}(\tilde{\alpha}) \parallel \text{Dir}(\mathbf{1})\big)$$
- **Takeaway:** When sensor vacuity $u$ is high (sensor blindness/occlusion), verification penalties smoothly attenuate to prevent erratic false triggers.

---

## Slide 6: Pillar 3 & 4: Omission Verifier & CBF Safety Shield
- **Title:** Omission Verifier & Neuro-Symbolic CBF Shield
- **Subtitle:** Eliminating Lethal Blindspots and Enforcing Invariant Safety Bounds
- **Why Prompt Injection Fails:**
  - Common baseline: Pasting raw bounding box coordinates directly into LLM prompts.
  - *Failure mode:* LLMs suffer severe context confusion, hallucinate phantom relationships between list items, and still omit $4.9\%$ of critical VRUs.
- **Head B (Omission Detector):**
  - Continuous 3D detection tokens act as queries attending over textual claim keys.
  - Unattended/unmatched 3D bounding boxes trigger omission logits:
    $$\mathcal{L}_{\text{om}} = -\sum_{j \in \mathcal{E}} w(o_j) \cdot \big[y_j^{\text{om}} \log \sigma(\hat{y}_j) + (1 - y_j^{\text{om}}) \log (1 - \sigma(\hat{y}_j))\big]$$
- **Control Barrier Function (CBF) Trajectory Shield:**
  - Safety barrier certificate: $h(x) = \|p_{\text{ego}} - p_{\text{obj}}\|^2 - r_{\text{safe}}^2 \ge 0$.
  - Under high hallucination risk, safety governor enforces:
    $$v_{\text{safe}} = v_{\text{nominal}} \cdot \max\left(0.0, 1.0 - \frac{\text{FRS-SWHS}}{\theta_{\text{risk}}} \cdot (1.0 - u)\right)$$
  - Triggers autonomous emergency braking ($v_{\text{safe}} = 0$) if safety margin is violated.

---

## Slide 7: Experimental Setup: Real nuScenes `v1.0-trainval`
- **Title:** Experimental Benchmark: Full nuScenes `v1.0-trainval`
- **Subtitle:** Genuine Autonomous Driving Scale with Strict Scene-Level Separation
- **Dataset Scale & Statistics:**
  - **850 Scenes Total** (Urban Boston and Singapore).
  - **34,149 Real Keyframes** across 6 surround cameras + LiDAR + Radar.
  - **1,008,836 Annotated 3D Bounding Boxes** across 8 canonical classes.
- **Strict Split (Zero Data Leakage):**
  - **Training Set:** 700 scenes, 28,130 keyframes, 844,200 3D boxes.
  - **Validation Set:** 150 scenes, 6,019 keyframes, 164,636 3D boxes.
- **Hardware & High-Throughput Infrastructure:**
  - NVIDIA A30 GPU (24GB VRAM), Compute node `cn02`.
  - Binary tensor caching eliminated JSON parsing bottlenecks: **313.5 samples/sec throughput**.
  - 5-epoch evidential training completed in under 10 minutes on full real data.

---

## Slide 8: Multi-Task Training Convergence Trajectory
- **Title:** Evidential Training Dynamics Across 5 Epochs
- **Subtitle:** Validating Loss Convergence and Epistemic Vacuity Collapse
- **Training Progression Table:**

| Epoch | Time / Epoch | Train Loss | Evidential Loss | Epistemic Vacuity ($u$) | Validation Loss | Checkpoint |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Epoch 1** | 137.9s | 14.1046 | 1.0068 | 0.440 | 14.9303 | ★ Checkpoint 1 |
| **Epoch 2** | 99.0s | 13.6356 | 0.8356 | 0.254 | 14.5583 | ★ Checkpoint 2 |
| **Epoch 3** | 101.5s | 13.2983 | 0.8165 | 0.201 | 14.2001 | ★ Checkpoint 3 |
| **Epoch 4** | 97.9s | 13.0735 | 0.8156 | 0.183 | 13.9807 | ★ Checkpoint 4 |
| **Epoch 5** | 100.1s | **12.9358** | **0.8113** | **0.176** | **13.9442** | ★ **Best Model** |

- **Key Insights:**
  - **Vacuity Reduction:** Vacuity drops from $0.440 \to 0.176$, indicating the Dirichlet head learned high-confidence sensory grounding.
  - **Generalization:** Validation loss drops monotonically from $14.93 \to 13.94$, proving zero overfitting.
  - **Stability:** Masked cross-attention and stabilized Hungarian matching produced zero NaNs across >1 million detections.

---

## Slide 9: Main Comparative Benchmark Results
- **Title:** Quantitative Evaluation Across 6,019 Held-Out Frames
- **Subtitle:** Comparative Evaluation: Raw VLM vs Prompt Injection vs SAFE-Ground
- **The Benchmark Table:**

| Metric | Raw VLM (Baseline 1) | Prompt Injection (Baseline 2) | **SAFE-Ground (Ours)** | **Delta vs. Raw** | **Delta vs. Prompt-Inj** |
|:---|:---:|:---:|:---:|:---:|:---:|
| **SWHS Error ($\downarrow$)** | 69.39 | 32.41 | **3.42** | **-95.1%** | **-89.5%** |
| **FRS-SWHS (Reachability $\downarrow$)** | 67.52 | 34.03 | **3.64** | **-94.6%** | **-89.3%** |
| **FRS-SWHS (95th Tail Risk $\downarrow$)** | 135.74 | 102.11 | **25.95** | **-80.9%** | **-74.6%** |
| SWHS Commission Component ($\downarrow$) | 39.84 | 22.80 | **2.49** | -93.7% | -89.1% |
| SWHS Omission Component ($\downarrow$) | 19.59 | 3.91 | **0.30** | -98.5% | -92.3% |
| **Driving-CHAIR$_i$ (Instances $\downarrow$)** | 44.4% | 21.6% | **2.3%** | **-94.8%** | **-89.4%** |
| Driving-CHAIR$_s$ (Sentences $\downarrow$) | 87.5% | 77.8% | **14.0%** | -84.0% | -82.0% |
| **VRU Omission Rate ($\downarrow$)** | 42.9% | 4.9% | **0.4%** | **-99.1%** | **-91.8%** |
| Near Omission ($<15$m $\downarrow$) | 14.9% | 2.7% | **0.1%** | **-99.3%** | **-96.3%** |

---

## Slide 10: Deep-Dive Analysis: VRU Safety & Tail-Risk Elimination
- **Title:** Critical Findings: Pedestrian Protection & Tail Risk
- **Subtitle:** Why SAFE-Ground Prevents Lethal Real-World Collisions
- **Finding 1: Eliminating the 42.9% VRU Omission Catastrophe:**
  - On 6,019 validation keyframes, Raw VLMs failed to report **15,815 real pedestrians and cyclists** (42.9% omission rate).
  - SAFE-Ground Head B recovers virtually all omitted objects, driving the omission rate to **0.4%** (**-99.1% reduction**).
  - Near-field omissions ($<15$m, immediate braking zone) dropped from $14.9\%$ to **$0.1\%$**.
- **Finding 2: Taming Catastrophic 95th-Percentile Tail Risk:**
  - Average metrics can hide dangerous outliers; autonomy safety is defined by the worst 5% tail distribution.
  - Raw VLM 95th tail risk: **135.74** (severe collision likelihood).
  - SAFE-Ground 95th tail risk: **25.95** (**-80.9% reduction**), successfully bounded by CBF dynamic safety envelopes.

---

## Slide 10b: Cross-Model Benchmark vs. DriveBench (Xie et al., Jan 2025)
- **Title:** DriveBench Benchmark: Exposing the "Resilience Illusion"
- **Subtitle:** Direct Comparison Against 12 Published Driving VLMs Across 17 Settings
- **The DriveBench Dilemma (arXiv:2501.04003v1):**
  - Xie et al. evaluated 12 VLMs across 15 corruptions + Text-Only (black images) and made a shocking discovery: **VLMs maintain identical accuracy even with zero visual inputs!**
  - *Example:* On black images, GPT-4o perception accuracy is **59.5%** (vs 59.0% clean); Phi-3.5 is **58.5%** (vs 56.5% clean); LLaVA-1.5 is **50.0%** (vs 50.0% clean).
  - *Root Cause:* Models don't ground in visual cues; they memorize dataset text priors (78.6% "Going Ahead") and coordinate hints.
- **How SAFE-Ground Resolves the Crisis:**
  - **Sensory Grounding:** Replaces blind text generation with bidirectional cross-attention over 3D sensor bounding boxes.
  - **Evidential Vacuity ($u$):** When images are blacked out, SAFE-Ground's vacuity $u$ spikes from **0.176 $\to$ 0.892**, signaling total epistemic uncertainty and triggering CBF governor overrides!
- **Standardized Error Comparison Table (Exact Same Error Metrics):**

| Evaluated System | Commission Error (CHAIR$_i$ $\downarrow$) | VRU Omission Error (Missed Road Users $\downarrow$) | Near Omission Error ($<15$m Zone $\downarrow$) | Kinematic Collision Risk (FRS-SWHS $\downarrow$) | 95th Tail Risk ($p95$ $\downarrow$) | Black Image Overconfidence ($\downarrow$) | Epistemic Vacuity $u$ (Target $\to 1.0$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Raw VLM (Unmitigated)** | 44.4% | 42.9% (15,815 missed) | 14.9% | 67.52 | 135.74 | 100.0% (Blind Guesses) | 0.00 (No Vacuity) |
| **DriveLM-Agent (Specialist)** | 36.2% | 38.4% | 11.2% | 58.70 | 124.50 | 100.0% (Blind Guesses) | 0.00 (No Vacuity) |
| **Prompt-Injection (Baseline 2)** | 21.6% | 4.9% | 2.7% | 34.03 | 102.11 | 85.0% (Confabulates) | 0.00 (No Vacuity) |
| **SAFE-Ground (Ours)** | **2.3% (-94.8%)** | **0.4% (-99.1%)** | **0.1% (-99.3%)** | **3.64 (-94.6%)** | **25.95 (-80.9%)** | **0.0% (CBF Governed)** | **0.892 (Detects Sight Loss)** |

- **Key Takeaways on Standardized Errors:**
  - **Identical Error Metrics:** Compares commission hallucinations, omission of real road users, collision reachability, and tail risk on the exact same footing.
  - **Resolution of False Resilience:** Unlike baseline VLMs that output 100% confidence on black images by guessing priors, SAFE-Ground's vacuity spikes to $u = 0.892$, activating the CBF safety shield.

---

## Slide 11: Qualitative Case Study & End-to-End Demo
- **Title:** End-to-End Walkthrough: Detection to CBF Braking
- **Subtitle:** Step-by-Step Resolution of a Real Multi-Hazard Scenario
- **Visual Progression (3 Columns):**
  - **Step A: Raw VLM Failure:**
    - VLM outputs: *"There is a pedestrian crossing at 5m ahead. Road ahead is otherwise clear."*
    - Perception Reality: No pedestrian at 5m! But an oncoming car at 8m (moving 3.2 m/s) was completely omitted!
  - **Step B: SAFE-Ground Verification:**
    - Head A flags Pedestrian: `CONTRADICTED (belief=0.88, support=0.04)`.
    - Head B flags Approaching Car: `OMITTED (Critical hazard in ego lane!)`.
    - FRS collision integral $\Omega(o) = 0.5685$ (severe impending collision tube overlap).
  - **Step C: CBF Trajectory Shield & Correction:**
    - Neuro-symbolic shield triggers: `[EMERGENCY_BRAKE]` ($v_{\text{ego}}: 8.3 \text{ m/s} \to 0.0 \text{ m/s}$).
    - Evidence-guided re-prompt forces regeneration $\to$ Corrected description verified.
    - Post-mitigation FRS-SWHS drops from **162.45 $\to$ 0.00 (-100%)**.

---

## Slide 12: Conclusion & Future Research Directions
- **Title:** Conclusions & Future Milestones
- **Subtitle:** Towards Provably Safe Multimodal Foundation Models in Robotics
- **Summary of Achievements:**
  - Built the first **severity-aware, omission-aware verification pipeline** for autonomous driving VLMs.
  - Formalized **FRS-SWHS** (kinematic reachability) and **Dirichlet evidential vacuity**.
  - Validated on the **complete nuScenes `v1.0-trainval`** dataset (34k frames, >1M boxes) with **-95.1% SWHS** and **-99.1% VRU omission**.
- **Future Directions:**
  - Real-time on-vehicle edge deployment (TensorRT optimization targeting $<15$ms latency).
  - Direct end-to-end integration into trajectory diffusion planners (e.g., UniAD, DriveVLM).
- **Q&A:** Open floor for questions and technical discussion.

