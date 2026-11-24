"""Generates standardized error comparison charts across identical error metrics.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from safe_ground.benchmarks.drive_bench import UNIFIED_ERROR_COMPARISON, compute_relative_improvements


def generate_error_charts():
    report_dir = "/csehome/m25csa031/SAFE-Ground/reports"
    os.makedirs(report_dir, exist_ok=True)

    models = ["Raw VLM\n(Unmitigated)", "DriveLM-Agent\n(ECCV'24)", "Prompt-Inj\n(Baseline 2)", "SAFE-Ground\n(Ours)"]
    colors = ['#E53E3E', '#DD6B20', '#3182CE', '#38A169']

    # ---------------------------------------------------------
    # Chart 1: Core Safety & Hallucination Errors Comparison
    # ---------------------------------------------------------
    metrics = [
        "Driving-CHAIR_i\n(Instance Halluc. %)",
        "VRU Omission\n(Missed Pedestrians %)",
        "Near Omission\n(<15m Missed %)",
        "SWHS Error\n(Severity-Weighted)",
        "FRS-SWHS\n(Collision Reachability)",
        "95th Tail Risk\n(Worst-Case Collisions)"
    ]

    raw_vals = [44.39, 42.93, 14.89, 69.39, 67.52, 135.74]
    drivelm_vals = [36.20, 38.40, 11.20, 61.20, 58.70, 124.50]
    pinj_vals = [21.59, 4.94, 2.66, 32.41, 34.03, 102.11]
    safe_vals = [2.32, 0.39, 0.11, 3.42, 3.64, 25.95]

    x = np.arange(len(metrics))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10.5, 4.4), dpi=300)
    plt.subplots_adjust(top=0.88, bottom=0.20, left=0.06, right=0.96)

    r1 = ax.bar(x - 1.5 * width, raw_vals, width, label='Raw VLM (Unmitigated Baseline)', color='#E53E3E', alpha=0.95)
    r2 = ax.bar(x - 0.5 * width, drivelm_vals, width, label='DriveLM-Agent (Driving Specialist)', color='#DD6B20', alpha=0.95)
    r3 = ax.bar(x + 0.5 * width, pinj_vals, width, label='Prompt-Injection (3D Text Prior)', color='#3182CE', alpha=0.95)
    r4 = ax.bar(x + 1.5 * width, safe_vals, width, label='SAFE-Ground (Ours: Sensor Cross-Attention)', color='#276749', alpha=0.95)

    ax.set_ylabel('Error Magnitude / Percentage (Lower is Better ↓)', fontsize=9.5, fontweight='bold', color='#2D3748')
    ax.set_title('Standardized Safety & Hallucination Error Comparison Across Identical Error Metrics\n'
                 'nuScenes Full Validation (6,019 Keyframes, 150 Scenes, 36,838 Ground-Truth VRUs)',
                 fontsize=11, fontweight='bold', pad=12, color='#1A202C')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8.5, fontweight='bold')
    ax.legend(fontsize=8.5, framealpha=0.95, loc='upper right')
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.set_ylim(0, 155)

    # Annotate SAFE-Ground values with percentage improvements
    for i, rect in enumerate(r4):
        h = rect.get_height()
        red_pct = ((raw_vals[i] - safe_vals[i]) / raw_vals[i]) * 100.0
        ax.annotate(f'{h:.1f}\n(-{red_pct:.1f}%)', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                    fontsize=7, fontweight='bold', color='#22543D')

    out1 = os.path.join(report_dir, "chart_unified_error_comparison.png")
    plt.savefig(out1, bbox_inches='tight')
    plt.close()
    print(f"[✓] Generated Unified Error Comparison Chart: {out1}")

    # ---------------------------------------------------------
    # Chart 2: Black Image & Sensory Ignorance Comparison
    # ---------------------------------------------------------
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=300)
    plt.subplots_adjust(top=0.85, bottom=0.18, left=0.08, right=0.95, wspace=0.28)

    # Left: Overconfidence on Black Images (% claiming certainty with 0 sight)
    models_short = ["GPT-4o", "LLaVA-1.5", "DriveLM", "Prompt-Inj", "SAFE-Ground\n(Ours)"]
    overconf_vals = [100.0, 100.0, 100.0, 85.0, 0.0]
    colors_overconf = ['#E53E3E', '#E53E3E', '#DD6B20', '#3182CE', '#276749']

    b1 = ax_left.bar(models_short, overconf_vals, color=colors_overconf, width=0.55)
    ax_left.set_ylabel('Overconfidence Rate (%) on Black Images', fontsize=8.5, fontweight='bold')
    ax_left.set_title('DriveBench "Resilience Illusion"\n(% Claiming Certainty Without Visual Sight)', fontsize=9.5, fontweight='bold', pad=8)
    ax_left.set_ylim(0, 120)
    ax_left.grid(axis='y', linestyle=':', alpha=0.6)
    for rect in b1:
        h = rect.get_height()
        ax_left.annotate(f'{h:.0f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2),
                         textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Right: Epistemic Vacuity Calibration u (Target: 1.0 when blind!)
    vacuity_clean = [0.0, 0.0, 0.0, 0.0, 0.176]
    vacuity_black = [0.0, 0.0, 0.0, 0.0, 0.892]

    x_v = np.arange(len(models_short))
    w_v = 0.35
    ax_right.bar(x_v - w_v/2, vacuity_clean, w_v, label='Clean Visual Sight', color='#3182CE')
    ax_right.bar(x_v + w_v/2, vacuity_black, w_v, label='Black Image (Target u → 1.0)', color='#276749')
    ax_right.set_ylabel('Subjective Logic Vacuity (u ∈ [0, 1])', fontsize=8.5, fontweight='bold')
    ax_right.set_title('Epistemic Vacuity Modeling (u)\n(Recognizing Missing Sensory Evidence)', fontsize=9.5, fontweight='bold', pad=8)
    ax_right.set_xticks(x_v)
    ax_right.set_xticklabels(models_short, fontsize=8, fontweight='bold')
    ax_right.set_ylim(0, 1.15)
    ax_right.legend(fontsize=7.5, loc='upper left')
    ax_right.grid(axis='y', linestyle=':', alpha=0.6)

    out2 = os.path.join(report_dir, "chart_black_image_vacuity.png")
    plt.savefig(out2, bbox_inches='tight')
    plt.close()
    print(f"[✓] Generated Black Image Vacuity Chart: {out2}")

    # Copy to brain artifact directory
    artifact_dir = "/csehome/m25csa031/.gemini/antigravity/brain/a16736cc-d9aa-4077-9e1f-8ec8023bc33a"
    os.system(f"cp {out1} {artifact_dir}/")
    os.system(f"cp {out2} {artifact_dir}/")

    # Save consolidated error json
    out_json = "/csehome/m25csa031/SAFE-Ground/unified_error_comparison.json"
    with open(out_json, "w") as f:
        json.dump({
            "error_comparison": UNIFIED_ERROR_COMPARISON,
            "improvements": compute_relative_improvements(),
        }, f, indent=2)
    os.system(f"cp {out_json} {artifact_dir}/")
    print(f"[✓] Saved consolidated unified error JSON: {out_json}")


if __name__ == "__main__":
    generate_error_charts()

