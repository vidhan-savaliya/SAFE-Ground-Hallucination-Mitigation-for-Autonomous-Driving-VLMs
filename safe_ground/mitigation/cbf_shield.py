"""Pillar 4: Neuro-Symbolic Control Barrier Function (CBF) Safety Shield.

Directly connects VLM verification and epistemic vacuity to the vehicle's
kinematic control loop, adjusting safety margins and emergency braking profiles.
"""

from typing import Dict, Any, Tuple
import math


class NeuroSymbolicSafetyShield:
    """Modulates trajectory planner speed and emergency braking envelope using verification risk."""

    def __init__(
        self,
        nominal_speed_mps: float = 8.33,  # ~30 km/h
        max_deceleration_mps2: float = 6.0,  # emergency braking limit
        margin_expansion_xi: float = 0.02,
    ):
        self.v_nom = nominal_speed_mps
        self.a_max = max_deceleration_mps2
        self.xi = margin_expansion_xi

    def compute_safe_control(
        self,
        predicted_hallucination_risk: float,
        epistemic_vacuity: float,
        nearest_obstacle_dist: float,
    ) -> Dict[str, Any]:
        """Solves the safety filtering condition:
        
        v_safe = v_nom * max(0.0, 1.0 - xi * (H_risk * (1 + vacuity)))
        """
        # Composite risk: higher epistemic vacuity amplifies caution
        effective_risk = predicted_hallucination_risk * (1.0 + 1.5 * epistemic_vacuity)

        speed_reduction_factor = max(0.0, 1.0 - self.xi * effective_risk)
        v_safe = self.v_nom * speed_reduction_factor

        # Dynamic CBF barrier margin expansion
        # Standard 2.5m buffer expands when visual scene understanding is untrusted
        cbf_buffer_expanded = 2.5 + 0.1 * min(effective_risk, 30.0)

        # Check required stopping distance: d_stop = v^2 / (2 * a_max)
        stopping_distance = (self.v_nom ** 2) / (2.0 * self.a_max)
        imminent_collision = nearest_obstacle_dist < (stopping_distance + cbf_buffer_expanded)

        action = "NORMAL"
        if imminent_collision and effective_risk > 15.0:
            action = "EMERGENCY_BRAKE"
            v_safe = 0.0
        elif speed_reduction_factor < 0.6:
            action = "DYNAMIC_CAUTION_SLOWDOWN"

        return {
            "action": action,
            "commanded_speed_mps": v_safe,
            "nominal_speed_mps": self.v_nom,
            "speed_reduction_pct": (1.0 - (v_safe / self.v_nom)) * 100.0,
            "expanded_barrier_radius": cbf_buffer_expanded,
            "composite_hallucination_risk": effective_risk,
        }

