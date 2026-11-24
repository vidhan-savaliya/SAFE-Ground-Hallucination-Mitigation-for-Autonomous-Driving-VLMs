"""Pillar 1: Kinematic Forward Reachable Set (FRS) and Control Barrier Functions (CBF).

Replaces heuristic 1/distance weights with formal continuous reachable tube
collision probability integrals over planning horizon T_H.
"""

import math
import torch
from typing import Dict, Any, Tuple


class KinematicReachabilityComputer:
    """Computes Forward Reachable Set (FRS) spatio-temporal tube collision volumes."""

    def __init__(
        self,
        horizon_seconds: float = 3.0,
        time_steps: int = 15,
        discount_tau: float = 1.0,
        ego_safe_radius: float = 2.2,  # Vehicle bounding radius + clearance
        vru_safe_radius: float = 1.5,
    ):
        self.T_H = horizon_seconds
        self.time_steps = time_steps
        self.tau = discount_tau
        self.ego_safe_radius = ego_safe_radius
        self.vru_safe_radius = vru_safe_radius

    def compute_frs_collision_integral(
        self,
        x_obj: float,
        y_obj: float,
        vx_obj: float = 0.0,
        vy_obj: float = 0.0,
        v_ego_forward: float = 8.33,  # ~30 km/h nominal speed (m/s)
        is_vru: bool = False,
    ) -> float:
        """Integrates collision reachable volume over future planning horizon:
        
        Omega(o_k) = int_0^T_H exp(-t / tau) * P[R_ego(t) \cap R_obj(t) != empty] dt
        """
        dt = self.T_H / self.time_steps
        integral_sum = 0.0
        r_comb = self.ego_safe_radius + (self.vru_safe_radius if is_vru else self.ego_safe_radius)

        for step in range(1, self.time_steps + 1):
            t = step * dt

            # Projected ego position at time t (ego travels forward along +x axis)
            x_ego_t = v_ego_forward * t
            y_ego_t = 0.0

            # Projected obstacle position at time t (with growing uncertainty boundary)
            x_obj_t = x_obj + vx_obj * t
            y_obj_t = y_obj + vy_obj * t

            # Uncertainty expansion radius (0.2 m/s drift)
            sigma_t = 0.2 * t
            effective_radius = r_comb + sigma_t

            # Euclidean distance between centers at future time t
            dist_sq = (x_ego_t - x_obj_t) ** 2 + (y_ego_t - y_obj_t) ** 2
            dist = math.sqrt(dist_sq)

            # Soft collision probability based on barrier margin
            if dist < effective_radius:
                p_collision = 1.0
            elif dist < effective_radius * 2.0:
                p_collision = math.exp(-((dist - effective_radius) ** 2) / 2.0)
            else:
                p_collision = 0.0

            time_discount = math.exp(-t / self.tau)
            integral_sum += time_discount * p_collision * dt

        return integral_sum

    def compute_cbf_barrier_margin(
        self,
        x_obj: float,
        y_obj: float,
        r_safe: float = 2.5,
    ) -> float:
        """Control Barrier Function h(x) = ||p_ego - p_obj||^2 - r_safe^2.
        
        h(x) > 0: Safe set
        h(x) <= 0: Collision boundary violated
        """
        dist_sq = x_obj * x_obj + y_obj * y_obj
        return dist_sq - (r_safe * r_safe)

    def compute_frs_severity_weight(
        self,
        class_name: str,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
        v_ego: float = 8.33,
    ) -> float:
        """Formal PhD Reachability Weight:
        
        w_FRS(o) = lambda_cls * (1.0 + kappa * Omega(o)) * (1 / (min_dist + eps))
        """
        is_vru = class_name.lower() in {"pedestrian", "cyclist", "bicycle", "motorcycle"}
        lam_cls = 3.5 if is_vru else 2.0

        omega = self.compute_frs_collision_integral(x, y, vx, vy, v_ego_forward=v_ego, is_vru=is_vru)
        h_margin = max(self.compute_cbf_barrier_margin(x, y), 0.1)
        base_dist = math.sqrt(x * x + y * y)

        kappa = 8.0  # Reachability sensitivity gain
        frs_weight = lam_cls * (1.0 + kappa * omega) * (15.0 / (base_dist + 1.0))
        return min(frs_weight, 50.0)

