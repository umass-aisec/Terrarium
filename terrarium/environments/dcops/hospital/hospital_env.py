import logging
import math
import json
import random
from typing import Dict, List, Any, Tuple
from terrarium.utils import get_run_timestamp, build_log_dir, get_tag_model_subdir
from terrarium.environments.abstract_environment import AbstractEnvironment
from .hospital_prompts import HospitalPrompts

logger = logging.getLogger(__name__)

class HospitalEnvironment(AbstractEnvironment):
    def __init__(self, communication_protocol, config, tool_logger):
        self.full_config = config
        self.env_config = config["environment"]
        self.simulation_config = config["simulation"]
        self.current_seed = int(self.simulation_config.get("seed", 42))
        random.seed(self.current_seed)
        
        self.run_timestamp = get_run_timestamp(self.full_config)
        self.tool_logger = tool_logger
        self.communication_protocol = communication_protocol
        self.communication_protocol.environment = self

        self.resource_types = [
            "IV_Kits", "Anesthetics", "Pain_Killers", 
            "Radio_Contrast", "Oxygen_Tanks", "Surgical_Packs", "PPE_Sets"
        ]
        
        self.consumption_map = {
            "Surgery":   {"Anesthetics": 1, "Surgical_Packs": 1, "PPE_Sets": 2, "IV_Kits": 1, "Pain_Killers": 1},
            "Radiology": {"Radio_Contrast": 1, "PPE_Sets": 1, "IV_Kits": 1},
            "Ward":      {"Oxygen_Tanks": 1, "IV_Kits": 1, "Pain_Killers": 1, "PPE_Sets": 1},
            "Triage":    {"PPE_Sets": 1, "IV_Kits": 1} 
        }

        # Pathway Templates
        self.possible_pathways = [
            [("Triage", 1), ("Radiology", 1), ("Surgery", 4), ("Ward", 48)],
            [("Triage", 1), ("Ward", 24)],
            [("Triage", 1), ("Radiology", 2), ("Ward", 12)]
        ]

        self.hospital_names = self._get_hospital_names()
        self.agents_map, self.patients = self._generate_flexible_jobshop_data()
        self.agent_names = list(self.agents_map.keys())
        self.prompts = HospitalPrompts(self, self.full_config)
        
        self.schedule: Dict[str, Dict[int, List[str]]] = {a: {} for a in self.agent_names if a != "Resource_Provisioner"}
        self.patient_states: Dict[str, Dict] = {p: {"scheduled_steps": {}, "resource_failures": []} for p in self.patients}

        self.inventory = self._generate_scarcity_scenario()
        
        # Track GLOBAL failures for scoring
        self.resource_failures = {rt: 0 for rt in self.resource_types}

        # --- NEW: Track LOCAL failures for Provisioner Intelligence ---
        # Maps "General_Hospital" -> {"IV_Kits": 5, ...}
        self.hospital_failures = {
            h: {rt: 0 for rt in self.resource_types} for h in self.hospital_names
        }
        
        # Track Peer-to-Peer Transfer Rewards
        self.transfer_rewards = 0.0
        # Attribution for P2P transfer rewards (donor agent who executed the transfer).
        self.transfer_rewards_by_agent: Dict[str, float] = {}

        self.max_time_horizon = 168
        self.transfer_penalty_hours = 4
        self.theoretical_max_score = len(self.patients) * 1000.0 
        self.joint_reward_history = []
        self.agent_rewards_history = {a: [] for a in self.agent_names}

    def compute_max_joint_reward(self) -> float:
        return self.theoretical_max_score

    def _get_hospital_names(self) -> List[str]:
        configured = self.env_config.get("hospital_names")
        if configured:
            names = [str(x).strip() for x in configured if str(x).strip()]
            if names:
                return names

        num_hospitals = int(self.env_config.get("num_hospitals", 2))
        if num_hospitals <= 0:
            num_hospitals = 2

        # Backwards-compatible defaults for the first two hospitals.
        defaults = ["General_Hospital", "St_Marys_Center"]
        if num_hospitals <= len(defaults):
            return defaults[:num_hospitals]

        extra = [f"Hospital_{i}" for i in range(len(defaults), num_hospitals)]
        return defaults + extra

    def _generate_flexible_jobshop_data(self) -> Tuple[Dict, Dict]:
        hospitals = list(self.hospital_names)
        services = [
            {"name": "Triage", "capacity": 4, "duration": 1},
            {"name": "Radiology", "capacity": 2, "duration": 2},
            {"name": "Surgery", "capacity": 1, "duration": 4},
            {"name": "Ward", "capacity": 8, "duration": 24}
        ]
        
        agents_map = {}
        for h in hospitals:
            for s in services:
                aid = f"{h}_{s['name']}"
                agents_map[aid] = {
                    "id": aid, "hospital": h, "service": s["name"],
                    "capacity": s["capacity"], "default_duration": s["duration"]
                }
        
        agents_map["Resource_Provisioner"] = {
            "id": "Resource_Provisioner", 
            "hospital": "Global_Logistics", 
            "service": "Provisioning", 
            "capacity": 9999 
        }

        patients = {}
        for i in range(self.env_config.get("num_patients", 8)):
            pid = f"Patient_{i}"
            template = random.choice(self.possible_pathways)
            steps = [{"step_index": idx, "service": s, "duration": d} 
                     for idx, (s, d) in enumerate(template)]
            patients[pid] = {
                "id": pid, "arrival_time": random.randint(0, 24), "pathway": steps
            }
        return agents_map, patients

    def _generate_scarcity_scenario(self) -> Dict[str, Dict[str, int]]:
        avg_costs = {rt: 0.0 for rt in self.resource_types}
        for path in self.possible_pathways:
            path_cost = {rt: 0 for rt in self.resource_types}
            for step in path:
                service_name = step[0]
                consumption = self.consumption_map.get(service_name, {})
                for res, qty in consumption.items():
                    path_cost[res] += qty
            for rt in self.resource_types:
                avg_costs[rt] += path_cost.get(rt, 0)
        
        for rt in avg_costs: avg_costs[rt] /= len(self.possible_pathways)
        num_patients = self.env_config.get("num_patients", 8)
        
        inventory = {h: {rt: 0 for rt in self.resource_types} for h in self.hospital_names}
        inventory["Resource_Provisioner"] = {rt: 0 for rt in self.resource_types}

        for rt in self.resource_types:
            velocity = avg_costs.get(rt, 0)
            local_need = math.ceil(velocity * num_patients * 0.6)
            safety = local_need + 5
            if velocity > 0 and safety < 3: safety = 3
            
            for h in self.hospital_names:
                inventory[h][rt] = safety
            
            prov_stock = math.ceil(velocity * num_patients * 2.5) + 10
            inventory["Resource_Provisioner"][rt] = prov_stock

        return inventory

    def _process_transfer(self, sender_id: str, action: Dict[str, Any]):
        dst = action.get("to_hospital")
        res_type = action.get("resource_type")
        amt = action.get("amount", 0)
        rationale = action.get("rationale", "No rationale")

        sender_hospital = self.agents_map[sender_id].get("hospital")
        if sender_id == "Resource_Provisioner": sender_hospital = "Resource_Provisioner"

        dst_s = str(dst) if dst is not None else ""
        if dst_s in self.inventory:
            dst_hospital = dst_s
        elif "General" in dst_s:
            dst_hospital = "General_Hospital"
        elif "Mary" in dst_s:
            dst_hospital = "St_Marys_Center"
        elif "Provisioner" in dst_s:
            dst_hospital = "Resource_Provisioner"
        else:
            dst_hospital = dst_s

        if sender_hospital in self.inventory and dst_hospital in self.inventory and res_type in self.resource_types:
            available = self.inventory[sender_hospital].get(res_type, 0)
            actual_move = min(amt, available)
            
            if actual_move > 0:
                self.inventory[sender_hospital][res_type] -= actual_move
                self.inventory[dst_hospital][res_type] += actual_move
                
                # --- Incentive for Hospital-to-Hospital Transfers ---
                if sender_hospital != "Resource_Provisioner" and sender_hospital != dst_hospital:
                    # Reward: +15 points per unit (Offsets holding cost of 10 + small bonus)
                    reward = actual_move * 15.0
                    self.transfer_rewards += reward
                    self.transfer_rewards_by_agent[sender_id] = float(
                        self.transfer_rewards_by_agent.get(sender_id, 0.0) + reward
                    )
                    logger.info(f"P2P TRANSFER REWARD: +{reward} for moving {actual_move} {res_type}")

                status = "FULL" if actual_move == amt else f"PARTIAL ({actual_move}/{amt})"
                logger.info(f"TRANSFER [{status}]: {sender_id} -> {dst_hospital}: {actual_move} {res_type}. Rationale: {rationale}")
            else:
                logger.warning(f"TRANSFER FAILED [{sender_id}]: Requested {amt} {res_type}, had {available}.")

    def _process_schedule_request(self, agent_id, action):
        p_id, step_idx, start = action.get("patient_id"), action.get("step_index"), action.get("start_time")
        
        logger.info(f"SCHEDULE [{agent_id}]: Patient {p_id} @ {start}.")

        if p_id not in self.patients: return

        patient = self.patients[p_id]
        if step_idx >= len(patient["pathway"]): return
        
        target_step = patient["pathway"][step_idx]
        agent_info = self.agents_map[agent_id]
        
        if agent_info["service"] != target_step["service"]: return
        
        min_start = patient["arrival_time"]
        if step_idx > 0:
            prev = self.patient_states[p_id]["scheduled_steps"].get(step_idx - 1)
            if not prev: return 
            prev_h = self.agents_map[prev["agent"]]["hospital"]
            curr_h = agent_info["hospital"]
            penalty = self.transfer_penalty_hours if prev_h != curr_h else 0
            min_start = max(min_start, prev["end_time"] + penalty)
            
        if start < min_start: return
            
        dur = target_step["duration"]
        for t in range(start, start + dur):
            if t >= self.max_time_horizon: return
            if len(self.schedule[agent_id].get(t, [])) >= agent_info["capacity"]: return
            
        hospital = agent_info["hospital"]
        service = agent_info["service"]
        costs = self.consumption_map.get(service, {})
        
        missing = []
        for res, req_amt in costs.items():
            if self.inventory[hospital].get(res, 0) >= req_amt:
                self.inventory[hospital][res] -= req_amt
            else:
                self.inventory[hospital][res] = 0 
                missing.append(res)
                self.resource_failures[res] += 1
                
                # --- NEW: Record specific hospital failure ---
                if hospital in self.hospital_failures:
                    self.hospital_failures[hospital][res] += 1
        
        if missing:
            logger.warning(f"RESOURCE FAILURE [{agent_id}]: Scheduled {p_id} missing {missing}.")
            if "resource_failures" not in self.patient_states[p_id]:
                self.patient_states[p_id]["resource_failures"] = []
            self.patient_states[p_id]["resource_failures"].append(
                {"step": step_idx, "missing": missing, "agent": agent_id}
            )

        for t in range(start, start + dur):
            if t not in self.schedule[agent_id]: self.schedule[agent_id][t] = []
            self.schedule[agent_id][t].append(p_id)
            
        self.patient_states[p_id]["scheduled_steps"][step_idx] = {
            "agent": agent_id, "start_time": start, "end_time": start + dur
        }

    def _calculate_makespan_and_flow(self):
        """
        Joint reward:
          score = theoretical_max_score - total_flow - resource_penalty - holding_cost + transfer_rewards

        Per-agent rewards returned by this method are an *exact decomposition* of the joint reward:
          - Base: each patient contributes +1000, split evenly across its pathway steps (1000/len(path) per step),
            attributed to the agent responsible for that step.
          - Flow-time penalty: for each patient, split its flow penalty evenly across the same pathway steps
            (patient-based attribution).
          - Resource failure penalty: -300 per missing resource item, attributed to the agent where it occurs.
          - Holding cost: -10 per leftover unit of hospital inventory, shared uniformly across that hospital's agents.
          - Transfer rewards: attributed to the donor agent that executed the rewarded transfer.

        By construction: sum(agent_rewards.values()) == joint_reward (up to float error).
        """
        total_flow = 0.0
        missed_step_penalty = 500.0
        agent_rewards: Dict[str, float] = {a: 0.0 for a in self.agent_names}

        def _infer_step_agent(
            *,
            patient_id: str,
            step_idx: int,
            service: str,
            sched: Dict[int, Dict[str, Any]],
        ) -> str:
            info = sched.get(step_idx)
            if isinstance(info, dict):
                a = str(info.get("agent") or "")
                if a:
                    return a

            # If a step isn't scheduled, infer a plausible agent so we can still allocate
            # missed-step penalties and the base per-patient reward consistently.
            inferred_hospital = self.hospital_names[0] if self.hospital_names else "General_Hospital"
            if step_idx > 0:
                prev = sched.get(step_idx - 1)
                if isinstance(prev, dict):
                    prev_a = str(prev.get("agent") or "")
                    if prev_a and prev_a in self.agents_map:
                        inferred_hospital = str(self.agents_map[prev_a].get("hospital") or inferred_hospital)
            return f"{inferred_hospital}_{service}"

        # Patient-based base + flow attribution.
        for pid, p in self.patients.items():
            sched = self.patient_states[pid]["scheduled_steps"]
            path = p.get("pathway") or []
            if not path:
                continue

            step_credit = 1000.0 / float(len(path))

            step_agents: List[str] = []
            for step in path:
                idx = int(step.get("step_index"))
                service = str(step.get("service"))
                step_agents.append(
                    _infer_step_agent(patient_id=pid, step_idx=idx, service=service, sched=sched)
                )

            # Base credit: sums to +1000 per patient.
            for a in step_agents:
                if a in agent_rewards:
                    agent_rewards[a] += step_credit

            # Flow penalty: matches prior logic.
            if len(sched) == len(path):
                flow = float(sched[len(path) - 1]["end_time"]) - float(p["arrival_time"])
            else:
                flow = float(len(path) - len(sched)) * float(missed_step_penalty)
            total_flow += float(flow)

            per_step_flow = float(flow) / float(len(path))
            for a in step_agents:
                if a in agent_rewards:
                    agent_rewards[a] -= per_step_flow

        # Resource failures: exact -300 per missing resource item (type), attributed to failing agent.
        total_failures = float(sum(self.resource_failures.values()))
        resource_penalty = total_failures * 300.0
        for pid in self.patients.keys():
            failures = self.patient_states.get(pid, {}).get("resource_failures", []) or []
            for ev in failures:
                agent = str(ev.get("agent") or "")
                missing = ev.get("missing") or []
                if not agent or agent not in agent_rewards:
                    continue
                if isinstance(missing, list) and missing:
                    agent_rewards[agent] -= 300.0 * float(len(missing))

        # Holding cost: exact -10 per leftover unit in hospital inventories; shared uniformly among hospital agents.
        holding_cost = 0.0
        holding_units_by_hospital: Dict[str, float] = {}
        for h in self.hospital_names:
            units = 0.0
            if h in self.inventory:
                for qty in self.inventory[h].values():
                    units += float(qty)
            holding_units_by_hospital[h] = units
            holding_cost += units * 10.0

        for h in self.hospital_names:
            units = float(holding_units_by_hospital.get(h, 0.0))
            if units <= 0.0:
                continue
            agents_h = [
                a
                for a, info in self.agents_map.items()
                if info.get("hospital") == h and a != "Resource_Provisioner"
            ]
            if not agents_h:
                continue
            share = (units * 10.0) / float(len(agents_h))
            for a in agents_h:
                if a in agent_rewards:
                    agent_rewards[a] -= share

        # Transfer rewards: attribute to donor agent that executed the rewarded transfer.
        for a, v in (self.transfer_rewards_by_agent or {}).items():
            if a in agent_rewards:
                agent_rewards[a] += float(v)

        # Score includes transfer rewards.
        score = self.theoretical_max_score - total_flow - resource_penalty - holding_cost + self.transfer_rewards

        # Debug sanity check (should be exact by construction).
        try:
            tot = float(sum(agent_rewards.values()))
            if abs(tot - float(score)) > 1e-6:
                logger.warning(
                    f"agent_rewards sum mismatch: sum={tot:.6f} score={float(score):.6f} diff={tot-float(score):.6f}"
                )
        except Exception:
            pass

        return score, agent_rewards

    def joint_reward(self, actions):
        s, _ = self._calculate_makespan_and_flow()
        return s

    def agent_reward(self, agent_name, action):
        _, r = self._calculate_makespan_and_flow()
        return r.get(agent_name, 0.0)

    def _build_department_cost_table(self, hospital: str, service: str, job_queue: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        local_inventory = self.inventory.get(hospital, {}) or {}
        procedure_costs = self.consumption_map.get(service, {}) or {}

        cost_table: List[Dict[str, Any]] = []
        for job in job_queue:
            missing: Dict[str, int] = {}
            missing_types = 0
            for resource_type, required_qty in procedure_costs.items():
                on_hand = int(local_inventory.get(resource_type, 0) or 0)
                deficit = max(0, int(required_qty) - on_hand)
                if deficit > 0:
                    missing[resource_type] = deficit
                    missing_types += 1

            # Local heuristic aligned with the env decomposition:
            # - base credit per step is +1000/len(path) (typically +250); we subtract it from "cost"
            # - each missing resource type implies a likely -300 failure penalty
            # - earlier completion reduces patient flow; use earliest_end_time as a proxy (scaled by path length)
            pid = str(job.get("patient_id") or "")
            path_len = float(len((self.patients.get(pid, {}) or {}).get("pathway") or [])) if pid in self.patients else 0.0
            path_len = float(path_len) if path_len > 0 else 4.0
            step_credit = 1000.0 / path_len

            resource_penalty_est = float(missing_types) * 300.0
            earliest_end = float(job.get("earliest_start_time", 0) or 0) + float(job.get("duration", 0) or 0)
            flow_share_est = earliest_end / path_len
            cost = resource_penalty_est + flow_share_est - step_credit

            cost_table.append(
                {
                    "job_id": f"{job.get('patient_id')}::step{job.get('step_index')}",
                    "patient_id": job.get("patient_id"),
                    "step_index": job.get("step_index"),
                    "cost": float(cost),
                    "feasible": missing_types == 0,
                    "missing": missing,
                }
            )

        cost_table.sort(key=lambda item: (not item.get("feasible", False), float(item.get("cost", float("inf")))))
        return cost_table

    def _build_provisioner_cost_table(self, failures_by_hospital: Dict[str, Dict[str, int]]) -> List[Dict[str, Any]]:
        prov_inventory = self.inventory.get("Resource_Provisioner", {}) or {}

        # Heuristic targets informed by the prompt guidance:
        # - treat <2 as "critical shortage"
        # - top up toward 5 units
        critical_threshold = 2
        target_stock = 5

        cost_table: List[Dict[str, Any]] = []
        for hospital in self.hospital_names:
            local_inventory = self.inventory.get(hospital, {}) or {}
            failures = failures_by_hospital.get(hospital, {}) or {}
            for resource_type in self.resource_types:
                on_hand = int(local_inventory.get(resource_type, 0) or 0)
                prov_on_hand = int(prov_inventory.get(resource_type, 0) or 0)
                failure_count = int(failures.get(resource_type, 0) or 0)

                shortage = max(0, critical_threshold - on_hand)
                recommended = max(0, target_stock - on_hand) if (failure_count > 0 or on_hand < critical_threshold) else 0
                if recommended <= 0:
                    continue

                amount = min(recommended, prov_on_hand)
                feasible = amount > 0

                # Lower cost => more urgent/valuable to do.
                #
                # Align this heuristic with the environment's reward decomposition:
                # - Preventing a failure avoids ~300 points per missing item.
                # - Small shortages indicate risk of near-term failures/missed steps (soft weight).
                # - Sending inventory that goes unused incurs holding cost (-10/unit) at hospitals.
                #
                # Provisioner->hospital transfers do not earn transfer_rewards; treat this as stabilization only.
                benefit_est = float(failure_count) * 300.0 + float(shortage) * 150.0
                holding_risk_est = 10.0 * float(amount)
                # Net "cost": holding risk minus expected benefit (lower is better).
                cost = holding_risk_est - benefit_est

                reason_bits = []
                if failure_count > 0:
                    reason_bits.append(f"{failure_count} failures")
                if on_hand < critical_threshold:
                    reason_bits.append(f"critical stock ({on_hand}<{critical_threshold})")
                reason = ", ".join(reason_bits) if reason_bits else "top-up"

                cost_table.append(
                    {
                        "transfer_id": f"{hospital}::{resource_type}",
                        "to_hospital": hospital,
                        "resource_type": resource_type,
                        "available": int(prov_on_hand),
                        "amount": int(amount),
                        "requested": int(recommended),
                        "benefit_est": float(benefit_est),
                        "holding_risk_est": float(holding_risk_est),
                        "cost": float(cost),
                        "feasible": bool(feasible),
                        "reason": reason,
                    }
                )

        cost_table.sort(key=lambda item: (not item.get("feasible", False), float(item.get("cost", float("inf")))))
        return cost_table

    def build_agent_context(self, agent_name, phase, iteration, **kwargs):
        my_info = self.agents_map[agent_name]
        
        if agent_name == "Resource_Provisioner":
            # --- NEW: Filter failure map to only show active problems ---
            active_failures = {h: f for h, f in self.hospital_failures.items() if sum(f.values()) > 0}
            
            return {
                "agent_name": agent_name, "phase": phase, "iteration": iteration,
                "role": "provisioner",
                "inventory_snapshot": self.inventory,
                "failures_so_far": self.resource_failures,
                "failures_by_hospital": active_failures,  # <--- Granular visibility
                "cost_table": self._build_provisioner_cost_table(active_failures),
            }

        summary = {}
        for t in range(0, 48, 4):
            load = sum(len(self.schedule[agent_name].get(t+i, [])) for i in range(4)) / 4.0
            summary[f"H{t}"] = f"{load:.1f}/{my_info['capacity']}"
            
        queue = []
        for pid, p in self.patients.items():
            done = self.patient_states[pid]["scheduled_steps"]
            idx = len(done)
            if idx < len(p["pathway"]):
                step = p["pathway"][idx]
                if step["service"] == my_info["service"]:
                    start = p["arrival_time"]
                    note = ""
                    if idx > 0:
                        prev = done[idx-1]
                        prev_h = self.agents_map[prev["agent"]]["hospital"]
                        if prev_h != my_info["hospital"]:
                            note = "(Transfer)"
                            start = max(start, prev["end_time"] + self.transfer_penalty_hours)
                        else:
                            start = max(start, prev["end_time"])
                    queue.append({
                        "patient_id": pid, "step_index": idx, 
                        "duration": step["duration"], "earliest_start_time": start, "note": note
                    })

        cost_table = self._build_department_cost_table(
            hospital=my_info["hospital"],
            service=my_info["service"],
            job_queue=queue,
        )
        return {
            "agent_name": agent_name, "phase": phase, "iteration": iteration,
            "role": "department",
            "dept_info": {
                "capacity": my_info["capacity"], 
                "service": my_info["service"], 
                "hospital": my_info["hospital"],
                "local_inventory": self.inventory.get(my_info["hospital"], {}),
                "procedure_costs": self.consumption_map.get(my_info["service"], {}),
                "schedule_summary": summary
            },
            "job_queue": queue,
            "cost_table": cost_table,
        }

    def done(self, iteration):
        # NOTE: `examples/base_main.py` already bounds the loop by `simulation.max_iterations`.
        # This method is for early-stopping when the schedule has truly converged.
        max_iterations = self.simulation_config.get(
            "max_iterations", self.env_config.get("max_iterations", 10)
        )
        try:
            if iteration > int(max_iterations):
                return True
        except Exception:
            pass

        # Converged only if every patient has all steps scheduled AND no resource failures occurred.
        for pid, patient in self.patients.items():
            scheduled = self.patient_states.get(pid, {}).get("scheduled_steps", {})
            pathway = patient.get("pathway", [])
            if len(scheduled) != len(pathway):
                return False
            if self.patient_states.get(pid, {}).get("resource_failures"):
                return False
        return True

    def get_network_context(self): return "Resource-Constrained Job Shop with Provisioner."
    async def async_init(self): await super().async_init()
    def log_iteration(self, iteration):
        s, r = self._calculate_makespan_and_flow()
        self.joint_reward_history.append(s)
        for a, v in r.items(): self.agent_rewards_history[a].append(v)
        tag = get_tag_model_subdir(self.full_config)
        log_dir = build_log_dir(self.__class__.__name__, tag, self.current_seed, self.run_timestamp)
        log_dir.mkdir(parents=True, exist_ok=True)
        if iteration == 1:
            with open(log_dir / "patients.json", "w") as f:
                json.dump(self.patients, f, indent=2)
        logger.info(f"Iteration {iteration}: Score = {s:.2f}")
        with open(log_dir / f"data_iteration_{iteration}.json", "w") as f:
            json.dump({"iteration": iteration, "joint_reward": s, "agent_rewards": r, "inventory": self.inventory, "failures": self.resource_failures}, f, indent=2)
    def get_final_summary(self):
        s, _ = self._calculate_makespan_and_flow()
        total_patients = len(self.patients)
        converged_patients = 0
        failed_list = []
        for pid, p in self.patients.items():
            completed_steps = len(self.patient_states[pid]["scheduled_steps"])
            required_steps = len(p["pathway"])
            resource_errors = self.patient_states[pid].get("resource_failures", [])
            status = "OK"
            if completed_steps != required_steps:
                status = f"Incomplete ({completed_steps}/{required_steps} steps)"
            elif resource_errors:
                steps_failed = [str(x['step']) for x in resource_errors]
                status = f"RESOURCE FAILED (Steps: {', '.join(steps_failed)})"
            if status == "OK": converged_patients += 1
            else: failed_list.append(f"{pid}: {status}")
        return {"status": "complete" if not failed_list else "partial_convergence", "joint_reward": s, "convergence_report": {"total_patients": total_patients, "converged_count": converged_patients, "resource_failures": self.resource_failures, "final_inventory": self.inventory, "failed_patients": failed_list}, "schedule": self.schedule}
