import logging
import math
import json
import random
from typing import Dict, List, Any, Tuple, Optional, Mapping
from datetime import datetime
from envs.abstract_environment import AbstractEnvironment
from src.utils import get_run_timestamp, build_log_dir, get_tag_model_subdir
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

        # Generate Data
        self.agents_map, self.patients = self._generate_flexible_jobshop_data()
        self.agent_names = list(self.agents_map.keys())
        self.prompts = HospitalPrompts(self, self.full_config)
        
        # State
        self.schedule: Dict[str, Dict[int, List[str]]] = {a: {} for a in self.agent_names}
        self.patient_states: Dict[str, Dict] = {p: {"scheduled_steps": {}} for p in self.patients}

        self.max_time_horizon = 168
        self.transfer_penalty_hours = 4
        self.theoretical_max_score = len(self.patients) * 1000.0 
        self.joint_reward_history = []
        self.agent_rewards_history = {a: [] for a in self.agent_names}

    def compute_max_joint_reward(self) -> float:
        return self.theoretical_max_score

    def _generate_flexible_jobshop_data(self) -> Tuple[Dict, Dict]:
        hospitals = ["General_Hospital", "St_Marys_Center"]
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
                
        patients = {}
        pathways = [
            [("Triage", 1), ("Radiology", 1), ("Surgery", 4), ("Ward", 48)],
            [("Triage", 1), ("Ward", 24)],
            [("Triage", 1), ("Radiology", 2), ("Ward", 12)]
        ]
        
        for i in range(self.env_config.get("num_patients", 8)):
            pid = f"Patient_{i}"
            steps = [{"step_index": idx, "service": s, "duration": d} 
                     for idx, (s, d) in enumerate(random.choice(pathways))]
            patients[pid] = {
                "id": pid, "arrival_time": random.randint(0, 24), "pathway": steps
            }
        return agents_map, patients

    def apply_state_updates(self, state_updates: Dict[str, Any]) -> None:
        if "schedule" in state_updates:
            for agent_id, acts in state_updates["schedule"].items():
                if not isinstance(acts, list): acts = [acts]
                for act in acts:
                    self._process_schedule_request(agent_id, act)

    def _process_schedule_request(self, agent_id, action):
        p_id, step_idx, start = action.get("patient_id"), action.get("step_index"), action.get("start_time")
        if p_id not in self.patients: return

        patient = self.patients[p_id]
        if step_idx >= len(patient["pathway"]): return
        
        # Validation Logic
        target_step = patient["pathway"][step_idx]
        agent_info = self.agents_map[agent_id]
        
        # Service Match
        if agent_info["service"] != target_step["service"]: return
        
        # Precedence
        min_start = patient["arrival_time"]
        if step_idx > 0:
            prev = self.patient_states[p_id]["scheduled_steps"].get(step_idx - 1)
            if not prev: return
            
            prev_h = self.agents_map[prev["agent"]]["hospital"]
            curr_h = agent_info["hospital"]
            penalty = self.transfer_penalty_hours if prev_h != curr_h else 0
            min_start = max(min_start, prev["end_time"] + penalty)
            
        if start < min_start: return
        
        # Capacity
        dur = target_step["duration"]
        for t in range(start, start + dur):
            if t >= self.max_time_horizon: return
            if len(self.schedule[agent_id].get(t, [])) >= agent_info["capacity"]: return
            
        # Commit
        for t in range(start, start + dur):
            if t not in self.schedule[agent_id]: self.schedule[agent_id][t] = []
            self.schedule[agent_id][t].append(p_id)
            
        self.patient_states[p_id]["scheduled_steps"][step_idx] = {
            "agent": agent_id, "start_time": start, "end_time": start + dur
        }

    def _calculate_makespan_and_flow(self):
        total_flow = 0.0
        agent_rewards = {a: 0.0 for a in self.agent_names}
        penalty = 500.0
        
        for pid, p in self.patients.items():
            sched = self.patient_states[pid]["scheduled_steps"]
            path = p["pathway"]
            if len(sched) == len(path):
                flow = sched[len(path)-1]["end_time"] - p["arrival_time"]
                total_flow += flow
                for info in sched.values():
                    agent_rewards[info["agent"]] -= (flow / len(path))
            else:
                total_flow += (len(path) - len(sched)) * penalty

        score = self.theoretical_max_score - total_flow
        base = self.theoretical_max_score / len(self.agent_names)
        for a in agent_rewards: agent_rewards[a] += base
        return score, agent_rewards

    def joint_reward(self, actions):
        s, _ = self._calculate_makespan_and_flow()
        return s

    def agent_reward(self, agent_name, action):
        _, r = self._calculate_makespan_and_flow()
        return r.get(agent_name, 0.0)

    def build_agent_context(self, agent_name, phase, iteration, **kwargs):
        my_info = self.agents_map[agent_name]
        
        # Summary
        summary = {}
        for t in range(0, 48, 4):
            load = sum(len(self.schedule[agent_name].get(t+i, [])) for i in range(4)) / 4.0
            summary[f"H{t}"] = f"{load:.1f}/{my_info['capacity']}"
            
        # Queue
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
                    
        return {
            "agent_name": agent_name, "phase": phase, "iteration": iteration,
            "dept_info": {"capacity": my_info["capacity"], "service": my_info["service"], "schedule_summary": summary},
            "job_queue": queue
        }

    def done(self, iteration):
        if iteration >= self.env_config.get("max_iterations", 10): return True
        return all(len(self.patient_states[p]["scheduled_steps"]) == len(self.patients[p]["pathway"]) for p in self.patients)

    def get_network_context(self): return "Flexible Job Shop Network."
    
    async def async_init(self): await super().async_init()

    def log_iteration(self, iteration):
        s, r = self._calculate_makespan_and_flow()
        self.joint_reward_history.append(s)
        for a, v in r.items(): self.agent_rewards_history[a].append(v)

        tag = get_tag_model_subdir(self.full_config)
        log_dir = build_log_dir(self.__class__.__name__, tag, self.current_seed, self.run_timestamp)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # --- SAVE PATIENTS ---
        if iteration == 1:
            with open(log_dir / "patients.json", "w") as f:
                json.dump(self.patients, f, indent=2)
            logger.info(f"Generated patients saved to {log_dir}/patients.json")
        # ---------------------

        logger.info(f"Iteration {iteration}: Joint Score = {s:.2f}")
        with open(log_dir / f"data_iteration_{iteration}.json", "w") as f:
            json.dump({"iteration": iteration, "joint_reward": s, "agent_rewards": r}, f, indent=2)

    def get_final_summary(self):
        s, _ = self._calculate_makespan_and_flow()
        
        # --- IMPROVED CONVERGENCE REPORTING ---
        total_patients = len(self.patients)
        converged_patients = 0
        failed_list = []
        
        for pid, p in self.patients.items():
            completed_steps = len(self.patient_states[pid]["scheduled_steps"])
            required_steps = len(p["pathway"])
            
            if completed_steps == required_steps:
                converged_patients += 1
            else:
                failed_list.append(f"{pid} ({completed_steps}/{required_steps} steps)")
        
        status_msg = "complete" if not failed_list else "partial_convergence"
        
        return {
            "status": status_msg,
            "joint_reward": s,
            "convergence_report": {
                "total_patients": total_patients,
                "converged_count": converged_patients,
                "success_rate": f"{(converged_patients/total_patients)*100:.1f}%",
                "failed_patients": failed_list
            },
            "schedule": self.schedule
        }

    def get_serializable_state(self):
        return {"schedule": self.schedule, "patient_states": self.patient_states, "agents": self.agents_map}