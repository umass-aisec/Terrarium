import logging
import math
import json
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Mapping
from datetime import datetime
from envs.abstract_environment import AbstractEnvironment
from src.utils import (
    get_run_timestamp, 
    build_log_dir, 
    extract_model_info, 
    get_tag_model_subdir
)
from .hospital_prompts import HospitalPrompts

logger = logging.getLogger(__name__)

class HospitalEnvironment(AbstractEnvironment):
    def __init__(self, communication_protocol, config, tool_logger):
        self.full_config = config
        self.env_config = config["environment"]
        self.simulation_config = config["simulation"]
        self.current_seed = int(self.simulation_config["seed"])
        self.run_timestamp = get_run_timestamp(self.full_config)
        
        self.tool_logger = tool_logger
        self.communication_protocol = communication_protocol
        self.communication_protocol.environment = self

        # ---------------------------------------------------------------------
        # 1. Load Data & Initialize Agents
        # ---------------------------------------------------------------------
        data_path = self.env_config.get("data_path", "data/hospital/outpatient_merged.csv")
        self.hospitals, self.patients = self._load_data(data_path)
        
        self.agent_names = list(self.hospitals.keys())
        self.prompts = HospitalPrompts(self, self.full_config)
        
        # ---------------------------------------------------------------------
        # 2. State Tracking
        # ---------------------------------------------------------------------
        self.assignment: Dict[str, str] = {
            p_id: p_data['location'] for p_id, p_data in self.patients.items()
        }
        
        self.joint_reward_history: List[float] = []
        self.agent_rewards_history: Dict[str, List[float]] = {agent: [] for agent in self.agent_names}
        
        self.max_joint_reward = len(self.patients) * 10.0 if self.patients else 100.0

    # --- ADDED THIS METHOD TO FIX THE VALUE ERROR ---
    def get_network_context(self) -> str:
        """Required by AbstractEnvironment to seed the blackboards."""
        return (
            "This is a hospital coordination network. "
            "Hospitals must coordinate to transfer patients to facilities with the "
            "correct specialties (Cardiology, Neurology, Orthopedics, etc.) while "
            "balancing capacity limits and minimizing transport costs."
        )

    # --- CORRECTED ASYNC INIT ---
    async def async_init(self):
        """Initialize the communication network."""
        await super().async_init()

    def _load_data(self, data_path: str) -> Tuple[Dict, Dict]:
        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            logger.error(f"Data file not found at {data_path}. Initializing empty environment.")
            return {}, {}

        limit = self.env_config.get("num_agents", 5)
        
        if 'THCIC_ID' not in df.columns:
            logger.warning("THCIC_ID column missing. Using mock IDs.")
            df['THCIC_ID'] = [f"H{i}" for i in range(len(df))]

        unique_ids = df['THCIC_ID'].unique()[:limit]
        hospitals = {}
        
        for h_id in unique_ids:
            row = df[df['THCIC_ID'] == h_id].iloc[0]
            
            specialties = []
            if str(row.get('FAC_CARDIOVASCULAR_IND', '')).strip() == '1': specialties.append("Cardiology")
            if str(row.get('FAC_NEUROLOGICAL_IND', '')).strip() == '1': specialties.append("Neurology") 
            if str(row.get('FAC_ORTHOPEDIC_IND', '')).strip() == '1': specialties.append("Orthopedics")
            if str(row.get('FAC_PEDS_IND', '')).strip() == '1': specialties.append("Pediatrics")
            if str(row.get('FAC_ONCOLOGY_IND', '')).strip() == '1': specialties.append("Oncology")
            if not specialties: specialties.append("General")

            loc_proxy = float(row.get('PAT_COUNTY', 0)) 

            hospitals[str(h_id)] = {
                "id": str(h_id),
                "name": str(row.get('PROVIDER_NAME', f"Hospital_{h_id}")),
                "location": (loc_proxy, loc_proxy),
                "capacity": self.env_config.get("default_capacity", 10),
                "specialties": specialties,
            }

        df_filtered = df[df['THCIC_ID'].isin(unique_ids)]
        patients = {}
        for idx, row in df_filtered.iterrows():
            p_id = str(row.get('RECORD_ID', f"P{idx}"))
            diag_code = str(row.get('PRINC_DIAG_CODE', ''))
            condition = self._map_diagnosis_to_specialty(diag_code)
            
            patients[p_id] = {
                "id": p_id,
                "location": str(row['THCIC_ID']),
                "condition": condition,
                "diagnosis_code": diag_code,
                "severity": str(row.get('PAT_STATUS', '01')),
            }
            
        return hospitals, patients

    def _map_diagnosis_to_specialty(self, code: str) -> str:
        code = code.upper()
        if code.startswith('I'): return "Cardiology"
        if code.startswith('G'): return "Neurology"
        if code.startswith('M'): return "Orthopedics"
        if code.startswith('P'): return "Pediatrics"
        if code.startswith('C'): return "Oncology"
        return "General"

    def build_agent_context(self, agent_name: str, phase: str, iteration: int, **kwargs) -> Dict[str, Any]:
        my_hospital = self.hospitals[agent_name]
        my_current_patients = {pid: p for pid, p in self.patients.items() if p['location'] == agent_name}
        current_decisions = {pid: self.assignment.get(pid, agent_name) for pid in my_current_patients}

        return {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "hospital_stats": {
                "name": my_hospital["name"],
                "specialties": my_hospital["specialties"],
                "capacity": my_hospital["capacity"],
                "location": my_hospital["location"]
            },
            "my_patients": my_current_patients,
            "current_assignments": current_decisions,
            "known_neighbors": list(self.hospitals.keys()), 
        }

    def _rewards(self, assignment: Dict[str, str]) -> Tuple[float, Dict[str, float]]:
        total_score = 0.0
        local_rewards = {h: 0.0 for h in self.hospitals}
        temp_loads = {h: 0 for h in self.hospitals}
        
        for p_id, target_h_id in assignment.items():
            if target_h_id not in self.hospitals: continue
            
            patient = self.patients[p_id]
            target_hospital = self.hospitals[target_h_id]
            source_hospital = self.hospitals[patient['location']]
            
            p_score = 0.0
            
            # Specialty Match
            if patient['condition'] in target_hospital['specialties']:
                p_score += 10.0
            elif "General" in target_hospital['specialties']:
                p_score += 2.0 
            else:
                p_score -= 5.0

            # Transport Cost
            if patient['location'] != target_h_id:
                loc_a = source_hospital['location']
                loc_b = target_hospital['location']
                dist = math.sqrt((loc_a[0]-loc_b[0])**2 + (loc_a[1]-loc_b[1])**2)
                p_score -= (dist * 0.1)
            
            total_score += p_score
            
            if patient['location'] in local_rewards:
                local_rewards[patient['location']] += p_score
                
            temp_loads[target_h_id] += 1

        for h_id, load in temp_loads.items():
            capacity = self.hospitals[h_id]['capacity']
            if load > capacity:
                penalty = (load - capacity) * 20.0
                total_score -= penalty
                if h_id in local_rewards:
                    local_rewards[h_id] -= penalty

        return total_score, local_rewards

    def joint_reward(self, actions: Mapping[str, Any]) -> float:
        score, _ = self._rewards(self.assignment)
        return score

    def agent_reward(self, agent_name: str, action: Any) -> float:
        _, local_rewards = self._rewards(self.assignment)
        return local_rewards.get(agent_name, 0.0)

    def log_iteration(self, iteration: int) -> None:
        joint_reward, agent_rewards = self._rewards(self.assignment)
        
        self.joint_reward_history.append(joint_reward)
        for agent, reward in agent_rewards.items():
            if agent in self.agent_rewards_history:
                self.agent_rewards_history[agent].append(reward)

        tag_model = get_tag_model_subdir(self.full_config)
        log_dir = build_log_dir(self.__class__.__name__, tag_model, self.current_seed, self.run_timestamp)
        log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Iteration {iteration}: Joint Reward = {joint_reward:.2f}")

        score_entry = {
            "environment": self.__class__.__name__,
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "joint_reward": joint_reward,
            "joint_reward_ratio": joint_reward / self.max_joint_reward if self.max_joint_reward != 0 else 0,
            "max_joint_reward": self.max_joint_reward,
            "agent_rewards": agent_rewards,
            "average_agent_reward": sum(agent_rewards.values()) / len(agent_rewards) if agent_rewards else 0,
            "model_info": extract_model_info(self.full_config),
            "full_config": self.full_config,
            "total_agents": len(agent_rewards),
            "total_patients": len(self.patients)
        }

        data_file = log_dir / f"data_iteration_{iteration}.json"
        with open(data_file, "w") as f:
            json.dump(score_entry, f, indent=2, ensure_ascii=False)

    def done(self, iteration: int) -> bool:
        max_iters = self.env_config.get("max_iterations", 5)
        return iteration >= max_iters
    
    def compute_max_joint_reward(self):
        return self.max_joint_reward

    def get_serializable_state(self) -> Dict[str, Any]:
        return {
            "assignment": self.assignment.copy(),
            "agents": list(self.hospitals.keys()),
            "patients": self.patients.copy() 
        }

    def apply_state_updates(self, state_updates: Dict[str, Any]) -> None:
        if "transfers" in state_updates:
            self.assignment.update(state_updates["transfers"])