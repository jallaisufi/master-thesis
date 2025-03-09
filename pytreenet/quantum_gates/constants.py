from pytreenet.time_evolution.bug import BUGConfig
import numpy as np

TIME_STEP_SIZE = "time_step_size"
CONFIG = "config"
PHASE_SHIFT = "phase_shift"
FINAL_TIME = "final_time"

GATE_CONFIGS = {
    "X": {
        TIME_STEP_SIZE: 0.01,
        CONFIG: BUGConfig(record_bond_dim=True),
        FINAL_TIME: np.pi / 2,
    },
    "Y": {
        TIME_STEP_SIZE: 0.01,
        CONFIG: BUGConfig(record_bond_dim=True),
        FINAL_TIME: np.pi / 2,
    },
    "Z": {
        TIME_STEP_SIZE: 0.01,
        CONFIG: BUGConfig(record_bond_dim=True),
        FINAL_TIME: np.pi / 2,
    },
    "H": {
        TIME_STEP_SIZE: 0.01,
        CONFIG: BUGConfig(record_bond_dim=True),
        FINAL_TIME: np.pi / 2,
    },  # final time 3.2?
    "CNOT": {
        TIME_STEP_SIZE: 0.01,
        CONFIG: BUGConfig(record_bond_dim=True),
        FINAL_TIME: np.pi / 4,
    },
    "SWAP": {
        TIME_STEP_SIZE: 0.01,
        CONFIG: BUGConfig(record_bond_dim=True),
        FINAL_TIME: None,
    },
    "P_phi": {
        TIME_STEP_SIZE: 0.01,
        CONFIG: BUGConfig(record_bond_dim=True),
        FINAL_TIME: np.pi / 2,  # ??
    },
    "CP_phi": {
        TIME_STEP_SIZE: 0.01,
        CONFIG: BUGConfig(record_bond_dim=True),
        FINAL_TIME: None,
    },
}
