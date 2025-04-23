from pytreenet.quantum_gates.QuantumGate import CNOTGate, ControlledPhaseGate, HadamardGate, PhaseShiftGate, SWAPGate, XGate, YGate, ZGate
from pytreenet.time_evolution.bug import BUGConfig
import numpy as np

TIME_STEP_SIZE = "time_step_size"
CONFIG = "config"
PHASE_SHIFT = "phase_shift"
FINAL_TIME = "final_time"
CLASS = "class"

GATE_CONFIGS = {
    "X": {
        TIME_STEP_SIZE: 0.01,
        CONFIG: BUGConfig(record_bond_dim=True),
        FINAL_TIME: np.pi / 2,
        CLASS: XGate()
    },
    "Y": {
        TIME_STEP_SIZE: 0.01,
        CONFIG: BUGConfig(record_bond_dim=True),
        FINAL_TIME: np.pi / 2,
        CLASS: YGate()
    },
    "Z": {
        TIME_STEP_SIZE: 0.01,
        CONFIG: BUGConfig(record_bond_dim=True),
        FINAL_TIME: np.pi / 2,
        CLASS: ZGate()
    },
    "H": {
        TIME_STEP_SIZE: 0.01,
        CONFIG: BUGConfig(record_bond_dim=True),
        FINAL_TIME: np.pi / 2,
        CLASS: HadamardGate()
    },  # final time 3.2?
    "CNOT": {
        TIME_STEP_SIZE: 0.01,
        CONFIG: BUGConfig(record_bond_dim=True),
        FINAL_TIME: np.pi / 4,
        CLASS: CNOTGate()
    },
    "SWAP": {
        TIME_STEP_SIZE: 0.01,
        CONFIG: BUGConfig(record_bond_dim=True),
        FINAL_TIME: np.pi / 4,
        CLASS: SWAPGate()
    },
    "P_phi": {
        TIME_STEP_SIZE: 0.01,
        CONFIG: BUGConfig(record_bond_dim=True),
        FINAL_TIME: np.pi / 2,  # ??
        CLASS: PhaseShiftGate()
    },
    "CP_phi": {
        TIME_STEP_SIZE: 0.01,
        CONFIG: BUGConfig(record_bond_dim=True),
        FINAL_TIME: np.pi / 4,
        CLASS: ControlledPhaseGate()
    },
}
