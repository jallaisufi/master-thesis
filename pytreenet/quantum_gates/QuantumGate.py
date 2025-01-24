from abc import ABC, abstractmethod
import numpy as np
from pytreenet.operators import pauli_matrices
from pytreenet.operators.hamiltonian import Hamiltonian
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.time_evolution.bug import BUG
from pytreenet.time_evolution.ttn_time_evolution import TTNTimeEvolutionConfig
from pytreenet.ttno.ttno_class import TreeTensorNetworkOperator


class QuantumGate(ABC):
    """
    Abstract base class for quantum gates.
    """

    @abstractmethod
    def apply_gate(self, ttns, *args, **kwargs):
        """
        Apply the gate to the given quantum state.
        Args:
            ttns: TreeTensorNetworkState or equivalent quantum state representation.
            args, kwargs: Additional arguments (e.g., node IDs).
        """
        pass


time_step_size = 0.01
final_time = 3.2
# TODO: Put dimenstion (2) into a variable
identity = np.eye(2)
X, Y, Z = pauli_matrices()


def generate_z_operator(node_id):
    sq_op_id = "SQ_Operator"
    sq_z_op = {sq_op_id: TensorProduct({node_id: Z})}
    return sq_z_op


class XGate(QuantumGate):
    def apply_gate(self, ttns, node_id):
        """
        Apply the Pauli-X gate to a single qubit.
        """
        term = TensorProduct({node_id: "X"})

        conv_dict = {"I2": identity, "X": X}
        hamiltonian = Hamiltonian(term, conversion_dictionary=conv_dict)
        ttno = TreeTensorNetworkOperator.from_hamiltonian(hamiltonian, ttns)

        z_operator = generate_z_operator(node_id)

        bug = BUG(ttns, ttno, time_step_size, final_time, z_operator)
        bug.run()

        return ttns

class YGate(QuantumGate):
    def apply_gate(self, ttns, node_id):
        """
        Apply the Pauli-Y gate to a single qubit.
        """
        term = TensorProduct({node_id: "Y"})

        conv_dict = {"I2": identity, "Y": Y}
        hamiltonian = Hamiltonian(term, conversion_dictionary=conv_dict)
        ttno = TreeTensorNetworkOperator.from_hamiltonian(hamiltonian, ttns)

        z_operator = generate_z_operator(node_id)

        bug = BUG(ttns, ttno, time_step_size, final_time, z_operator)
        bug.run()

        return ttns

class ZGate(QuantumGate):
    def apply_gate(self, ttns, node_id):
        """
        Apply the Pauli-Z gate to a single qubit.
        """
        term = TensorProduct({node_id: "Z"})

        conv_dict = {"I2": identity, "Z": Z}
        hamiltonian = Hamiltonian(term, conversion_dictionary=conv_dict)
        ttno = TreeTensorNetworkOperator.from_hamiltonian(hamiltonian, ttns)

        z_operator = generate_z_operator(node_id)

        bug = BUG(ttns, ttno, time_step_size, final_time, z_operator)
        bug.run()

        return ttns

class HadamardGate(QuantumGate):
    def apply_gate(self, ttns, node_id):
        """
        Apply the Hadamard gate to a single qubit.
        """

        H = (1 / np.sqrt(2)) * (X + Z)
        term = TensorProduct({node_id: "H"})

        conv_dict = {"I2": identity, "H": H}
        hamiltonian = Hamiltonian(term, conversion_dictionary=conv_dict)
        ttno = TreeTensorNetworkOperator.from_hamiltonian(hamiltonian, ttns)

        z_operator = generate_z_operator(node_id)

        bug = BUG(ttns, ttno, time_step_size, final_time, z_operator)
        bug.run()

        return ttns

class CNOTGate(QuantumGate):
    def apply_gate(self, ttns, control_id, *target_ids):
        """
        Apply the CNOT gate to one control qubit and one or more target qubits.

        Args:
            ttns: Tree Tensor Network State.
            control_id (str): ID of the control qubit.
            target_ids (str): IDs of the target qubits.
        """
        terms = []
        conv_dict = {"I2": identity}
        operators = {control_id: TensorProduct({control_id: Z})}

        control_op = identity - Z
        conv_dict["control_op"] = control_op

        for idx, target_id in enumerate(target_ids):
            target_op = identity - X
            conv_dict[f"target_op_{idx}"] = target_op

            term = TensorProduct({control_id: "control_op", target_id: f"target_op_{idx}"})
            terms.append(term)

            operators[target_id] = TensorProduct({target_id: Z})

        hamiltonian = Hamiltonian(terms, conversion_dictionary=conv_dict)

        ttno = TreeTensorNetworkOperator.from_hamiltonian(hamiltonian, ttns)

        config = TTNTimeEvolutionConfig(record_bond_dim=True)
        bug = BUG(ttns, ttno, time_step_size, final_time, operators, config=config)

        bug.run()

        return ttns

class SWAPGate(QuantumGate):
    def apply_gate(self, ttns, qubit1_id, qubit2_id):
        """
        Apply the SWAP gate to two qubits.
        """
        terms = [
            TensorProduct({qubit1_id: "X", qubit2_id: "X"}),
            TensorProduct({qubit1_id: "Y", qubit2_id: "Y"}),
            TensorProduct({qubit1_id: "Z", qubit2_id: "Z"}),
            TensorProduct({qubit1_id: "I2", qubit2_id: "I2"}),
        ]
        conv_dict = {"X": X, "Y": Y, "Z": Z, "I2": identity}
        hamiltonian = Hamiltonian(terms, conv_dict)

        operators = {
            qubit1_id: TensorProduct({qubit1_id: Z}),
            qubit2_id: TensorProduct({qubit2_id: Z}),
        }

        ttno = TreeTensorNetworkOperator.from_hamiltonian(hamiltonian, ttns)
        config = TTNTimeEvolutionConfig(record_bond_dim=True)
        final_time = 2 * 3.2
        bug = BUG(ttns, ttno, time_step_size, final_time, operators, config=config)
        bug.run()

        return ttns
