from abc import ABC, abstractmethod
import numpy as np
from pytreenet.operators import pauli_matrices
from pytreenet.operators.hamiltonian import Hamiltonian
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.time_evolution.bug import BUG, BUGConfig
from pytreenet.ttno.ttno_class import TreeTensorNetworkOperator
from matplotlib.pyplot import subplots, show


class QuantumGate(ABC):
    """
    Abstract base class for quantum gates.
    """

    bug_instance = None

    @abstractmethod
    def apply_gate(self, ttns, *args, **kwargs):
        """
        Apply the gate to the given quantum state.
        Args:
            ttns: TreeTensorNetworkState or equivalent quantum state representation.
            args, kwargs: Additional arguments (e.g., node IDs).
        """
        pass

    @abstractmethod
    def plot(self, qubit0_id=None, qubit1_id=None):
        """
        Abstract method to plot the total local magnetization of a gate.

        Args:
            qubit0_id (str, optional): The first qubit's ID (for two-qubit gates).
            qubit1_id (str, optional): The second qubit's ID (for two-qubit gates).
        """
        pass


# TODO: Put dimenstion (2) into a variable
identity = np.eye(2)
X, Y, Z = pauli_matrices()


def generate_z_operator(node_id):
    sq_op_id = "SQ_Operator"
    sq_z_op = {sq_op_id: TensorProduct({node_id: Z})}
    return sq_z_op


class XGate(QuantumGate):
    def apply_gate(self, ttns, node_id, time_step_size, final_time):
        """
        Apply the Pauli-X gate to a single qubit.
        """
        term = TensorProduct({node_id: "X"})

        conv_dict = {"I2": identity, "X": X}
        hamiltonian = Hamiltonian(term, conversion_dictionary=conv_dict)
        ttno = TreeTensorNetworkOperator.from_hamiltonian(hamiltonian, ttns)

        z_operator = generate_z_operator(node_id)

        self.bug_instance = BUG(ttns, ttno, time_step_size, final_time, z_operator)
        self.bug_instance.run()

        ttns = self.bug_instance.state

        return ttns

    def plot(self, qubit0_id=None, qubit1_id=None):
        """
        Plot the total local magnetization of the X-Gate
        """
        print("Sanity Check: ", self.bug_instance.results_real())
        sq_op_id = "SQ_Operator"
        times = self.bug_instance.times()
        sq_results = self.bug_instance.operator_result(sq_op_id, realise=True)

        fig, axs = subplots(1, 2, figsize=(10, 5))
        axs[0].plot(times, sq_results, label="BUG")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Expectation Value")
        axs[0].set_title("Expectation Value of Z-Operator")
        axs[0].legend()

        show()


class YGate(QuantumGate):
    def apply_gate(self, ttns, node_id, time_step_size, final_time):
        """
        Apply the Pauli-Y gate to a single qubit.
        """
        term = TensorProduct({node_id: "Y"})

        conv_dict = {"I2": identity, "Y": Y}
        hamiltonian = Hamiltonian(term, conversion_dictionary=conv_dict)
        ttno = TreeTensorNetworkOperator.from_hamiltonian(hamiltonian, ttns)

        z_operator = generate_z_operator(node_id)

        self.bug_instance = BUG(ttns, ttno, time_step_size, final_time, z_operator)
        self.bug_instance.run()

        ttns = self.bug_instance.state

        return ttns

    def plot(self, qubit0_id=None, qubit1_id=None):
        """
        Plot the total local magnetization of the X-Gate
        """
        print("Sanity Check: ", self.bug_instance.results_real())
        sq_op_id = "SQ_Operator"
        times = self.bug_instance.times()
        sq_results = self.bug_instance.operator_result(sq_op_id, realise=True)

        fig, axs = subplots(1, 2, figsize=(10, 5))
        axs[0].plot(times, sq_results, label="BUG")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Expectation Value")
        axs[0].set_title("Expectation Value of Z-Operator")
        axs[0].legend()

        show()


class ZGate(QuantumGate):
    def apply_gate(self, ttns, node_id, time_step_size, final_time):
        """
        Apply the Pauli-Z gate to a single qubit.
        """
        term = TensorProduct({node_id: "Z"})

        conv_dict = {"I2": identity, "Z": Z}
        hamiltonian = Hamiltonian(term, conversion_dictionary=conv_dict)
        ttno = TreeTensorNetworkOperator.from_hamiltonian(hamiltonian, ttns)

        z_operator = generate_z_operator(node_id)

        self.bug_instance = BUG(ttns, ttno, time_step_size, final_time, z_operator)
        self.bug_instance.run()

        ttns = self.bug_instance.state

        return ttns

    def plot(self, qubit0_id=None, qubit1_id=None):
        """
        Plot the total local magnetization of the X-Gate
        """
        print("Sanity Check: ", self.bug_instance.results_real())
        sq_op_id = "SQ_Operator"
        times = self.bug_instance.times()
        sq_results = self.bug_instance.operator_result(sq_op_id, realise=True)

        fig, axs = subplots(1, 2, figsize=(10, 5))
        axs[0].plot(times, sq_results, label="BUG")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Expectation Value")
        axs[0].set_title("Expectation Value of Z-Operator")
        axs[0].legend()

        show()


class HadamardGate(QuantumGate):
    def apply_gate(self, ttns, node_id, time_step_size, final_time):
        """
        Apply the Hadamard gate to a single qubit.
        """

        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        term = TensorProduct({node_id: "H"})

        conv_dict = {"I2": identity, "H": H}
        hamiltonian = Hamiltonian(term, conversion_dictionary=conv_dict)
        ttno = TreeTensorNetworkOperator.from_hamiltonian(hamiltonian, ttns)

        z_operator = generate_z_operator(node_id)

        self.bug_instance = BUG(ttns, ttno, time_step_size, final_time, z_operator)
        self.bug_instance.run()

        ttns = self.bug_instance.state

        return ttns

    def plot(self, qubit0_id=None, qubit1_id=None):
        """
        Plot the total local magnetization of the X-Gate
        """
        print("Sanity Check: ", self.bug_instance.results_real())
        sq_op_id = "SQ_Operator"
        times = self.bug_instance.times()
        sq_results = self.bug_instance.operator_result(sq_op_id, realise=True)

        fig, axs = subplots(1, 2, figsize=(10, 5))
        axs[0].plot(times, sq_results, label="BUG")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Expectation Value")
        axs[0].set_title("Expectation Value of Z-Operator")
        axs[0].legend()

        show()


class CNOTGate(QuantumGate):
    def apply_gate(
        self, ttns, control_id, target_id, time_step_size, config, final_time
    ):
        """
        Apply the CNOT gate to one control qubit and one or more target qubits.

        Args:
            ttns: Tree Tensor Network State.
            control_id (str): ID of the control qubit.
            target_ids (str): IDs of the target qubits.
        """
        term = TensorProduct({control_id: "q0_op", target_id: "q1_op"})
        control_op = np.eye(2) - pauli_matrices()[2]
        target_op = np.eye(2) - pauli_matrices()[0]
        conv_dict = {"I2": identity, "q0_op": control_op, "q1_op": target_op}
        hamiltonian = Hamiltonian(term, conversion_dictionary=conv_dict)
        ttno = TreeTensorNetworkOperator.from_hamiltonian(hamiltonian, ttns)

        operators = {
            control_id: TensorProduct({control_id: pauli_matrices()[2]}),
            target_id: TensorProduct({target_id: pauli_matrices()[2]}),
        }
        self.bug_instance = BUG(
            ttns, ttno, time_step_size, final_time, operators, config=config
        )
        self.bug_instance.run()

        ttns = self.bug_instance.state

        return ttns

    def plot(self, qubit0_id, qubit1_id):
        """
        Plot the total local magnetization of the X-Gate
        """
        print("Sanity Check: ", self.bug_instance.results_real())
        times = self.bug_instance.times()
        sq_results0 = self.bug_instance.operator_result(qubit0_id, realise=True)
        sq_results1 = self.bug_instance.operator_result(qubit1_id, realise=True)

        fig, axs = subplots(2,2, figsize=(10, 10))
        axs[0,0].plot(times, sq_results0, label="BUG")
        axs[0,0].set_xlabel("Time")
        axs[0,0].set_ylabel("Expectation Value")
        axs[0,0].set_title("Expectation Value of Z-Operator on Qubit 0")
        axs[0,0].legend()

        axs[1,0].plot(times, sq_results1, label="BUG")
        axs[1,0].set_xlabel("Time")
        axs[1,0].set_ylabel("Expectation Value")
        axs[1,0].set_title("Expectation Value of Z-Operator on Qubit 1")
        axs[1,0].legend()

        show()


class SWAPGate(QuantumGate):
    def apply_gate(
        self, ttns, qubit1_id, qubit2_id, time_step_size, config, final_time
    ):
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

        self.bug_instance = BUG(
            ttns, ttno, time_step_size, final_time, operators, config=config
        )
        self.bug_instance.run()

        ttns = self.bug_instance.state

        return ttns
    
    def plot(self, qubit0_id, qubit1_id):
        """
        Plot the total local magnetization of the X-Gate
        """
        print("Sanity Check: ", self.bug_instance.results_real())
        times = self.bug_instance.times()
        sq_results0 = self.bug_instance.operator_result(qubit0_id, realise=True)
        sq_results1 = self.bug_instance.operator_result(qubit1_id, realise=True)

        fig, axs = subplots(2,2, figsize=(10, 10))
        axs[0,0].plot(times, sq_results0, label="BUG")
        axs[0,0].set_xlabel("Time")
        axs[0,0].set_ylabel("Expectation Value")
        axs[0,0].set_title("Expectation Value of Z-Operator on Qubit 0")
        axs[0,0].legend()

        axs[1,0].plot(times, sq_results1, label="BUG")
        axs[1,0].set_xlabel("Time")
        axs[1,0].set_ylabel("Expectation Value")
        axs[1,0].set_title("Expectation Value of Z-Operator on Qubit 1")
        axs[1,0].legend()

        show()


class PhaseShiftGate(QuantumGate):
    def apply_gate(self, ttns, node_id, phase_shift, time_step_size, final_time):
        """
        Apply the Phase Shift Gate P(ϕ) to a single qubit, where ϕ is passed as an argument.
        """

        P_phi = np.array([[1, 0], [0, np.exp(1j * phase_shift)]], dtype=complex)

        term = TensorProduct({node_id: "P_phi"})
        # final_time = 3.2

        conv_dict = {"I2": identity, "P_phi": P_phi}
        hamiltonian = Hamiltonian(term, conversion_dictionary=conv_dict)
        ttno = TreeTensorNetworkOperator.from_hamiltonian(hamiltonian, ttns)

        z_operator = generate_z_operator(node_id)

        self.bug_instance = BUG(ttns, ttno, time_step_size, final_time, z_operator)
        self.bug_instance.run()

        ttns = self.bug_instance.state

        return ttns
    
    def plot(self, qubit0_id=None, qubit1_id=None):
        """
        Plot the total local magnetization of the X-Gate
        """
        print("Sanity Check: ", self.bug_instance.results_real())
        sq_op_id = "SQ_Operator"
        times = self.bug_instance.times()
        sq_results = self.bug_instance.operator_result(sq_op_id, realise=True)

        fig, axs = subplots(1, 2, figsize=(10, 5))
        axs[0].plot(times, sq_results, label="BUG")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Expectation Value")
        axs[0].set_title("Expectation Value of Z-Operator")
        axs[0].legend()

        show()


class ControlledPhaseGate(QuantumGate):
    def apply_gate(
        self, ttns, control_id, target_id, phase_shift, time_step_size, final_time
    ):
        """
        Apply the Controlled Phase Shift Gate CP(ϕ) to a two-qubit system.
        - control_node_id: The control qubit
        - target_node_id: The qubit on which the phase shift is applied
        - phi: The phase angle to apply
        """

        CP_phi = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, np.exp(1j * phase_shift)],
            ],
            dtype=complex,
        )

        term = TensorProduct({control_id: "I2", target_id: "CP_phi"})
        # final_time = 3.2

        conv_dict = {"I2": np.eye(2, dtype=complex), "CP_phi": CP_phi}
        hamiltonian = Hamiltonian(term, conversion_dictionary=conv_dict)
        ttno = TreeTensorNetworkOperator.from_hamiltonian(hamiltonian, ttns)

        z_operator = generate_z_operator(target_id)

        self.bug_instance = BUG(ttns, ttno, time_step_size, final_time, z_operator)
        self.bug_instance.run()

        ttns = self.bug_instance.state

        return ttns
    
    def plot(self, qubit0_id, qubit1_id):
        """
        Plot the total local magnetization of the X-Gate
        """
        print("Sanity Check: ", self.bug_instance.results_real())
        times = self.bug_instance.times()
        sq_results0 = self.bug_instance.operator_result(qubit0_id, realise=True)
        sq_results1 = self.bug_instance.operator_result(qubit1_id, realise=True)

        fig, axs = subplots(2,2, figsize=(10, 10))
        axs[0,0].plot(times, sq_results0, label="BUG")
        axs[0,0].set_xlabel("Time")
        axs[0,0].set_ylabel("Expectation Value")
        axs[0,0].set_title("Expectation Value of Z-Operator on Qubit 0")
        axs[0,0].legend()

        axs[1,0].plot(times, sq_results1, label="BUG")
        axs[1,0].set_xlabel("Time")
        axs[1,0].set_ylabel("Expectation Value")
        axs[1,0].set_title("Expectation Value of Z-Operator on Qubit 1")
        axs[1,0].legend()

        show()
