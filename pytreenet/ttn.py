from __future__ import annotations
from typing import Tuple, Dict, List, Callable
from copy import deepcopy

import numpy as np

from .tree_structure import TreeStructure
from .leg_node import LegNode
from .node import Node
from .tensor_util import (tensor_qr_decomposition,
                          contr_truncated_svd_splitting)
from .leg_specification import LegSpecification
from .canonical_form import canonical_form
from .tree_contraction import (completely_contract_tree,
                               contract_two_ttn)

class TreeTensorNetwork(TreeStructure):
    """
    A tree tensor network (TTN) a tree, where each node contains a tensor,
    that is part of the network. Here a tree tensor network is a dictionary
    _nodes of tensor nodes with their identifiers as keys.

    General structure and parts of the codes are from treelib.tree

    Attributes
    -------
    _nodes: dict[str, TensorNode] mapping node ids (str) to TensorNode objects
    _root_id: str identifier for root node of TTN
    """

    def __init__(self):
        """
        Initiates a new TreeTensorNetwork or a deep or shallow copy of a
        different one.
        """
        super().__init__()
        self._tensors = {}

    @property
    def tensors(self):
        """
        A dict[str, np.ndarray] mapping the tensor tree node identifiers to
        the corresponding tensor data.

        Since during addition of nodes the tensors are not actually transposed,
        this has to be done here. This way whenever tensors are accessed, their
        leg ordering is
            (parent_leg, children_legs, open_legs)
        """
        for node_id in self._tensors:
            self._transpose_tensor(node_id)

        return self._tensors

    def _transpose_tensor(self, node_id: str):
        """
        Since during addition of nodes the tensors are not actually transposed,
        this has to be done when accesing them. 
        This way whenever a tensor is accessed, its leg ordering is
            (parent_leg, children_legs, open_legs)
        """
        node = self.nodes[node_id]
        tensor = self._tensors[node_id]
        transposed_tensor = np.transpose(tensor, node.leg_permutation)
        self._tensors[node_id] = transposed_tensor
        node.reset_permutation()

    def __getitem__(self, key: str) -> Tuple[LegNode, np.ndarray]:
        node = super().__getitem__(key)
        self._transpose_tensor(key)
        tensor = self._tensors[key]
        return (node, tensor)

    def add_root(self, node: Node, tensor: np.ndarray):
        """
        Adds a root tensor node to the TreeTensorNetwork
        """
        leg_node = LegNode.from_node(tensor, node)
        super().add_root(leg_node)

        self.tensors[leg_node.identifier] = tensor

    def add_child_to_parent(self, child: Node, tensor: np.ndarray,
                            child_leg: int, parent_id: str, parent_leg: int):
        """
        Adds a Node to the TreeTensorNetwork which is the child of the Node
        with identifier `parent_id`. The two tensors are contracted along one
        leg. The child via child_leg and the parent via parent_leg
        """
        child_node = LegNode.from_node(tensor, child)

        child_node.open_leg_to_parent(child_leg)
        parent_node = self.nodes[parent_id]
        parent_node.open_leg_to_child(parent_leg)

        super().add_child_to_parent(child_node, parent_id)

        self._tensors[child.identifier] = tensor

    def add_parent_to_root(self, root_leg: int, parent: Node, tensor: np.ndarray,
                           parent_leg: int):
        """
        Adds the node parent as parent to the TreeTensorNetwork's root node. The two
        are contracted. The root via root_leg and the parent via parent_leg.
        The root is updated to be the parent.
        """
        parent_node = LegNode.from_node(tensor, parent)
        former_root_node = self.nodes[self.root_id]

        parent_node.open_leg_to_child(parent_leg)
        former_root_node.open_leg_to_parent(root_leg)

        super().add_parent_to_root(parent_node)

        self.tensors[parent.identifier] = tensor

    def conjugate(self):
        """
        Returns a new TTN that is a conjugated version of the current TTN

        Returns
        -------
        ttn_conj:
            A conjugated copy of the current TTN.

        """
        ttn_conj = deepcopy(self)
        for node_id, tensor in ttn_conj.tensors.items():
            ttn_conj.tensors[node_id] = tensor.conj()
        return ttn_conj

    def absorb_tensor(self, node_id: str, absorbed_tensor: np.ndarray,
                      absorbed_tensors_leg_index: int,
                      this_tensors_leg_index: int):
        """
        Absorbs `absorbed_tensor` into this instance's tensor by contracting
        the absorbed_tensors_leg of the absorbed_tensor and the leg
        this_tensors_leg of this instance's tensor'

        Parameters
        ----------
        absorbed_tensor: np.ndarray
            Tensor to be absorbed.
        absorbed_tensors_leg_index: int
            Leg that is to be contracted with this instance's tensor.
        this_tensors_leg_index:
            The leg of this instance's tensor that is to be contracted with
            the absorbed tensor.
        """
        node_tensor = self.tensors[node_id]
        new_tensor = np.tensordot(node_tensor, absorbed_tensor,
                                      axes=(this_tensors_leg_index, absorbed_tensors_leg_index))

        this_tensors_indices = tuple(range(new_tensor.ndim))
        transpose_perm = (this_tensors_indices[0:this_tensors_leg_index]
                              + (this_tensors_indices[-1], )
                              + this_tensors_indices[this_tensors_leg_index:-1])
        self.tensors[node_id] = new_tensor.transpose(transpose_perm)

    def absorb_tensor_into_neighbour_leg(self, node_id: str, neighbour_id: str,
                                         tensor: np.ndarray, tensor_leg: int):
        """
        Absorb a tensor into a node, by contracting one of the tensor's legs with one of the
        neighbour_legs of the node.

        Args:
            node_id (str): The identifier of the node into which the tensor is absorbed
            neighbour_id (str): The identifier of the neighbour to which the leg points, which
                                 is to be contracted with the tensor
            tensor (np.ndarray): The tensor to be contracted
            tensor_leg (int): The leg of the external tensor which is to be contracted
        """
        assert tensor.ndim == 2
        node = self.nodes[node_id]
        neighbour_leg = node.get_neighbour_leg(neighbour_id)
        self.absorb_tensor(node_id, tensor, tensor_leg, neighbour_leg)

    def absorb_into_open_legs(self, node_id: str, tensor: np.ndarray):
        """
        Absorb a tensor into the open legs of the tensor of a node.
        This tensor will be absorbed into all open legs and it is assumed, the
         leg order of the tensor to be absorbed is the same as the order of
         the open legs of the node.
        The tensor to be absorbed has to have twice as many open legs as the node tensor.
         The first half of the legs is contracted with the tensor node's open legs and
         the second half become the new open legs of the tensor node.

        Args:
            node_id (str): The identifier of the node which is to be contracted with the tensor
            tensor (np.ndarray): The tensor to be contracted.
        """
        node, node_tensor = self[node_id]
        assert tensor.ndim == 2 * node.nopen_legs()

        tensor_legs = list(range(node.nopen_legs()))
        new_tensor = np.tensordot(node_tensor, tensor, axes=(node.open_legs,tensor_legs))
        # The leg ordering was not changed here
        self.tensors[node_id] = new_tensor

    def contract_nodes(self, node_id1: str, node_id2: str, new_identifier: str = ""):
        """
        Contracts two node and inserts a new node with the contracted tensor
        into the ttn.
        Note that one of the nodes will be the parent of the other.
        The resulting leg order is the following:
            `(parent_parent_leg, remaining_parent_children_legs, child_children_legs,
            parent_open_legs, child_open_legs)`
        The resulting node will have the identifier `parent_id + "contr" + child_id`.

        Deletes the originial nodes and tensors from the TTN.

        Args:
            node_id1 (str): Identifier of first tensor
            node_id2 (str): Identifier of second tensor
            new_identifier (str): A potential new identifier. Otherwise defaults to
                `parent_id + 'contr' + child_id`
        """
        parent_id, child_id = self.determine_parentage(node_id1, node_id2)
        if new_identifier == "":
            new_identifier = parent_id + "contr" + child_id

        # Swap child to be the first child -> leg value 1
        parent_node = self.nodes[parent_id]
        parent_node.swap_with_first_child(child_id)

        # Contracting tensors
        parent_tensor = self.tensors[parent_id]
        child_tensor = self.tensors[child_id]
        new_tensor = np.tensordot(parent_tensor, child_tensor,
                                  axes=(parent_node.nparents(),0))

        # Actual tensor leg now have the form
        # (parent_of_parent, remaining_children_of_parent, open_of_parent, 
        # children_of_child, open_of_child)
        parent_nvirt_leg = parent_node.nvirt_legs() - 1
        parent_nlegs = parent_node.nlegs() - 1
        child_node = self.nodes[child_id]
        child_nchild_legs = child_node.nchild_legs()

        # Create proper connectivity (Old nodes are deleted)
        self.combine_nodes(parent_id, child_id, new_identifier=new_identifier)

        # Delete old tensors
        self._tensors.pop(node_id1)
        self._tensors.pop(node_id2)

        # Make Node a LegNode
        new_node = self.nodes[new_identifier]
        new_leg_node = LegNode.from_node(new_tensor, new_node)

        # Building correct permutation (TODO: Move it to LegNode?)
        child_children_legs = [new_leg_node._leg_permutation.pop(parent_nlegs)
                               for i in range(0, child_nchild_legs)]
        new_leg_node._leg_permutation[parent_nvirt_leg:parent_nvirt_leg] = child_children_legs

        # Delete Node, add LegNode and new tensor
        self.nodes.pop(new_identifier)
        self.nodes[new_identifier] = new_leg_node
        self._tensors[new_identifier] = new_tensor

    def _split_nodes(self, node_id: str, out_legs: Dict[str, List], in_legs: Dict[str, List],
                     splitting_function: Callable, out_identifier: str = "", in_identifier = "",
                     **kwargs):
        """
        Splits an node into two nodes using a specified function

        Args:
            node_id (str): The identifier of the node to be split.
            out_legs (Dict[str, List]): The legs associated to the output of the matricised
                node tensor. (The Q legs for QR and U legs for SVD)
            in_legs (Dict[str, List]): The legs associated to the input of the matricised
                node tensor: (The R legs for QR and the SVh legs for SVD)
            splitting_function (Callable): The function to be used for the splitting
            out_identifier (str, optional): An identifier for the tensor with the output
                legs. Defaults to "".
            in_identifier (str, optional): An identifier for the tensor with the input
                legs. Defaults to "".

        """
        node, tensor = self[node_id]
        out_legs = LegSpecification.from_dict(out_legs, node)
        in_legs = LegSpecification.from_dict(in_legs, node)

        # Getting the numerical value of the legs
        out_legs_int = out_legs.find_leg_values()
        in_legs_int = in_legs.find_leg_values()
        out_tensor, in_tensor = splitting_function(tensor, out_legs_int, in_legs_int, **kwargs)

        # Changing the ttn-structure (Here the old node is removed)
        if out_identifier == "":
            out_identifier = "out_of_" + node_id
        if in_identifier == "":
            in_identifier = "in_of_" + node_id
        self.split_node(node_id, out_identifier, out_legs.find_all_neighbour_ids(),
                         in_identifier, in_legs.find_all_neighbour_ids())
        self._tensors.pop(node_id)

        # Adding Tensors
        self._tensors[out_identifier] = out_tensor
        self._tensors[in_identifier] = in_tensor

        # New LegNodes
        out_node = LegNode.from_node(out_tensor, self.nodes[out_identifier])
        in_node = LegNode.from_node(in_tensor, self.nodes[in_identifier])
        self._nodes[out_identifier] = out_node
        self._nodes[in_identifier] = in_node

        # Currently the tensors q and r have the leg ordering
        # (new_leg(for out), parent_leg, virtual_leg, open_legs, new_leg(for in))
        if out_node.parent == in_node.identifier:
            out_node.last_leg_to_parent_leg()
            in_node.leg_to_last_child_leg(0)
        else:
            out_node.leg_to_last_child_leg(len(out_node.leg_permutation) - 1)
            # The new leg in the in_node is already in the correct place. Yay!

    def split_node_qr(self, node_id: str, q_legs: Dict[str, List], r_legs: Dict[str, List],
                       q_identifier: str = "", r_identifier: str = ""):
        """
        Splits a node into two nodes via QR-decomposition.

        The legs should be given as a dictionary with the keys
        "parent_leg", "child_legs" and "open_legs".
        If there is no parent, it should be denoted by None.

        Args:
            node_id (str): The node to be split
            q_legs (Dict[str, List]): The legs which should be part of the Q-tensor
            r_legs (Dict[str, List]): The legs which should be part of the R-tensor
            q_identifier (str, optional): An identifier for the Q-tensor. Defaults to "".
            r_identifier (str, optional): An identifier for the R-tensor. Defaults to "".
        """
        self._split_nodes(node_id, q_legs, r_legs, tensor_qr_decomposition,
                          out_identifier=q_identifier, in_identifier=r_identifier)

    def split_node_svd(self, node_id: str, u_legs: Dict, v_legs: Dict,
                       u_identifier: str = "", v_identifier: str = "",
                       **truncation_param):
        """
        Splits a node in two using singular value decomposition. In the process the tensors
         are truncated as specified by truncation parameters. The singular values
         are absorbed into the v_legs.

        Args:
            node_id (str): Identifier of the nodes to be split
            u_legs (Dict): The legs which should be part of the U tensor
            v_legs (Dict): The legs which should be part of the V tensor
            u_identifier (str): An identifier for the U-tensor. Defaults to "".
            v_identifier (str): An identifier for the V-tensor. Defaults to "".
        """
        self._split_nodes(node_id, u_legs, v_legs, contr_truncated_svd_splitting,
                          out_identifier=u_identifier, in_identifier=v_identifier,
                          **truncation_param)

    # Functions below this are just wrappers of external functions that are
    # linked tightly to the TTN and its structure. This allows these functions
    # to be overwritten for subclasses of the TTN with more known structure.
    # The additional structure allows for more efficent algorithms than the
    # general case.

    def canonical_form(self, orthogonality_center_id):
        """
        Brings the tree_tensor_network in canonical form with

        Parameters
        ----------
        orthogonality_center_id : str
            The id of the tensor node, which sould be the orthogonality center for
            the canonical form

        Returns
        -------
        None.
        """
        canonical_form(self, orthogonality_center_id)

    def completely_contract_tree(self, to_copy: bool = False):
        """
        Completely contracts the given tree_tensor_network by combining all
        nodes.
        (WARNING: Can get very costly very fast. Only use for debugging.)

        Parameters
        ----------
        to_copy: bool
            Wether or not the contraction should be perfomed on a deep copy.
            Default is False.

        Returns
        -------
        In case copy is True a deep copy of the completely contracted TTN is
        returned.

        """
        return completely_contract_tree(self, to_copy=to_copy)

    def contract_two_ttn(self, other):
        """
        Contracts two TTN with the same structure. Assumes both TTN use the same
        identifiers for the nodes.

        Parameters
        ----------
        other : TreeTensorNetwork

        Returns
        -------
        result_tensor: np.ndarray
            The contraction result.

        """
        return contract_two_ttn(self, other)

    def apply_hamiltonian(self, hamiltonian: Hamiltonian, conversion_dict: dict[str, np.ndarray], skipped_vertices=None):
        """
        Applies a Hamiltonian term by term locally to a TTN. Assumes that the input TTN represents a statevector
        such that each TensorNode has the following memory layout: [parent, child_1, child_2, ..., child_n, output].
        """

        def allocate_output_tensor(node_id, state_diagram, conversion_dict):
            """
            Allocates output tensor for each node
            """
            he = state_diagram.hyperedge_colls[node_id].contained_hyperedges[0]
            operator_label = he.label
            operator = conversion_dict[operator_label]
            # Should be square operators
            phys_dim = operator.shape[0]

            total_tensor_shape = [0] * len(he.vertices)
            slice_tensor_shape = [0] * len(he.vertices)
            total_tensor_shape.extend([phys_dim])
            slice_tensor_shape.extend([phys_dim])
            node = self.nodes[node_id]
            neighbours = node.neighbouring_nodes()
            for leg_index, neighbour_id in enumerate(neighbours.keys()):
                vertex_coll = state_diagram.get_vertex_coll_two_ids(
                    node_id, neighbour_id)
                for index_value, vertex in enumerate(vertex_coll.contained_vertices):
                    vertex.index = (leg_index, index_value)
                # The number of vertices is equal to the number of bond-dimensions required.
                total_tensor_shape[leg_index] = len(
                    vertex_coll.contained_vertices) * self.tensors[node_id].shape[neighbours[neighbour_id]]
                slice_tensor_shape[leg_index] = self.tensors[node_id].shape[neighbours[neighbour_id]]
            output_tensor = np.zeros(total_tensor_shape, dtype=np.cdouble)
            return output_tensor, slice_tensor_shape

        from .ttno.state_diagram import StateDiagram
        state_diagram = StateDiagram.from_hamiltonian(hamiltonian, self)

        # Adding the operator corresponding to each hyperedge to the tensor
        for node_id, hyperedge_coll in state_diagram.hyperedge_colls.items():
            output_tensor, output_slice_shape = allocate_output_tensor(
                node_id, state_diagram, hamiltonian.conversion_dictionary)
            local_tensor = self._nodes[node_id]

            for he in hyperedge_coll.contained_hyperedges:
                index_value = [0] * len(he.vertices)

                for vertex in he.vertices:
                    index_value[vertex.index[0]] = vertex.index[1]

                slice_indexing = [slice(index * size, (index+1) * size, 1)
                                  for (index, size) in zip(index_value, output_slice_shape)]
                slice_indexing.extend([slice(None)])
                slice_indexing = tuple(slice_indexing)
                operator_label = he.label
                operator = hamiltonian.conversion_dictionary[operator_label]
                output_slice = np.tensordot(
                    self.tensors[local_tensor.identifier], operator, axes=([local_tensor.nlegs()-1], [1]))
                output_tensor[slice_indexing] += output_slice

            self.tensors[node_id] = output_tensor
