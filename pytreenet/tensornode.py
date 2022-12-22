import numpy as np
import uuid

from copy import deepcopy

from .util import crandn, copy_object

class TensorNode(object):
    """
    A node in a tree tensor network that contains a tensor and which legs
    are contracted to which other tensors.

    General structure and parts of the code from treelib.node
    """

    def __init__(self, tensor, tag=None, identifier=None):

        self._tensor = tensor

        if identifier == None:
            self._identifier = str(uuid.uuid1())
        else:
            self._identifier = str(identifier)
        if tag == None:
            self._tag = self.identifier
        else:
            self._tag = tag
            
        #At the beginning all legs are open
        self._open_legs = list(np.arange(tensor.ndim))
        
        self._parent_leg = []
        self._children_legs = dict()

    @property
    def tensor(self):
        """
        The tensor in form of a numpy array associated to a tensornode.
        """
        return self._tensor

    @tensor.setter
    def tensor(self, new_tensor):
        """
        Set value of tensor. Can be used to update the tensor of a node.
        """
        assert self._tensor.ndim == new_tensor.ndim, "Tensors at the same node should have the same number of dimensions/legs."
        self._tensor = new_tensor

    @property
    def identifier(self):
        """
        An identifier that is unique to this node.
        """
        return self._identifier

    @identifier.setter
    def identifier(self, new_identifier):
        if new_identifier == None:
            self._identifier = str(uuid.uuid1())
        else:
            self._identifier = str(new_identifier)

    @property
    def tag(self):
        """
        A human readable tag for this node.
        """
        return self._tag

    @tag.setter
    def tag(self, new_tag):
        if new_tag == None:
            self._tag = self.identifier
        else:
            self._tag = new_tag

    @property
    def open_legs(self):
        """
        A list of tensor legs that are not contracted with other tensors
        """
        return self._open_legs

    @property
    def parent_leg(self):
        """
        A one or zero element list, that potentially contains the parent's
        identifier as first enty and the tensor leg that is contracted with it as the second.
        """
        return self._parent_leg

    def parent_leg_dict(self):
        """
        Returns
        -------
        parent_leg_dict: dict
            The parent_leg list as a dictionary.
        """
        if len(self.parent_leg) == 0:
            return dict()
        else:
            return {self.parent_leg[0]: self.parent_leg[1]}

    @property
    def children_legs(self):
        """
        The legs contracted with the children's tensors.
        The dictionary contains the children's identifier as key and the
        corresponding contracted leg as value.
        """
        return self._children_legs

    def __eq__(self, other):
        """
        Two tensors nodes are considered equal, if everything is equal, except
        for the tags and identifier
        """
        return ((self.tensor, self.open_legs, self.parent_leg, self.children_legs) ==
                (other.tensor, other.open_legs, other.parent_leg, other.children_legs))

    def neighbouring_nodes(self, with_legs=True):
        """
        Finds the neighbouring tensor nodes of this node with varying
        additional information.

        Parameters
        ----------
        with_legs : boolean, optional
            If True the legs of neighbours are also returned. The default is True.

        Returns
        -------
        neighbour_legs: dict
            Is returned, if with_legs=True. A dictionary that contains all the
            identifiers of tensor nodes that are contracted with this one and
            the leg they are attached to.
        neighbour_ids: list of str
            Is returned it with_legs=False. A list containing the identifiers
            of all the tensor nodes this node is contracted with.
        """

        if with_legs:
            parent_dict = self.parent_leg_dict()
            neighbour_legs = deepcopy(self.children_legs)
            neighbour_legs.update(parent_dict)
            return neighbour_legs

        else:
            neighbour_ids = list(self.children_legs.keys())
            if not self.is_root():
                neighbour_ids.append(self.parent_leg[0])
            return neighbour_ids

    def check_existence_of_open_legs(self, open_leg_list):
        if len(open_leg_list) == 1:
            assert open_leg_list[0] in self.open_legs, f"Tensor node with identifier {self.identifier} has no open leg {open_leg_list[0]}."
        else:
            assert all(open_leg in self.open_legs for open_leg in open_leg_list)

    def open_leg_to_parent(self, open_leg, parent_id):
        """
        Change an open leg to be a leg contracted with the parent node.
        """
        self.check_existence_of_open_legs([open_leg])

        self._parent_leg.append(parent_id)
        self._parent_leg.append(open_leg)
        self._open_legs.remove(open_leg)

    def open_legs_to_children(self, open_leg_list, identifier_list):
        """
        Change a list of open legs to legs contracted with children.
        """
        open_leg_list = list(open_leg_list)
        identifier_list = list(identifier_list)

        self.check_existence_of_open_legs(open_leg_list)
        assert len(open_leg_list) == len(identifier_list), "Children and identifier list should be the same length"

        new_children_legs = dict(zip(identifier_list, open_leg_list))
        self._children_legs.update(new_children_legs)
        self._open_legs = [open_leg for open_leg in self._open_legs if open_leg not in open_leg_list]

    def open_leg_to_child(self, open_leg, child_id):
        """
        Only changes a single open leg to be contracted with a child
        """
        self.open_legs_to_children([open_leg], [child_id])

    def parent_leg_to_open_leg(self):
        """
        If existant, changes the leg contracted with a parent node, to an
        open leg. (Note: this will remove any relation of this node to the parent)
        """
        self.open_legs.append(self.parent_leg[1])
        self._parent_leg = []

    def children_legs_to_open_legs(self, children_identifier_list):
        """
        Makes legs contracted with children identified in children_identifier_list
        into open legs.
        """
        children_identifier_list = list(children_identifier_list)

        assert all(identifiers in self.children_legs for identifiers in children_identifier_list), "All identifiers must correspond a child of the node."

        children_legs_list = [self.children_legs[identifier] for identifier in children_identifier_list]
        self._open_legs.extend(children_legs_list)

        for identifier in children_identifier_list:
            del self._children_legs[identifier]

    def child_leg_to_open_leg(self, child_identifier):
        """
        Makes a leg contracted with the child identified by child_identifier
        into an open leg.
        """
        assert child_identifier in self.children_legs, "Identifier should belong to a child of this node."

        self.children_legs_to_open_legs(child_identifier)

    def absorb_tensor(self, absorbed_tensor, absorbed_tensors_leg_indices, this_tensors_leg_indices):
        """
        Absorbes the absorbed_tensor into this instance's tensor by contracting
        the absorbed_tensors_legs of the absorbed_tensor and the legs
        this_tensors_legs of this instance's tensor'

        Parameters
        ----------
        absorbed_tensor: ndarray
            Tensor to be absorbed.
        absorbed_tensors_leg_indices: int or tuple of int
            Legs that are to be contracted with this instance's tensor.
        this_tensors_leg_indices:
            The legs of this instance's tensor that are to be contracted with
            the absorbed tensor.
        """
        if type(absorbed_tensors_leg_indices) == int:
            absorbed_tensors_leg_indices = (absorbed_tensors_leg_indices, )
        if type(this_tensors_leg_indices) == int:
            this_tensors_leg_indices = (this_tensors_leg_indices, )
            
        assert len(absorbed_tensors_leg_indices) == len(this_tensors_leg_indices)
        
        if len(absorbed_tensors_leg_indices) == 1:
            this_tensors_leg_index = this_tensors_leg_indices[0]
            self.tensor = np.tensordot(self.tensor, absorbed_tensor,
                                       axes=(this_tensors_leg_indices, absorbed_tensors_leg_indices))
            
            this_tensors_indices = tuple(range(self.tensor.ndim))
            transpose_perm = (this_tensors_indices[0:this_tensors_leg_index]
                              + (this_tensors_indices[-1], )
                              + this_tensors_indices[this_tensors_leg_index:-1])
            self.tensor = self.tensor.transpose(transpose_perm)
        else:
            raise NotImplementedError

    def is_root(self):
        """
        Determines if this node is a root node, i.e., a node without a parent.
        """
        if len(self.parent_leg) == 0:
            return True
        else:
            return False

    def has_x_children(self, x: int):
        """
        Determines if the node has at least x-many children
        """
        assert x > 0, "The number of children will be at least zero. Choose a bigger number."

        if len(self._children_legs) >= x:
            return True
        else:
            return False

    def is_leaf(self):
        """
        Determines if the node is a leaf, i.e., has at least one child.
        """
        return not self.has_x_children(x=1)

    def has_open_leg(self):
        """
        Determines if the node has any open legs.
        """
        return len(self.open_legs) > 0

    def is_child_of(self, other_node_id):
        """
        Determines if this instance is the child of the node with identifier
        other_node_id
        """
        return other_node_id in self.parent_leg

    def is_parent_of(self, other_node_id):
        """
        Determines if this instance is the parent of the node with identifier
        other_node_id
        """
        return other_node_id in self.children_legs

def random_tensor_node(shape, tag=None, identifier=None):
    """
    Creates a tensor node with an a random associated tensor with shape=shape.
    """
    rand_tensor = crandn(shape)
    return TensorNode(tensor = rand_tensor, tag=tag, identifier=identifier)

def assert_legs_matching(node1, leg1, node2, leg2):
    """
    Asserts if the dimensions of leg1 of node1 and leg2 of node2 match.
    """
    leg1_dimension = node1.tensor.shape[leg1]
    leg2_dimension = node2.tensor.shape[leg2]
    assert leg1_dimension == leg2_dimension

def conjugate_node(node, deep_copy=True, conj_neighbours=False):
    """
    Returns a copy of the same node but with all entries complex conjugated
    and with a new identifier and tag. If conj_neighbours all the identifiers
    in node's legs will have ann added "conj_" in front.
    """
    conj_node = copy_object(node, deep = deep_copy)
    new_identifier = "conj_" + conj_node.identifier
    new_tag = "conj_" + conj_node.tag

    conj_node.identifier = new_identifier
    conj_node.tag = new_tag

    if conj_neighbours:
        if not node.is_root():
            conj_node.parent_leg[0] = "conj_" + conj_node.parent_leg[0]
        children_legs = conj_node.children_legs
        conj_children_legs = {("conj_" + child_id) : children_legs[child_id]
                              for child_id in children_legs}
        conj_node.children_legs = conj_children_legs

    conj_node.tensor = np.conj(conj_node.tensor)
    return conj_node
