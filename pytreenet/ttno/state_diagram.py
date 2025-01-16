from __future__ import annotations
from typing import Dict, Tuple, List, Union
from copy import copy
from collections import deque
from enum import Enum
import uuid

import numpy as np

from ..core.tree_structure import TreeStructure
from ..operators.hamiltonian import Hamiltonian
from ..util.ttn_exceptions import NoConnectionException, NotCompatibleException
from .vertex import Vertex
from .hyperedge import HyperEdge
from .collections import VertexColl, HyperEdgeColl
from .single_term_diagram import SingleTermDiagram

from .bipartite_graph import BipartiteGraph, minimum_vertex_cover
from .symbolic_gaussian_elimination_fraction import gaussian_elimination, print_matrix
from copy import deepcopy
from fractions import Fraction
import hashlib


class TTNOFinder(Enum):
    """
    An Enum to switch between different construction modes of a state diagram.
    TREE: The tree comparison method construction of state diagram bottom up.
    SGE: The Symbolic Gaussian Elimination method with bipartite graph theory to construct TTNO. 
    BIPARTITE: The Bipartite Graph method. Suboptimal implemented method in Ren et al. 2020.
    BASE: The base method without any compression. Worst possible construction method.
    """
    TREE = "Tree"
    SGE = "Symbolic Gaussian Elimination"
    BIPARTITE = "Bipartite Graph"
    BASE = "Base"


class StateDiagram():
    """ 
    A state diagram represents a Hamiltonian (or other operator)

    In principle it can represent any tree tensor network. It contains vertices
    and hyperedges, that can be sorted into an underlying tree structure.

    Attributes:
        vertex_colls (VertexColl): A collection of all vertices in the state
            diagram. They are grouped by the edge that corresponds to them.
        hyperedge_colls (HyperEdgeColl): A collection of all hyperedges in the
            state diagram. They are grouped by the node that corresponds to
            them.
        reference_tree (TreeStructure): Provides the underlying tree structure
            with connectivity and identifiers.
    """

    def __init__(self, reference_tree: TreeStructure):
        """
        Initialises a state diagram.

        Args:
            reference_tree (TreeStructure): Provides the underlying tree
                structure with connectivity and identifiers.
        """
        self.vertex_colls: Dict[Tuple[str, str], VertexColl] = {}
        self.hyperedge_colls: Dict[str, HyperEdgeColl] = {}
        self.reference_tree = reference_tree

    def __repr__(self) -> str:
        """
        A string representation of a state diagram.
        """
        all_he = self.get_all_hyperedges()
        all_vert = self.get_all_vertices()
        string = "hyperedges:\n"
        for hyperedge in all_he:
            string += str(hyperedge) + "\n"
        string += "\n vertices:\n"
        for vertex in all_vert:
            string += str(vertex) + "\n"

        return string

    def get_all_vertices(self) -> List[Vertex]:
        """
        Returns all vertices from all collections in a list.
        """
        all_vert = []
        for vertex_coll in self.vertex_colls.values():
            all_vert.extend(vertex_coll.contained_vertices)
        return all_vert

    def get_all_hyperedges(self) -> List[HyperEdge]:
        """
        Returns all hyperedges from all collections in a list
        """
        all_he = []
        for hyperedge_coll in self.hyperedge_colls.values():
            all_he.extend(hyperedge_coll.contained_hyperedges)
        return all_he

    def get_vertex_coll_two_ids(self,
                                id1: str,
                                id2: str) -> VertexColl:
        """
        Obtain the vertex collection corresponding to a specified edge.

        The edge is specified by the two node identifiers given and is the edge
        connecting the two. Since the order of the identifiers in the edge is
        irrelevant, the same is true for the supplied identifiers.

        Args:
            id1 (str): One identifier corresponding a node connected by the
                edge.
            id2 (str): One identifier corresponding a node connected by the
                edge.

        Returns:
            VertexColl: The vertex collection corresponding to the specified
                edge.
        """
        key1 = (id1, id2)
        key2 = (id2, id1)
        if key1 in self.vertex_colls:
            return self.vertex_colls[key1]
        if key2 in self.vertex_colls:
            return self.vertex_colls[key2]
        errstr = f"There is no vertex collection corresponding to and edge between {id1} and {id2}!"
        raise NoConnectionException(errstr)

    def add_hyperedge(self, hyperedge: HyperEdge):
        """
        Adds a hyperedge to the correct collection and to the state diagram.

        Args:
            hyperedge (HyperEdge): The hyperedge to be added.
        """
        node_id = hyperedge.corr_node_id

        if node_id in self.reference_tree.nodes:
            self.hyperedge_colls[node_id].contained_hyperedges.append(
                hyperedge)
        else:
            errstr = f"No node with identifier {node_id} in reference tree!"
            raise NotCompatibleException(errstr)

    @classmethod
    def from_hamiltonian(cls,
                         hamiltonian: Hamiltonian,
                         ref_tree: TreeStructure,
                         method: TTNOFinder = TTNOFinder.TREE) -> StateDiagram:
        """
        Creates a state diagram equivalent to a given Hamiltonian.

        Args:
            hamiltonian (Hamiltonian): Hamiltonian for which the state
                diagram is to be found
            ref_tree (TreeTensorNetwork): Supplies the tree topology which
                is to be incorporated into the state diagram.
            method (TTNOFinder): The construction method to be used.

        Returns:
            StateDiagram: The final state diagram

        """
        state_diagram = None
        
        if method == TTNOFinder.TREE:
            state_diagram = cls.from_hamiltonian_tree_comparison(hamiltonian,
                                                                 ref_tree)
        elif method == TTNOFinder.SGE:
            state_diagram = cls.from_hamiltonian_modified(hamiltonian, ref_tree, method.SGE)

        elif method == TTNOFinder.BIPARTITE:
            state_diagram = cls.from_hamiltonian_modified(hamiltonian, ref_tree, method.BIPARTITE)

        elif method == TTNOFinder.BASE:
            state_diagram = cls.from_hamiltonian_base(hamiltonian, ref_tree)
        else:
            errstr = "Invalid construction method!"
            raise ValueError(errstr)
        return state_diagram

    @classmethod
    def from_hamiltonian_tree_comparison(cls,
                                         hamiltonian: Hamiltonian,
                                         ref_tree: TreeStructure):
        """
        Constructs a Hamiltonian using the leaf to root comparison method.

        Args:
            hamiltonian (Hamiltonian): Hamiltonian for which the state
                diagram is to be found
            ref_tree (TreeTensorNetwork): Supplies the tree topology which
                is to be incorporated into the state diagram.

        Returns:
            StateDiagram: The final state diagram.

        """
        state_diagram = None
        for term in hamiltonian.terms:
            if state_diagram is None:
                state_diagram = cls.from_single_term(term, ref_tree)
            else:
                state_diagram.add_single_term(term)
        return state_diagram

    @classmethod
    def from_single_term(cls,
                         term: tuple[Fraction,str,Dict[str,str]],
                         reference_tree: TreeStructure):
        """
        Basically a wrap of ``SingleTermDiagram.from_single_term``.
        """
        single_term_diagram = SingleTermDiagram.from_single_term(term,
                                                                 reference_tree)
        return cls.from_single_state_diagram(single_term_diagram)

    @classmethod
    def from_single_state_diagram(cls,
                                  single_term_diag: SingleTermDiagram) -> StateDiagram:
        """
        Transforms a single state diagram to a general one.

        Args:
            single_term_diag (SingleTermDiagram): Represents a single term
                using a simpler structure than the general state diagrams.

        Returns:
            state_diagram (StateDiagram): The equivalent general state diagram.
        """
        state_diagram = cls(single_term_diag.reference_tree)
        # Creating HyperEdgeCollections
        for node_id, hyperedge in single_term_diag.hyperedges.items():
            new_hyperedge_coll = HyperEdgeColl(node_id, [hyperedge])
            state_diagram.hyperedge_colls[node_id] = new_hyperedge_coll
        # Creating VertexCollections
        for edge_id, vertex in single_term_diag.vertices.items():
            new_vertex_coll = VertexColl(edge_id, [vertex])
            state_diagram.vertex_colls[edge_id] = new_vertex_coll
        return state_diagram

    def add_single_term(self, term: tuple[Fraction,str,Dict[str,str]]):
        """
        Modifies the state diagram to add a term.

        Adds a term to the state diagram. This means the diagram is modified in
        a way such that it represents Hamiltonian + term instead of only
        the Hamiltonian.

        Args:
            term (Dict[str,str]): A dictionary containing the node_ids as keys
                and the operator applied to that node as a value.
        """
        single_term_diagram = SingleTermDiagram.from_single_term(
            term, self.reference_tree)

        self._mark_contained_vertices(single_term_diagram)
        self._add_hyperedges(single_term_diagram)
        # At this point all vertices have their marking reset already.

    def _mark_contained_vertices(self, single_term_diagram):
        leaves = self.reference_tree.get_leaves()
        for leaf_id in leaves:
            hyperedge_label = single_term_diagram.get_hyperedge_label(leaf_id)
            hyperedge_coll = self.hyperedge_colls[leaf_id]
            # Ensure the hyperedges have the correct label
            potential_start_hyperedges = hyperedge_coll.get_hyperedges_by_label(
                hyperedge_label)
            potential_start_hyperedges = [hyperedge for hyperedge in potential_start_hyperedges
                                          if hyperedge.all_but_one_vertex_contained()]
            next_vertex = None
            for hyperedge in potential_start_hyperedges:
                next_vertex = self._find_and_mark_new_vertex(hyperedge)
                if not next_vertex is None:
                    break

            if not next_vertex is None:
                new_node_id = next_vertex.get_second_node_id(leaf_id)
                self._find_new_he(next_vertex, new_node_id,
                                  single_term_diagram)

    def _find_and_mark_new_vertex(self, current_hyperedge):
        current_node_id = current_hyperedge.corr_node_id
        uncontained_vertex = current_hyperedge.get_single_uncontained_vertex()
        next_node_id = uncontained_vertex.get_second_node_id(current_node_id)
        if not current_hyperedge.vertex_single_he(next_node_id):
            # In this case we would add multiple paths
            return None
        vertex_coll = self.get_vertex_coll_two_ids(
            current_node_id, next_node_id)
        if vertex_coll.contains_contained():
            # This vertex collection already has a contained vertex
            return None
        uncontained_vertex.contained = True
        return uncontained_vertex

    def _find_new_he(self, current_vertex, node_id, single_term_diagram):

        he_of_vertex = current_vertex.get_hyperedges_for_one_node_id(node_id)
        potential_new_he = [hyperedge for hyperedge in he_of_vertex
                            if hyperedge.all_but_one_vertex_contained()]

        desired_label = single_term_diagram.get_hyperedge_label(node_id)
        potential_new_he = [hyperedge for hyperedge in potential_new_he
                            if hyperedge.label == desired_label]

        next_vertex = None
        for hyperedge in potential_new_he:
            next_vertex = self._find_and_mark_new_vertex(hyperedge)
            if not next_vertex is None:
                break

        if not next_vertex is None:
            new_node_id = next_vertex.get_second_node_id(node_id)
            self._find_new_he(next_vertex, new_node_id, single_term_diagram)

    def _add_hyperedges(self, single_term_diagram):
        for node_id in self.reference_tree.nodes:
            self._add_hyperedges_rec(node_id, single_term_diagram)

    def _add_hyperedges_rec(self, node_id, single_term_diagram):
        hyperedge_coll = self.hyperedge_colls[node_id]
        hyperedges = hyperedge_coll.get_completely_contained_hyperedges()
        desired_label = single_term_diagram.get_hyperedge_label(node_id)

        if len(hyperedges) == 0:
            vertices_to_connect_to_new_he = self._find_vertices_connecting_to_he(
                node_id)
            new_hyperedge = HyperEdge(
                node_id, desired_label, vertices_to_connect_to_new_he, hyperedges[0].lambda_coeff, hyperedges[0].gamma_coeff)

        # DOES THIS WORK AS INTENDED?
        elif len(hyperedges) >= 1:
            for hyperedge in hyperedges:
                vertices_to_connect_to_new_he = copy(hyperedge.vertices)
                if hyperedge.label == desired_label:
                    new_hyperedge = None
                    break
                else:
                    new_hyperedge = HyperEdge(
                        node_id, desired_label, vertices_to_connect_to_new_he, hyperedge.lambda_coeff, hyperedge.gamma_coeff)

        # Allows for reset directly instead of after the fact
        # Thus we don't have to run through all vertices of the entire diagram.
        for vertex in vertices_to_connect_to_new_he:
            vertex.runtime_reset()

        if not new_hyperedge is None:
            self.add_hyperedge(new_hyperedge)
            for vertex in vertices_to_connect_to_new_he:
                vertex.hyperedges.append(new_hyperedge)

    def _find_vertices_connecting_to_he(self, node_id):
        node = self.reference_tree.nodes[node_id]
        neighbour_ids = node.neighbouring_nodes()

        vertices_to_connect_to_new_he = []
        for neighbour_id in neighbour_ids:
            vertex_coll = self.get_vertex_coll_two_ids(node_id, neighbour_id)
            vertex_to_connect = vertex_coll.get_all_marked_vertices()
            if len(vertex_to_connect) == 0:
                vertex_to_connect = Vertex((node_id, neighbour_id), [])
                vertex_to_connect.new = True
                vertex_coll.contained_vertices.append(vertex_to_connect)
            elif len(vertex_to_connect) == 1:
                vertex_to_connect = vertex_to_connect[0]
            else:
                assert False, "Something went terribly wrong!"

            vertices_to_connect_to_new_he.append(vertex_to_connect)
        return vertices_to_connect_to_new_he

    def obtain_tensor_shape(self, node_id: str,
                            conversion_dict: Dict[str, np.ndarray]) -> Tuple[int, ...]:
        """
        Find the shape of the tensor corresponding to a node in the equivalent
        TTNO.

        Args:
            node_id (str): The identifier of a node.
            conversion_dict (Dict[str, np.ndarray]): A dictionary to convert
                the labels into arrays, to determine the required physical
                dimension.

        Returns:
            Tuple[int, ...]: The shape of the tensor in the equivalent TTNO in
                the format 
                ``(parent_shape, children_shape, phys_dim, phys_dim)``.
                The children are in the same order as in the node.
        """
        he = self.hyperedge_colls[node_id].contained_hyperedges[0]
        operator_label = he.label
        operator = conversion_dict[operator_label]
        # Should be square operators
        assert operator.shape[0] == operator.shape[1]
        phys_dim = operator.shape[0]
        total_shape = [0] * len(he.vertices)
        total_shape.extend([phys_dim, phys_dim])
        neighbours = self.reference_tree.nodes[node_id].neighbouring_nodes()
        for leg_index, neighbour_id in enumerate(neighbours):
            vertex_coll = self.get_vertex_coll_two_ids(node_id, neighbour_id)
            # The number of vertices is equal to the number of bond-dimensions
            # required.
            total_shape[leg_index] = len(vertex_coll.contained_vertices)
        return tuple(total_shape)

    def set_all_vertex_indices(self):
        """
        Indexes all vertices contained in this state diagram.

        This index is the index value to which this vertex corresponds in the
        bond dimension.
        """
        for vertex_coll in self.vertex_colls.values():
            vertex_coll.index_vertices()

    def reset_markers(self):
        """
        Resets the contained and new markers of every vertex in the diagram.
        """
        for vertex_col in self.vertex_colls.values():
            for vertex in vertex_col.contained_vertices:
                vertex.contained = False
                vertex.new = False

    @classmethod
    def from_hamiltonian_base(cls, hamiltonian, ref_tree,)-> StateDiagram:
        """
        Constructs state diagram without any optimization.
        Args:
            hamiltonian (Hamiltonian): Hamiltonian for which the state
                diagram is to be found
            ref_tree (TreeTensorNetwork): Supplies the tree topology which
                is to be incorporated into the state diagram.
        Returns:
            StateDiagram: The final state diagram
        """

        state_diagrams = cls.get_state_diagrams(hamiltonian,ref_tree)
        compound_state_diagram = cls.get_state_diagram_compound(state_diagrams)

        compound_state_diagram.coeffs = hamiltonian.coeffs

        compound_state_diagram.hamiltonian = hamiltonian

        for i, hyperedge in enumerate(compound_state_diagram.hyperedge_colls[ref_tree.root_id].contained_hyperedges):
            hyperedge.lambda_coeff *= compound_state_diagram.coeffs[i][0]
            hyperedge.gamma_coeff = compound_state_diagram.coeffs[i][1]

        return compound_state_diagram

    @classmethod
    def from_hamiltonian_modified(cls, hamiltonian, ref_tree,  method: TTNOFinder = TTNOFinder.SGE ) -> StateDiagram:
        """Creates optimal state diagram equivalent to a given Hamiltonian

        Args:
            hamiltonian (Hamiltonian): Hamiltonian for which the state
                diagram is to be found
            ref_tree (TreeTensorNetwork): Supplies the tree topology which
                is to be incorporated into the state diagram.

        Returns:
            StateDiagram: The final state diagram
        """
        
        # Get individual state diagrams and combine them into a compound state diagram
        state_diagrams = cls.get_state_diagrams(hamiltonian,ref_tree)
        compound_state_diagram = cls.get_state_diagram_compound(state_diagrams)

        compound_state_diagram.coeffs = hamiltonian.coeffs
        
        compound_state_diagram.SGE = method == TTNOFinder.SGE

        coeffs_assigned = False 

        compound_state_diagram.hamiltonian = hamiltonian

        # Queue for tree traversal. We traverse the tree in a BFS manner
        queue = deque()

        # Add all children of the root node to the queue
        for child in ref_tree.nodes[ref_tree.root_id].children:
            queue.append((ref_tree.root_id,child))

        while queue:
            
            # Level size is the number of nodes on the current level. This allows us to pass twice.
            level_size = len(queue)

            # Combining hyperedges on the current level
            for parent, current_node in queue:

                # Combining subtrees          
                local_hyperedges = copy(compound_state_diagram.hyperedge_colls[current_node].contained_hyperedges) 
                compound_state_diagram.combine_subtrees(local_hyperedges, parent)
                                
            # Combining hyperedges on the parent level, optimising the cut site
            for _ in range(level_size):

                # After second pass, we pop the element from the queue
                parent, current_node = queue.popleft()

                # Combining v nodes            
                local_vs = copy(compound_state_diagram.hyperedge_colls[parent].contained_hyperedges)

                # Initial assign of coefficients to the root hyperedges
                if not coeffs_assigned:
                    coeffs_assigned = True
                    for i, hyperedge in enumerate(local_vs):
                        
                        hyperedge.lambda_coeff *= compound_state_diagram.coeffs[i][0]
                        hyperedge.gamma_coeff = compound_state_diagram.coeffs[i][1]
                        
                compound_state_diagram.cut_and_optimise(local_vs, current_node, parent)
                
                # Add all children of the current node to the queue (BFS traversal)
                for child in ref_tree.nodes[current_node].children:
                    queue.append((current_node,child))

        return compound_state_diagram

    def cut_and_optimise(self, local_vs, current_node, parent):
        """
        Cuts the hyperedges at the cut site and optimises the hyperedges.
        Applies symbolic gaussian elimination and bipartite graph theory depending on the method.
        """

        hash_dict = {}
        
        # Comparing V nodes and combining them if they are the same
        for element in local_vs:
            v_hash = element.calculate_v_hash(current_node, parent)

            if v_hash not in hash_dict:
                hash_dict[v_hash] = []

            for element2 in hash_dict[v_hash]:
                v1 = element.find_vertex(current_node)
                v2 = element2.find_vertex(current_node)

                # This is the problematic case that we try to avoid, final implementation does not have this case anymore.
                if (element.lambda_coeff != element2.lambda_coeff or element.gamma_coeff != element2.gamma_coeff) and self._compare_lists_by_identity(v1.get_hyperedges_for_one_node_id(current_node), v2.get_hyperedges_for_one_node_id(current_node)):
                    v_hash = hashlib.sha256((v_hash + str(uuid.uuid1()) ).encode()).hexdigest()
                    element.v_hash = v_hash
                    hash_dict[v_hash] = []
            
                    
            hash_dict[v_hash].append(element)
        

        u_nodes_enumerated = {item.identifier : i for i, item in enumerate(copy(self.hyperedge_colls[current_node].contained_hyperedges))}
        u_nodes_enumerated_ind = {i : item for i, item in enumerate(copy(self.hyperedge_colls[current_node].contained_hyperedges))}

        v_nodes_enumerated = {item : i for i, item in enumerate(hash_dict.keys())}
        v_nodes_enumerated_ind = {i : item for i, item in enumerate(hash_dict.keys())}

        # Setting up and filling the Gamma matrix
        
        Gamma = [[Fraction(0) for _ in range(len(v_nodes_enumerated))] for _ in range(len(u_nodes_enumerated))]    
        for element in local_vs:
            el_vertex = element.find_vertex(current_node)
            connected_u_nodes = el_vertex.get_hyperedges_for_one_node_id(current_node)

            for u_node in connected_u_nodes:
                Gamma[u_nodes_enumerated[u_node.identifier]][v_nodes_enumerated[element.v_hash]] = Fraction(element.lambda_coeff) if element.gamma_coeff == "Free" else (Fraction(element.lambda_coeff), element.gamma_coeff)     

            element.gamma_coeff = "Free"
            element.lambda_coeff = 1

        # Symbolic Gaussian Elimination

        Op_l, Gamma_u, Op_r = gaussian_elimination(deepcopy(Gamma))

        # Can check if Op_l . Gamma_u . Op_r = Gamma

        # Bipartite graph algorithm to new Gamma matrix

        m, n = len(Gamma_u), len(Gamma_u[0])

        # Remove all vertices in the cut site
        for u in self.hyperedge_colls[current_node].contained_hyperedges:
            u.vertices.remove(u.find_vertex(parent))
        for v in self.hyperedge_colls[parent].contained_hyperedges:
            v.vertices.remove(v.find_vertex(current_node))

        self.vertex_colls[(parent,current_node)].contained_vertices = []

        # Remove duplicated v vertices in the cut size

        for hash_key, elements in hash_dict.items():
            if len(elements) > 1:
                for element in elements[1:]:
                    self._remove_hyperedge_from_diagram(element)
                    for vert in element.vertices:
                        self._remove_hyperedge_from_vertex(vert, element)
            hash_dict[hash_key] = elements[0]   
        
        if self.SGE : 
            edges_enumerated =  {(i, j) : Gamma_u[i][j] for i in range(m) for j in range(n) if Gamma_u[i][j] != 0}
    
            edges = list(edges_enumerated.keys())

            bigraph = BipartiteGraph(m, n, edges)
            u_cover, v_cover = minimum_vertex_cover(bigraph)

            edges_enumerated_old = {(i, j) : Gamma[i][j] for i in range(len(u_nodes_enumerated)) for j in range(len(v_nodes_enumerated)) if Gamma[i][j] != 0}
            bigraph_old  = BipartiteGraph(len(u_nodes_enumerated), len(v_nodes_enumerated), list(edges_enumerated_old.keys()))
            u_cover_old, v_cover_old = minimum_vertex_cover(bigraph_old)

            # When Gaussian Elimination result is not better, we revert back to the old state
            if len(u_cover + v_cover) >= len(u_cover_old + v_cover_old):
                
                u_cover = u_cover_old
                v_cover = v_cover_old

                edges_enumerated = edges_enumerated_old
                edges = list(edges_enumerated_old)
                bigraph = bigraph_old

                m, n = len(Gamma), len(Gamma[0])
    
                Op_l = [[1 if i == j else 0 for j in range(m)] for i in range(m)]
                Op_r = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

        # Only Bipartite Graph Algorithm
        else:
            
            edges_enumerated = {(i, j) : Gamma[i][j] for i in range(len(u_nodes_enumerated)) for j in range(len(v_nodes_enumerated)) if Gamma[i][j] != 0}
            bigraph  = BipartiteGraph(len(u_nodes_enumerated), len(v_nodes_enumerated), list(edges_enumerated.keys()))
            u_cover, v_cover = minimum_vertex_cover(bigraph)
            edges = list(edges_enumerated)

            m, n = len(Gamma), len(Gamma[0])
            Op_l = [[1 if i == j else 0 for j in range(m)] for i in range(m)]
            Op_r = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
                      
        
        # Creating Combined U and V lists
        ulist = [[] for _ in range(m)] # [[(u_temp, Op_l[i,j]), .. ] , [] , [] , []]
        vlist = [[] for _ in range(n)]

        for i in range(len(Op_l)):
            u_temp = u_nodes_enumerated_ind[i]
            dupl = False

            for j in range(len(Op_l[0])):
                if Op_l[i][j] != 0:
                    if not dupl:
                        ulist[j].append((u_temp, Op_l[i][j]))

                        dupl = True
                    else:

                        new_h = self._copy_node(u_temp, current_node, parent, current_node)
                        ulist[j].append((new_h, Op_l[i][j]))

        for j in range(len(Op_r[0])):
            v_temp = hash_dict[v_nodes_enumerated_ind[j]]
            dupl = False
            for i in range(len(Op_r)):
                if Op_r[i][j] != 0:
                    if not dupl:
                        vlist[i].append((v_temp, Op_r[i][j]))
                        dupl = True
                    else:
                        
                        new_h = self._copy_node(v_temp, current_node, parent, parent)
                        vlist[i].append((new_h,Op_r[i][j]))

        used_us, used_vs = set(), set()
        for i in u_cover:
            vert = None
            for j in bigraph.adj_u[i]:
                if vert == None:
                    vert = Vertex((parent, current_node), [])
                    
                    new_u_list = ulist[i]
                    new_v_list = vlist[j]
                    if i in used_us:
                        new_u_list = [(self._copy_node(u, current_node, parent, current_node),lam) for u, lam in ulist[i]]
                    if j in used_vs:
                        new_v_list = [(self._copy_node(v, current_node, parent, parent),lam) for v,lam in vlist[j]]
                    
                    for u_node,lam in new_u_list:
                        u_node.gamma_coeff = "Free"
                        u_node.lambda_coeff = lam

                    
                    for v,lam in new_v_list:
                        if isinstance(edges_enumerated[(i,j)], tuple):
                            lambda_temp, gamma_temp = edges_enumerated[(i,j)]
                            v.gamma_coeff = gamma_temp
                            v.lambda_coeff = lambda_temp * lam
                        else:
                            v.lambda_coeff = edges_enumerated[(i,j)] * lam
                            v.gamma_coeff = "Free"


                    vert.add_hyperedges([t[0] for t in new_u_list] + [t[0] for t in new_v_list])
                    self.vertex_colls[(parent,current_node)].contained_vertices.append(vert)
                    
                else:
                    new_v_list = vlist[j]
                    if j in used_vs:
                        new_v_list = [(self._copy_node(v, current_node, parent, parent), lam) for v,lam in vlist[j]]

                    for v, lam in new_v_list:
                        if isinstance(edges_enumerated[(i,j)], tuple):
                            lambda_temp, gamma_temp = edges_enumerated[(i,j)]
                            v.gamma_coeff = gamma_temp
                            v.lambda_coeff = lambda_temp * lam
                        else:
                            v.lambda_coeff = edges_enumerated[(i,j)] * lam
                            v.gamma_coeff = "Free"
                    vert.add_hyperedges([t[0] for t in new_v_list])


                used_vs.add(j) 
                edges.remove((i, j))
            
            used_us.add(i)

        for j in v_cover:
            vert = None
            for i in bigraph.adj_v[j]:
                if (i, j) not in edges:
                    continue

                if vert == None:
                    vert = Vertex((parent, current_node), [])
                    
                    new_u_list = ulist[i]
                    new_v_list = vlist[j]
                    if i in used_us:
                        new_u_list = [(self._copy_node(u, current_node, parent, current_node),lam) for u,lam in ulist[i]]
                    if j in used_vs:
                        new_v_list = [(self._copy_node(v, current_node, parent, parent),lam) for v,lam in vlist[j]]
                    
                    for v,lam in new_v_list:
                        v.gamma_coeff = "Free"
                        v.lambda_coeff = lam

                    for u, lam in new_u_list:
                        if isinstance(edges_enumerated[(i,j)], tuple):
                            lambda_temp, gamma_temp = edges_enumerated[(i,j)]
                            u.gamma_coeff = gamma_temp
                            u.lambda_coeff = lambda_temp * lam
                        else:
                            u.lambda_coeff = edges_enumerated[(i,j)] * lam
                            u.gamma_coeff = "Free"


                    vert.add_hyperedges([t[0] for t in new_u_list] + [t[0] for t in new_v_list])
                    self.vertex_colls[(parent,current_node)].contained_vertices.append(vert)

                else:
                    new_u_list = ulist[i]
                    if i in used_us:
                        new_u_list = [(self._copy_node(u, current_node, parent, current_node), lam) for u, lam in ulist[i]]

                    for u,lam in new_u_list:
                        if isinstance(edges_enumerated[(i,j)], tuple):
                            lambda_temp, gamma_temp = edges_enumerated[(i,j)]
                            u.gamma_coeff = gamma_temp
                            u.lambda_coeff = lambda_temp * lam
                        else:
                            u.lambda_coeff = edges_enumerated[(i,j)] * lam
                            u.gamma_coeff = "Free"
                    vert.add_hyperedges([t[0] for t in new_u_list])


                edges.remove((i, j))
                used_us.add(i) 
            used_vs.add(j)
            
    def combine_subtrees(self, local_hyperedges, parent):
        """
        Checks if the hyperedges in the local_hyperedges list can be combined 
        and when it is possible, combines them.

        Combining two hyperedge is basically removing one of the subtree of the hyperedge
        and connecting deleted subtree's vertex to the remaining hyperedge.
        """
        combined = {}

        for element2 in local_hyperedges:

            # Checking if two hyperedges are suitable for combining.
            # It is enough to check hashes as hashes are containing unique information 
            # of the hyperedge and all of the children hyperedges till leaves, aka subtree.
            if element2.hash in combined:

                element1 = combined[element2.hash]

                # Find the vertex to be deleted and the vertex to be kept.
                del_vertex = element2.find_vertex(parent)
                keep_vertex = element1.find_vertex(parent)
                
                # Erase the subtree of the element2 hyperedge from the state diagram
                self.erase_subtree(element2, erased=[del_vertex])

                # Find the hyperedges that the del_vertex is connected to in the parent site.
                fathers_del = del_vertex.get_hyperedges_for_one_node_id(parent)

                # Remove the del_vertex from the vertex collection
                self.vertex_colls[del_vertex.corr_edge].contained_vertices.remove(del_vertex)
            
                # Reconnect hyperedges of the del_vertex on the parent site to the keep_vertex. 
                for father in fathers_del:
                    father.vertices.remove(del_vertex)
                    keep_vertex.add_hyperedge(father)
                
            else:
                combined[element2.hash] = element2

    def erase_subtree(self, start_edge, erased=None):
        """
        Erases the subtree of a hyperedge from the state diagram recursively.
        """

        if erased is None:
            erased = []  # Tracks visited nodes to avoid cycles

        # Remove the hyperedge from the hyperedge collection
        for i in range(len(self.hyperedge_colls[start_edge.corr_node_id].contained_hyperedges)):
            if self.hyperedge_colls[start_edge.corr_node_id].contained_hyperedges[i].identifier == start_edge.identifier:
                self.hyperedge_colls[start_edge.corr_node_id].contained_hyperedges.pop(
                    i)
                break

        for vertex in start_edge.vertices:
            # Check if the vertex has been visited to avoid processing the same edge multiple times
            if vertex not in erased:

                erased.append(vertex)
                self.vertex_colls[vertex.corr_edge].contained_vertices.remove(
                    vertex)

                for edge in vertex.hyperedges:
                    if edge != start_edge:
                        self.erase_subtree(edge, erased)

    @classmethod
    def get_state_diagram_compound(cls, state_diagrams):
        """
        Forms a compound state diagram from a list of state diagrams.
        """

        state_diagram = None

        for term in state_diagrams:
            if state_diagram != None:
                state_diagram = cls.sum_states(
                    state_diagram, term, term.reference_tree)
            else:
                state_diagram = term

        return state_diagram

    @classmethod
    def sum_states(cls, s1, s2, ref_tree):
        """
        Combines two state diagrams into a single one and returns it. 
        """

        state_diag = cls(ref_tree)

        for n1, h1 in s1.hyperedge_colls.items():
            if n1 in s2.hyperedge_colls:
                state_diag.hyperedge_colls[n1] = HyperEdgeColl(
                    n1, h1.contained_hyperedges + s2.hyperedge_colls[n1].contained_hyperedges)
            else:
                state_diag.hyperedge_colls[n1] = h1
        for n2, h2 in s2.hyperedge_colls.items():
            if n2 not in state_diag.hyperedge_colls:
                state_diag.hyperedge_colls[n2] = h2

        for n1, h1 in s1.vertex_colls.items():
            if n1 in s2.vertex_colls:
                state_diag.vertex_colls[n1] = VertexColl(
                    n1, h1.contained_vertices + s2.vertex_colls[n1].contained_vertices)
            else:
                state_diag.vertex_colls[n1] = h1
        for n2, h2 in s2.vertex_colls.items():
            if n2 not in state_diag.vertex_colls:
                state_diag.vertex_colls[n2] = h2

        return state_diag

    @classmethod
    def get_state_diagrams(cls, hamiltonian, ref_tree):
        """
        Creates single term diagrams for each term in the Hamiltonian.
        Calculates hash values for each state diagram.
        """

        state_diagrams = []

        for term in hamiltonian.terms:
            state_diagram = cls.from_single_term(term, ref_tree)
            state_diagrams.append(state_diagram)

            state_diagram.calculate_hashes(
                ref_tree.nodes[ref_tree.root_id], ref_tree)

        return state_diagrams

    def calculate_hashes(self, node, ref_tree):
        """
        Calculates and returns the hash of all of the hyperedges in the state diagram recursively. 
        Hash is formed by the label of the hyperedge and concatenation of the hashes of its children.
        """
        # Base case: if the node is None, just return
        if node is None:
            return ""
        # Process all children first
        children_hash = ""
        for child_id in node.children:
            children_hash += self.calculate_hashes(
                ref_tree.nodes[child_id], ref_tree)
        # Process the current node (parent node is processed after its children)
        return self.hyperedge_colls[node.identifier].contained_hyperedges[0].calculate_hash(children_hash)

    @classmethod
    
    def _remove_hyperedge_from_diagram(self, element):
        """
        Removes a hyperedge from the state diagram checking its identifier.
        """
        for i in range(len(self.hyperedge_colls[element.corr_node_id].contained_hyperedges)):
            if self.hyperedge_colls[element.corr_node_id].contained_hyperedges[i].identifier == element.identifier:
                self.hyperedge_colls[element.corr_node_id].contained_hyperedges.pop(
                    i)
                break

    @classmethod
    def _remove_hyperedge_from_vertex(cls,vert,element):
        for i in range(len(vert.hyperedges)):
            if vert.hyperedges[i].identifier == element.identifier:
                vert.hyperedges.pop(i)
                break
    
    def _copy_node(self, u_temp, current_node, parent, goal_site):

        new_h = HyperEdge(goal_site, u_temp.label, [])
        new_h.set_hash(u_temp.hash)
        new_h.lambda_coeff = u_temp.lambda_coeff
        #new_h.gamma_coeff = u_temp.gamma_coeff

        # Add vertices to new hyperedge unrelated to current node-parent
        for v in u_temp.vertices:
            if not(v.corr_edge == (current_node,parent) or v.corr_edge == (parent,current_node)):
                new_h.add_vertex(v)
        self.add_hyperedge(new_h)
        return new_h
    
    @classmethod
    def _compare_lists_by_identity(cls, list1, list2):
        # Check if the lengths are the same
        if len(list1) != len(list2):
            return False
        # Compare the identity (memory address) of each object in the lists
        return all(id(obj1) == id(obj2) for obj1, obj2 in zip(list1, list2))