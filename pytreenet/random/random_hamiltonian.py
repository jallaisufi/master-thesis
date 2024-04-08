from typing import List, Union

from numpy.random import default_rng
from numpy import ndarray, eye

from ..operators.tensorproduct import TensorProduct
from ..operators.hamiltonian import Hamiltonian
from .random_matrices import random_hermitian_matrix
from .random_ttns import random_big_ttns_two_root_children

def random_terms(
        num_of_terms: int, possible_operators: list, sites: List[str],
        min_strength: float = -1, max_strength: float = 1, min_num_sites: int = 2,
        max_num_sites: int = 2):
    """
    Creates random interaction terms.

    Parameters
    ----------
    num_of_terms : int
        The number of random terms to be generated.
    possible_operators : list of arrays
        A list of all possible single site operators. We assume all sites have
        the same physical dimension.
    sites : list of str
        A list containing the possible identifiers of site nodes.
    min_strength : float, optional
        Minimum strength an interaction term can have. The strength is
        multiplied to the first operator of the term. The default is -1.
    max_strength : float, optional
        Minimum strength an interaction term can have. The strength is
        multiplied to the first operator of the term. The default is 1.
    min_num_sites : int, optional
        The minimum numberof sites that can partake in a single interaction
        term. The default is 2.
    max_num_sites : int, optional
        The minimum numberof sites that can partake in a single interaction
        term. The default is 2.

    Returns
    -------
    rterms : list of dictionaries
        A list containing all the random terms.
    """

    rterms = []

    rng = default_rng()
    number_of_sites = rng.integers(low=min_num_sites, high=max_num_sites + 1,
                                   size=num_of_terms)
    strength = rng.uniform(low=min_strength, high=max_strength,
                           size=num_of_terms)

    for index, nsites in enumerate(number_of_sites):
        term = {}
        operator_indices = rng.integers(len(possible_operators), size=nsites)
        sites_list = []
        first = True

        for operator_index in operator_indices:

            operator = possible_operators[operator_index]

            if first:
                # The first operator has the interaction strength
                operator = strength[index] * operator
                first = False

            site = sites[rng.integers(len(sites))]
            # Every site should appear maximally once (Good luck)
            while site in sites_list:
                site = sites[rng.integers(len(sites))]

            term[site] = operator

        rterms.append(term)

    return rterms

def random_symbolic_terms(num_of_terms: int, possible_operators: List[ndarray], sites: List[str],
                          min_num_sites: int = 2,  max_num_sites: int = 2,
                          seed=None) -> List[TensorProduct]:
    """
    Creates random interaction terms.

    Parameters
    ----------
    num_of_terms : int
        The number of random terms to be generated.
    possible_operators : list of arrays
        A list of all possible single site operators. We assume all sites have
        the same physical dimension.
    sites : list of str
        A list containing the possible identifiers of site nodes.
    min_num_sites : int, optional
        The minimum numberof sites that can partake in a single interaction
        term. The default is 2.
    max_num_sites : int, optional
        The minimum numberof sites that can partake in a single interaction
        term. The default is 2.

    Returns
    -------
    rterms : list of dictionaries
        A list containing all the random terms.
    """
    rterms = []
    rng = default_rng(seed=seed)
    for _ in range(num_of_terms):
        number_of_sites = rng.integers(low=min_num_sites,
                                        high=max_num_sites + 1,
                                        size=1)
        term = random_symbolic_term(possible_operators, sites,
                                    num_sites=number_of_sites,
                                    seed=rng)
        while term in rterms:
            term = random_symbolic_term(possible_operators, sites,
                                        num_sites=number_of_sites,
                                        seed=rng)
        rterms.append(term)
    return rterms


def random_symbolic_term(possible_operators: List[str], sites: List[str],
                         num_sites: int = 2, seed: Union[int, None]=None) -> TensorProduct:
    """
    Creates a random interaction term.

    Args:
        possible_operators (list[ndarray]): Symbolic operators to choose from.
        sites (list[str]): Identifiers of the nodes to which they may be applied.
        num_sites (int, optional): Number of non-trivial sites in a term. Defaults to 2.
        seed (Union[int, None], optional): A seed for the random number generator. Defaults to None.

    Returns:
        TensorProduct: A random term in the form of a tensor product
    """
    rng = default_rng(seed=seed)
    rand_sites = rng.choice(sites, size=num_sites, replace=False)
    rand_operators = rng.choice(possible_operators, size=num_sites)
    return TensorProduct(dict(zip(rand_sites, rand_operators)))

def random_hamiltonian_compatible() -> Hamiltonian:
    """
    Generates a Hamiltonian that is compatible with the TTNS produced by
     `ptn.ttns.random_big_ttns_two_root_children`. It is already padded with
     identities.

    Returns:
        Hamiltonian: A Hamiltonian to use for testing.
    """
    conversion_dict = {chr(i): random_hermitian_matrix()
                       for i in range(65,70)} # A, B, C, D, E
    conversion_dict["I2"] = eye(2)
    terms = [TensorProduct({"site1": "A", "site2": "B", "site0": "C"}),
             TensorProduct({"site4": "A", "site3": "D", "site5": "C"}),
             TensorProduct({"site4": "A", "site3": "B", "site1": "A"}),
             TensorProduct({"site0": "C", "site6": "E", "site7": "C"}),
             TensorProduct({"site2": "A", "site1": "A", "site6": "D"}),
             TensorProduct({"site1": "A", "site3": "B", "site5": "C"})]
    ham = Hamiltonian(terms, conversion_dictionary=conversion_dict)
    ref_tree = random_big_ttns_two_root_children()
    return ham.pad_with_identities(ref_tree)
