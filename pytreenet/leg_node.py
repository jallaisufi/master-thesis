from __future__ import annotations
from .node import Node


class LegNode(Node):
    """ 
    The leg node contains a permutation that is responsible for everything
    that has to do with legs. On the other hand the superclass Node contains
    all that has to do with tree connectivity.

    The attribute `leg_permutation` is a list of integers with the same length
    as the associated tensor has dimensions. The associated permutation is such
    that the associated tensor transposed with it has the leg ordering:
        `(parent, child0, ..., childN-1, open_leg0, ..., open_legM-1)`
    Is compatible with `np.transpose`.
    So in the permutatio we have the format
        `[leg of tensor corr. to parent, leg of tensor corr. to child0, ...]`
    """

    def __init__(self, tensor: ndarray, tag=None, identifier=None):
        super().__init__(tag, identifier)

        self._leg_permutation = list(range(tensor.ndim))

    @property
    def leg_permutation(self):
        """
        Get the leg permutation, cf. class docstring.
        """
        return self._leg_permutation
