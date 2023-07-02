from __future__ import annotations
from collections import UserDict
from typing import List, Union, Dict

import numpy as np

class Operator:
    """
    An operator hold the information what operation to apply to which node in a TTN.
    """

    def __init__(self, operator: Union[str, np.ndarray], node_identifiers: List[str]):
        self.operator = operator
        self.node_identifiers = node_identifiers

class NumericOperator(Operator):
    """
    An operator that holds the operator associated with it directly as an array.
    """

    def __init__(self, operator: np.ndarray, node_identifiers: List[str]):
        super().__init__(operator, node_identifiers)

class SymbolicOperator(Operator):
    """
    An operator that holds the operator associated with it only as a symbolic value.
    That operator has to be converted before actual use.
    """

    def __init__(self, operator: str, node_identifiers: List[str]):
        super().__init__(operator, node_identifiers)

    def to_numeric(self, conversion_dict: Dict[str, np.ndarray]) -> NumericOperator:
        """
        Converts a symbolic operator into an equivalent numeric operator.

        Args:
            conversion_dict (Dict[str, np.ndarray]): The numeric values in the form of
             an array for the symbol.

        Returns:
            NumericOperator: The converted operator.
        """
        return NumericOperator(conversion_dict[self.operator],
                               self.node_identifiers)

class Term(UserDict):
    """
    Contains multiple single site matrices and the identifiers of the nodes they are applied
     to. It is basically a dictionary, where the keys are node identifiers and the values
     are the operators that should be applied to the node with that identifier.

    Represents: \bigotimes_{site_ids} operator_{site_id}
    """

    def __init__(self, matrix_dict: Dict[str, Union[np.ndarray, str]] = None):
        if matrix_dict is None:
            matrix_dict = {}
        super().__init__(matrix_dict)

    @classmethod
    def from_operators(cls, operators: List[Operator]) -> Term:
        """
        Obtain a term from a list of single site operators.
        """
        term = Term()
        for operator in operators:
            assert len(operator.node_identifiers) == 1
            term[operator.node_identifiers[0]] = operator.operator

    def into_operator(self,
                      conversion_dict: Union[Dict[str, np.ndarray], None] = None) -> NumericOperator:
        """
        Computes the numeric value of a term, by calculating their tensor product.
        If the term contains symbolic operators, a conversion dictionary has to be provided.

        Args:
            conversion_dict (Union[Dict[str, np.ndarray], None], optional): A dictionaty
             that contains the numeric values of all symbolic operators in this term.
             Defaults to None.

        Returns:
            NumericOperator: Numeric operator with the value of the computed tensor product of
                all contained terms.
        """
        total_operator = 1
        for operator in self.values():
            if isinstance(operator, str):
                if conversion_dict is not None:
                    operator = conversion_dict[operator]
                else:
                    errstr = "If the term contains symbolic operators, there must be a dictionary for conversion!"
                    raise TypeError(errstr)
            total_operator = np.kron(total_operator, operator)
        return NumericOperator(total_operator, list(self.keys()))           
