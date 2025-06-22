import numpy as np
from .base_tests import UnitTest, CompositeTest, test_results_to_score

from exercise_code.gaussian_elemination import swap_rows, multiply_row, add_row, perform_gaussian_elemination

class GaussianElimination_mat(UnitTest):
    def __init__(self) -> None:
        self.A = np.array([[1, 2], [3, 4]])
        self.A_inv = np.linalg.inv(self.A)

    def test(self):
        _, A_inv = perform_gaussian_elemination(self.A)
        return np.allclose(A_inv, self.A_inv)

    def define_success_message(self):
        return f"Congratulations: You passed the test case for the Gaussian elimination."

    def define_failure_message(self):
        return f"The output of the Gaussian elimination is incorrect (expected {self.A_inv}, got {self.output})."
    

class GaussianElimination_ops(UnitTest):
    def __init__(self) -> None:
        self.A = np.array([[1, 2], [3, 4]])
        self.A_inv = np.linalg.inv(self.A)

    def test(self):
        ops, _ = perform_gaussian_elemination(self.A)
        dim = self.A.shape[0]

        A_inv = np.eye(dim)
        for ops in ops:
            if ops == "DEGENERATE":
                self.output = "DEGENERATE"
                return False
            if ops == 'SOLUTION':
                self.output = A_inv
                return np.allclose(A_inv, self.A_inv)
            if ops[0] == 'S':
                i, j = ops[1], ops[2]
                A_inv = swap_rows(A_inv, i, j)
            elif ops[0] == 'M':
                i, scalar = ops[1], ops[2]
                A_inv = multiply_row(A_inv, i, scalar)
            elif ops[0] == 'A':
                i, j, scalar = ops[1], ops[2], ops[3]
                A_inv = add_row(A_inv, i, j, scalar)
        

    def define_success_message(self):
        return f"Congratulations: You passed the test case for the Gaussian elimination."

    def define_failure_message(self):
        return f"The output of the Gaussian elimination is incorrect (expected {self.A_inv}, got {self.output})."
    


class GaussianEliminationTest(CompositeTest):
    def define_tests(self):
        return [
            GaussianElimination_mat(),
            GaussianElimination_ops()
        ]


def test_gaussian_elimination():
    test = GaussianEliminationTest()
    return test_results_to_score(test())
