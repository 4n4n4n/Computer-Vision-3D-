import numpy as np
from .base_tests import UnitTest, CompositeTest, test_results_to_score

from exercise_code.pseudo_inverse import solve_linear_equation_SVD
from exercise_code.null_space import get_null_vector
import numpy as np

def generate_plt_data():
    x_star = np.array([4, -3, 2, -1])
    b = np.ones(3)
    D = generate_matrix(x_star, b, eps=0.)
    x_hat, _ = solve_linear_equation_SVD(D, b)
    v = get_null_vector(D)[0]

    scalings = np.linspace(-100, 100, 200)
    values_norm = np.zeros_like(scalings)
    values_error = np.zeros_like(scalings)
    for i, l in enumerate(scalings): # i is the index and l is the actuall value
        # solutions are x_hat + l*v 

    ########################################################################
    # TODO:                                                                #
    # Generate data for the plot.                                          #
    # the scaling should be in [-100, 100] with 200 points.                #
    # for each scale, we want to know the norm of x (values_norm)          #
    # and the corresponding error (values_error) of D @ x - b.             #
    # Generate possible solutions along the null space of D.               #
    # Fill in the values vor values_norm and values_error.                 #
    #                                                                      #
    ########################################################################
        x = x_hat + v*l
        values_norm[i] = np.linalg.norm(x)
        err = np.matmul(D, x) - b
        values_error[i] = np.linalg.norm(err)

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return scalings, values_norm, values_error

def generate_matrix(x_star, b, eps=1e-4):
    '''
    Inputs:
    - x_star: numpy.ndarray, vector of shape (n,)
    - b: numpy.ndarray, vector of shape (m,)
    - eps: float, noise level
    Outputs:
    - D: numpy.ndarray, matrix of shape (m,n)
    
    '''
    m = len(b)
    n = len(x_star)
    D = np.random.randn(m, n)

    ########################################################################
    # TODO:                                                                #
    # Generate a matrix D such that D @ x_star = b.                        #
    #                                                                      #
    # Construct D[:,-1] such that D @ x_star = b.                          #
    ########################################################################


    for i in range(m):
        sum = 0
        for j in range(n-1):
            sum += D[i][j]*x_star[j]
        D[i][n-1] = (b[i]-sum)/x_star[n-1]

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    # add some noise
    D = D + eps * np.random.randn(m, n)

    
    return D # shape (m,n)

class MatrixConstructionTest(UnitTest):
    def __init__(self) -> None:
        return

    def test(self):
        m = 3
        b = np.ones(m)
        x_star = np.array([4, -3, 2, -1])
        D = generate_matrix(x_star,  b, eps=1e-4)
        b_ = D @ x_star
        return np.allclose(b_, b, rtol=1.e-1)
    
    def define_success_message(self):
        return f"Congratulations: Your matrix D is correct."
    
    def define_failure_message(self):
        return f"Failure: Your matrix D is incorrect."
    

    
class PseudoInverseTest1(UnitTest):
    def __init__(self) -> None:
        return

    def test(self):
        m = 3
        b = np.ones(m)
        x_star = np.array([4, -3, 2, -1])
        D = generate_matrix(x_star,  b, eps=1e-4)
        x_hat, _ = solve_linear_equation_SVD(D, b)
        self.v = get_null_vector(D)
        self.delta_x = (x_star - x_hat)
        self.delta_x = self.delta_x / np.linalg.norm(self.delta_x)
        return np.allclose(self.delta_x, self.v, rtol=1.e-1) or np.allclose(-self.delta_x, self.v, rtol=1.e-1)

    def define_success_message(self):
        return f"Congratulations: You passed the first test case for the pseudo inverse."

    def define_failure_message(self):
        return f"Failure: You failed the first test case for the pseudo inverse. v = {self.v}, delta_x = {self.delta_x}"


class PseudoInverseTest2(UnitTest):
    def __init__(self) -> None:
        return

    def test(self):
        m = 4
        b = np.ones(m)
        self.x_star = np.array([4, -3, 2, -1])
        D = generate_matrix(self.x_star,  b, eps=1e-4)
        self.D_inv = np.linalg.pinv(D)
        _, self.D_inv_hat = solve_linear_equation_SVD(D, b)
        return np.allclose(self.D_inv, self.D_inv_hat)

    def define_success_message(self):
        return f"Congratulations: You passed the second test case for the pseudo inverse."

    def define_failure_message(self):
        return f"Failure: You failed the second test case for the pseudo inverse. D_inv = {self.D_inv}, D_inv_hat = {self.D_inv_hat}"


class PseudoInverseTest3(UnitTest):
    def __init__(self) -> None:
        return

    def test(self):
        # print("test 3 Pinv")
        m = 3
        self.b = np.ones(m)
        self.x_star = np.array([4, -3, 2, -1])
        D = generate_matrix(self.x_star, self.b, eps=1e-4)
        
        x_hat, D_inv_hat = solve_linear_equation_SVD(D, self.b)
        v = get_null_vector(D)[0]
        # print("null \n", v)
        lbd = np.random.randn(1)
        self.b_ = D @ (x_hat + lbd * v) 
        # print("b' \n", self.b)
        return np.allclose(self.b_, self.b)

    def define_success_message(self):
        return f"Congratulations: You passed the third test case for the pseudo inverse."

    def define_failure_message(self):
        return f"Failure: You failed the third test case for the pseudo inverse. b = {self.b}, b_ = {self.b_}"


class PseudoInverseTest(CompositeTest):
    def define_tests(self):
        return [
            MatrixConstructionTest(),
            PseudoInverseTest1(),
            PseudoInverseTest2(),
            PseudoInverseTest3()
        ]

def test_pseudo_inverse():
    test = PseudoInverseTest()
    return test_results_to_score(test())
