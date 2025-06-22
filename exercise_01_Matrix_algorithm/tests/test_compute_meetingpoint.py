import numpy as np
from .base_tests import UnitTest, CompositeTest, test_results_to_score

from exercise_code.meeting_point import meeting_point_linear

class MeetingPointOneDim(UnitTest):
    def __init__(self) -> None:

        self.output = None
        # Create the two subplanes in R^3
        self.c = np.array([1, 0, 0])
        self.a = np.array([0, 1, 0])
        self.b = np.array([0, 0, 1])
        PTS_a = [2*self.a+2*self.c, 3*self.a+self.c, 4*self.a+9*self.c, 5*self.a+self.c]
        PTS_b = [2*self.b+self.c, 3*self.b+4*self.c, 4*self.b+self.c, 5*self.b+2*self.c]

        self.PTS_a, self.PTS_b = np.array(PTS_a).T, np.array(PTS_b).T

    def test(self):
        self.output = meeting_point_linear([self.PTS_a, self.PTS_b])
        self.output[0]

        return np.allclose(self.output[:,0], self.c) or np.allclose(self.output[:,0], -self.c) 

    def define_success_message(self):
        return f"Congratulations: You passed the test case for the case of a line as intersection."

    def define_failure_message(self):
        return f"The output of the meeting point is incorrect (expected {self.c}, got {self.output[:,0]})."


class MeetingPointZeroDim(UnitTest):
    def __init__(self) -> None:
        # Create the two subplanes in R^3
        self.c = np.array([0, 0, 0])
        self.a = np.array([0, 1, 0])
        self.b = np.array([0, 0, 1])
        PTS_a = [2*self.a+2*self.c, 3*self.a+self.c, 4*self.a+9*self.c, 5*self.a+self.c]
        PTS_b = [2*self.b+self.c, 3*self.b+4*self.c, 4*self.b+self.c, 5*self.b+2*self.c]

        self.PTS_a, self.PTS_b = np.array(PTS_a).T, np.array(PTS_b).T

    def test(self):
        self.output = meeting_point_linear([self.PTS_a, self.PTS_b])
        self.output[0]
        return np.allclose(self.output[:,0], self.c)

    def define_success_message(self):
        return f"Congratulations: You passed the test case for the case of only zero as intersection."

    def define_failure_message(self):
        return f"The output of the meeting point is incorrect (expected {self.c}, got {self.output[:,0]})."


class MeetingPointTest(CompositeTest):
    def define_tests(self):
        return [
            MeetingPointOneDim(),
            MeetingPointZeroDim()
        ]


def test_compute_meetingpoint():
    test = MeetingPointTest()
    return test_results_to_score(test())
