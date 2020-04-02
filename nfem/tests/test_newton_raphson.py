import pytest
import numpy as np

from nfem import newton_raphson_solve


@pytest.fixture
def calculate_system():
    def f(x):
        rhs = np.array([np.sin(x[0])])
        lhs = np.array([np.cos(x[0])])

        return lhs, rhs

    return f


def test_not_converged_raises(calculate_system):
    with pytest.raises(RuntimeError):
        newton_raphson_solve(calculate_system, x_initial=[1], max_iterations=1)
