import pytest
import nfem
import numpy as np
from numpy.testing import assert_almost_equal


@pytest.fixture
def model():
    youngs_modulus = 1
    area = 1
    height = 1

    model = nfem.Model()

    model.add_node(id='A', x=0, y=0, z=0, support='xyz')
    model.add_node(id='B', x=1, y=height, z=0, support='z', fy=-1)
    model.add_node(id='C', x=2, y=0, z=0, support='xyz')

    model.add_truss(id='1', node_a='A', node_b='B', youngs_modulus=youngs_modulus, area=area)
    model.add_truss(id='2', node_a='B', node_b='C', youngs_modulus=youngs_modulus, area=area)

    return model


def test_lpb(model):
    model = model.get_duplicate()
    model.load_factor = 0.01
    model.perform_linear_solution_step()
    model.solve_linear_eigenvalues()

    assert_almost_equal(model.load_displacement_curve(('B', 'v')).T, [
        [0, 0.0],
        [-0.0141421356237309, 0.01],
    ])


def test_nonlinear_solution(model):
    model = model.get_duplicate()

    model.predict_tangential(strategy="lambda", value=0.01)

    model.perform_non_linear_solution_step(
        strategy="load-control",
        solve_det_k=True,
        solve_attendant_eigenvalue=True,
    )

    model = nfem.bracketing(model, max_steps=500)

    assert_almost_equal(model.load_displacement_curve(('B', 'v')).T, [
        [0, 0.0],
        [-0.01445385294069157, 0.01],
        [-0.02911419289950612, 0.019696525414532585],
        [-0.043979938731305124, 0.029077033593218927],
        [-0.05904942872433283, 0.03812870996579104],
        [-0.07432015374710399, 0.04683888976409461],
        [-0.08978875255985252, 0.05519511147060876],
        [-0.1054509757637917, 0.06318524366071163],
        [-0.12130166206965876, 0.07079762108851484],
        [-0.13733472922201573, 0.07802118750853695],
        [-0.15354318163094482, 0.08484564206217667],
        [-0.16991913631027034, 0.09126158545799541],
        [-0.1864538681017618, 0.09726066170487259],
        [-0.2031378744006196, 0.1028356908762965],
        [-0.21996095873061206, 0.10798078834272727],
        [-0.23691233160622194, 0.11269146614060233],
        [-0.25398072623477297, 0.11696471266094366],
        [-0.2711545258287691, 0.12079904761942606],
        [-0.28842189868766566, 0.12419455026641102],
        [-0.30577093682474543, 0.12715285993845618],
        [-0.3231897937924053, 0.12967714925393567],
        [-0.3406668175050561, 0.13177207142031067],
        [-0.3581906742529073, 0.13344368416037664],
        [-0.3757504606983879, 0.1346993536066094],
        [-0.39333580138917257, 0.1355476421073386],
        [-0.41093693013882837, 0.13599818421406865],
        [-0.4197406706107589, 0.1360775725019799],
        [-0.4219417023369679, 0.13607948607622206],
        [-0.4224919602685202, 0.1360799644697826],
        [-0.42262952475140825, 0.13608008406817274],
        [-0.4226467203117692, 0.1360800990179715],
        [-0.42264886975681426, 0.13608010088669636],
        [-0.4226494071180755, 0.13608010135387757],
        [-0.42264967579870616, 0.1360801015874682],
        [-0.4226498101390215, 0.1360801017042635],
    ])
