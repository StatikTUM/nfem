import pytest
import nfem
import numpy as np
from numpy.testing import assert_almost_equal


@pytest.fixture
def model():
    model = nfem.Model()

    model.add_node(id='A', x=0, y=0, z=0, support='xyz')
    model.add_node(id='B', x=2, y=0, z=0, support='xyz')
    model.add_node(id='C', x=1, y=0.5, z=3, fz=-1)
    model.add_node(id='D', x=0, y=1, z=0, support='xyz')
    model.add_node(id='E', x=2, y=1, z=0, support='xyz')

    model.add_truss(id='1', node_a='A', node_b='C', youngs_modulus=1, area=1)
    model.add_truss(id='2', node_a='B', node_b='C', youngs_modulus=1, area=1)
    model.add_truss(id='3', node_a='D', node_b='C', youngs_modulus=1, area=1)
    model.add_truss(id='4', node_a='E', node_b='C', youngs_modulus=1, area=1)

    return model


def test_example(model):
    model = model.get_duplicate()
    model.predict_tangential(strategy='lambda', value=0.05)
    model.perform_non_linear_solution_step(strategy='load-control')

    model = model.get_duplicate()
    model.predict_tangential(strategy='lambda', value=0.1)
    model.perform_non_linear_solution_step(strategy='load-control')

    model = model.get_duplicate()
    model.predict_tangential(strategy='lambda', value=0.15)
    model.perform_non_linear_solution_step(strategy='load-control')

    model = model.get_duplicate()
    model.predict_tangential(strategy='lambda', value=0.2)
    model.perform_non_linear_solution_step(strategy='load-control')

    model = model.get_duplicate()
    model.predict_tangential(strategy='lambda', value=0.25)
    model.perform_non_linear_solution_step(strategy='load-control')

    model = nfem.bracketing(model)

    model = model.get_duplicate()

    model.predict_tangential(strategy='arc-length')

    model.combine_prediction_with_eigenvector(beta=1)

    desired_delta_u = 0.1
    current_delta_u = model.get_dof_increment(dof=('C', 'u'))
    model.scale_prediction(desired_delta_u/current_delta_u)

    model.perform_non_linear_solution_step(strategy='arc-length-control')

    for step in range(30):
        model = model.get_duplicate()
        model.predict_tangential(strategy='arc-length')
        model.perform_non_linear_solution_step(strategy='arc-length-control')

    assert_almost_equal(model.load_displacement_curve(('C', 'u')).T, [
        [0, 0.0],
        [0.0, 0.05],
        [0.0, 0.1],
        [0.0, 0.15],
        [0.0, 0.2],
        [0.0, 0.25],
        [0.0, 0.2978923263679038],
        [0.0, 0.32098534901280695],
        [0.0, 0.32240518342702057],
        [0.0, 0.3224495532524647],
        [0.0, 0.3224717381651868],
        [0.0, 0.32248283062154787],
        [0.0, 0.3224883768497284],
        [0.0, 0.32249114996381867],
        [0.10000000000000009, 0.32226596408505037],
        [0.19987301311489625, 0.3215743411132848],
        [0.2994740963342972, 0.3204231568536892],
        [0.39865863085050224, 0.31881365564724984],
        [0.49728256894163225, 0.31674797450689474],
        [0.5952026844219822, 0.3142289736679443],
        [0.6922768151188325, 0.3112602071401334],
        [0.7883641007337197, 0.3078459096465948],
        [0.8833252159181328, 0.3039909860843248],
        [0.9770225979836491, 0.2997010007978705],
        [1.0693206686112715, 0.2949821659761016],
        [1.1600860489515181, 0.28984132899116133],
        [1.2491877675605352, 0.2842859586590704],
        [1.3364974606827182, 0.2783241304614203],
        [1.421889564461413, 0.27196451079330974],
        [1.505241498733338, 0.2652163403149052],
        [1.5864338421376258, 0.2580894164900275],
        [1.6653504983454628, 0.2505940753978426],
        [1.741878853289974, 0.2427411729043282],
        [1.8159099233472817, 0.23454206527924484],
        [1.8873384944875609, 0.22600858934221374],
        [1.9560632524787014, 0.2171530422183856],
        [2.0219869042841054, 0.20798816078021165],
        [2.0850162908496275, 0.19852710084714453],
        [2.1450624915222667, 0.18878341620981362],
        [2.2020409203845372, 0.1787710375394311],
        [2.255871414823257, 0.16850425123702073],
        [2.306478316679636, 0.15799767827060285],
        [2.3537905463489786, 0.14726625304183258],
        [2.3977416702130467, 0.13632520231685333],
        [2.438269961796317, 0.1251900242493964],
    ])

    assert_almost_equal(model.load_displacement_curve(('C', 'v')).T, [
        [0.0, 0.0],
        [0.0, 0.05],
        [0.0, 0.1],
        [0.0, 0.15],
        [0.0, 0.2],
        [0.0, 0.25],
        [0.0, 0.2978923263679038],
        [0.0, 0.32098534901280695],
        [0.0, 0.32240518342702057],
        [0.0, 0.3224495532524647],
        [0.0, 0.3224717381651868],
        [0.0, 0.32248283062154787],
        [0.0, 0.3224883768497284],
        [0.0, 0.32249114996381867],
        [0.0, 0.32226596408505037],
        [0.0, 0.3215743411132848],
        [0.0, 0.3204231568536892],
        [0.0, 0.31881365564724984],
        [0.0, 0.31674797450689474],
        [0.0, 0.3142289736679443],
        [0.0, 0.3112602071401334],
        [0.0, 0.3078459096465948],
        [0.0, 0.3039909860843248],
        [0.0, 0.2997010007978705],
        [0.0, 0.2949821659761016],
        [0.0, 0.28984132899116133],
        [0.0, 0.2842859586590704],
        [0.0, 0.2783241304614203],
        [0.0, 0.27196451079330974],
        [0.0, 0.2652163403149052],
        [0.0, 0.2580894164900275],
        [0.0, 0.2505940753978426],
        [0.0, 0.2427411729043282],
        [0.0, 0.23454206527924484],
        [0.0, 0.22600858934221374],
        [0.0, 0.2171530422183856],
        [0.0, 0.20798816078021165],
        [0.0, 0.19852710084714453],
        [0.0, 0.18878341620981362],
        [0.0, 0.1787710375394311],
        [0.0, 0.16850425123702073],
        [0.0, 0.15799767827060285],
        [0.0, 0.14726625304183258],
        [0.0, 0.13632520231685333],
        [0.0, 0.1251900242493964],
    ])

    assert_almost_equal(model.load_displacement_curve(('C', 'w')).T, [
        [0, 0.0],
        [-0.04666015874231988, 0.05],
        [-0.0956838865817482, 0.1],
        [-0.14742088617765203, 0.15],
        [-0.20231563385787998, 0.2],
        [-0.26094659903397854, 0.25],
        [-0.321327693686853, 0.2978923263679038],
        [-0.3521870402504428, 0.32098534901280695],
        [-0.354133144656259, 0.32240518342702057],
        [-0.35419396041894036, 0.3224495532524647],
        [-0.35422436830028126, 0.3224717381651868],
        [-0.3542395722409517, 0.32248283062154787],
        [-0.35424717421128715, 0.3224883768497284],
        [-0.35425097519645465, 0.32249114996381867],
        [-0.3561385129325165, 0.32226596408505037],
        [-0.3618078202933881, 0.3215743411132848],
        [-0.37125054759973874, 0.3204231568536892],
        [-0.38445410516317136, 0.31881365564724984],
        [-0.401400483081511, 0.31674797450689474],
        [-0.4220660567755168, 0.3142289736679443],
        [-0.44642160086318494, 0.3112602071401334],
        [-0.47443233950828656, 0.3078459096465948],
        [-0.5060580139394726, 0.3039909860843248],
        [-0.5412529628653777, 0.2997010007978705],
        [-0.5799662142188273, 0.2949821659761016],
        [-0.6221415873032097, 0.28984132899116133],
        [-0.6677178045814469, 0.2842859586590704],
        [-0.7166286123863062, 0.2783241304614203],
        [-0.768802909831749, 0.27196451079330974],
        [-0.8241648851976162, 0.2652163403149052],
        [-0.8826341590542741, 0.2580894164900275],
        [-0.9441259333936416, 0.2505940753978426],
        [-1.0085511460397845, 0.2427411729043282],
        [-1.075816629626223, 0.23454206527924484],
        [-1.1458252744481994, 0.22600858934221374],
        [-1.2184761945259015, 0.2171530422183856],
        [-1.2936648962484762, 0.20798816078021165],
        [-1.3712834490079309, 0.19852710084714453],
        [-1.451220657275905, 0.18878341620981362],
        [-1.5333622336240815, 0.1787710375394311],
        [-1.6175909722398398, 0.16850425123702073],
        [-1.7037869225418638, 0.15799767827060285],
        [-1.791827562555, 0.14726625304183258],
        [-1.8815879717589643, 0.13632520231685333],
        [-1.9729410031807817, 0.1251900242493964],
    ])

    assert_almost_equal([m.det_k for m in model.get_model_history()[1:]], [
        0.0014872416447633016,
        -0.0003440253338515905,
        -0.0014393913796696158,
        -0.0018245281870440124,
        -0.0015325374880115978,
        -0.0006584274239246974,
        -4.426698333056571e-05,
        -2.4907570224368014e-06,
        -1.1799126768587783e-06,
        -5.243698733476725e-07,
        -1.9656832266850807e-07,
        -3.266001118448634e-08,
        4.929602844666738e-08,
        1.3568079255375943e-05,
        5.423195344124841e-05,
        0.00012178046417978568,
        0.00021582886171580306,
        0.00033584435349276476,
        0.0004811442673377249,
        0.0006508993658109814,
        0.000844138585307853,
        0.0010597547572992155,
        0.0012965111882055403,
        0.0015530490283432927,
        0.0018278953701638925,
        0.0021194720160152046,
        0.002426104853315921,
        0.0027460337723332643,
        0.0030774230593270057,
        0.0034183721958856174,
        0.0037669269939084303,
        0.004121090994901155,
        0.004478837062037908,
        0.004838119093776973,
        0.00519688378866366,
        0.00555308239226749,
        0.005904682358939464,
        0.006249678863183665,
        0.006586106097870942,
        0.006912048299224686,
        0.007225650441435123,
        0.007525128546858328,
        0.0078087795609874,
        0.008074990744704328,
    ])
