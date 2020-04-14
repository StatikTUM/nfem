import pytest
import nfem
from numpy.testing import assert_almost_equal


@pytest.fixture
def model():
    model = nfem.Model()

    model.add_node(id='A', x=0, y=0, z=0, support='xyz')
    model.add_node(id='B', x=1, y=3, z=0, support='z', fy=-1)
    model.add_node(id='C', x=2, y=0, z=0, support='xyz')

    model.add_truss(id='1', node_a='A', node_b='B', youngs_modulus=1, area=1)
    model.add_truss(id='2', node_a='B', node_b='C', youngs_modulus=1, area=1)

    return model


def test_example(model):
    model = model.get_duplicate()
    model.predict_tangential(strategy='lambda', value=0.05)
    model.perform_non_linear_solution_step(strategy='load-control')
    model = nfem.bracketing(model)

    assert_almost_equal(model.load_displacement_curve(('B', 'u')).T, [
        [0, 0.0],
        [0.0, 0.05],
        [0.0, 0.0962387777555058],
        [0.0, 0.13854081588970923],
        [0.0, 0.15816343630014382],
        [0.0, 0.16290636547143378],
        [0.0, 0.16524503046917205],
        [0.0, 0.16641436296804119],
        [0.0, 0.16699902921747578],
        [0.0, 0.16729136234219305],
        [0.0, 0.16732790398278272],
        [0.0, 0.16733018783531955],
    ])

    assert_almost_equal(model.load_displacement_curve(('B', 'v')).T, [
        [0, 0.0],
        [-0.09202342171627098, 0.05],
        [-0.18601750526312344, 0.0962387777555058],
        [-0.2818730181111735, 0.13854081588970923],
        [-0.33046270534515054, 0.15816343630014382],
        [-0.3426748353405946, 0.16290636547143378],
        [-0.34879353782279887, 0.16524503046917205],
        [-0.3518528890639008, 0.16641436296804119],
        [-0.35338256468445195, 0.16699902921747578],
        [-0.35414740249472754, 0.16729136234219305],
        [-0.3542430072210121, 0.16732790398278272],
        [-0.35424898251640524, 0.16733018783531955],
    ])

    model = model.get_duplicate()

    model.predict_tangential(strategy='arc-length')

    model.combine_prediction_with_eigenvector(beta=1.0)

    desired_delta_u = 0.2
    current_delta_u = model.get_dof_increment(dof=('B', 'u'))
    model.scale_prediction(desired_delta_u / current_delta_u)

    model.perform_non_linear_solution_step(strategy='arc-length-control')
    assert_almost_equal(model.load_displacement_curve(('B', 'u')).T, [
        [0, 0.0],
        [0.0, 0.05],
        [0.0, 0.0962387777555058],
        [0.0, 0.13854081588970923],
        [0.0, 0.15816343630014382],
        [0.0, 0.16290636547143378],
        [0.0, 0.16524503046917205],
        [0.0, 0.16641436296804119],
        [0.0, 0.16699902921747578],
        [0.0, 0.16729136234219305],
        [0.0, 0.16732790398278272],
        [0.0, 0.16733018783531955],
        [0.19985658896939396, 0.16685391359165272],
    ])

    assert_almost_equal(model.load_displacement_curve(('B', 'v')).T, [
        [0, 0.0],
        [-0.09202342171627098, 0.05],
        [-0.18601750526312344, 0.0962387777555058],
        [-0.2818730181111735, 0.13854081588970923],
        [-0.33046270534515054, 0.15816343630014382],
        [-0.3426748353405946, 0.16290636547143378],
        [-0.34879353782279887, 0.16524503046917205],
        [-0.3518528890639008, 0.16641436296804119],
        [-0.35338256468445195, 0.16699902921747578],
        [-0.35414740249472754, 0.16729136234219305],
        [-0.3542430072210121, 0.16732790398278272],
        [-0.35424898251640524, 0.16733018783531955],
        [-0.36180792864058997, 0.16685391359165272],
    ])

    for step in range(30):
        model = model.get_duplicate()
        model.predict_tangential(strategy='arc-length')
        model.perform_non_linear_solution_step(strategy='arc-length-control')

    assert_almost_equal(model.load_displacement_curve(('B', 'u')).T, [
        [0, 0.0],
        [0.0, 0.05],
        [0.0, 0.0962387777555058],
        [0.0, 0.13854081588970923],
        [0.0, 0.15816343630014382],
        [0.0, 0.16290636547143378],
        [0.0, 0.16524503046917205],
        [0.0, 0.16641436296804119],
        [0.0, 0.16699902921747578],
        [0.0, 0.16729136234219305],
        [0.0, 0.16732790398278272],
        [0.0, 0.16733018783531955],
        [0.19985658896939396, 0.16685391359165272],
        [0.39856667875356755, 0.16542242148337993],
        [0.5949906356239214, 0.163045833795918],
        [0.7880025258117109, 0.15973793405767941],
        [0.9764968467783044, 0.15551789363683483],
        [1.1593950767226238, 0.15041014564277722],
        [1.3356520067936053, 0.1444442266629032],
        [1.504261810897666, 0.137654587338225],
        [1.6642638130888763, 0.1300803739240164],
        [1.8147479183097346, 0.12176518321941962],
        [1.9548596782720202, 0.1127567933873729],
        [2.0838049702667996, 0.1031068732503044],
        [2.2008542724326587, 0.09287067264332573],
        [2.305346524282919, 0.08210669634249726],
        [2.3966925659244662, 0.07087636397102055],
        [2.4743781532651346, 0.05924365813271625],
        [2.537966549523105, 0.04727476284282178],
        [2.587100695488927, 0.03503769413435012],
        [2.6215049622641944, 0.02260192452712941],
        [2.6409864906702945, 0.010038002868460412],
        [2.6454361212845154, -0.0025828291000073167],
        [2.634828918250607, -0.015189022215975737],
        [2.6092242887837185, -0.02770910625905551],
        [2.568765698820955, -0.040072076087451544],
        [2.513679983744357, -0.05220777642448072],
        [2.4442762517111167, -0.06404728411503192],
        [2.3609443760487925, -0.07552328657356505],
        [2.264153072580063, -0.0865704550093954],
        [2.1544475577797235, -0.09712581084969887],
        [2.032446784454904, -0.10712908359577915],
        [1.8988402532612891, -0.11652305815519652],
    ])

    assert_almost_equal(model.load_displacement_curve(('B', 'v')).T, [
        [0, 0.0],
        [-0.09202342171627098, 0.05],
        [-0.18601750526312344, 0.0962387777555058],
        [-0.2818730181111735, 0.13854081588970923],
        [-0.33046270534515054, 0.15816343630014382],
        [-0.3426748353405946, 0.16290636547143378],
        [-0.34879353782279887, 0.16524503046917205],
        [-0.3518528890639008, 0.16641436296804119],
        [-0.35338256468445195, 0.16699902921747578],
        [-0.35414740249472754, 0.16729136234219305],
        [-0.3542430072210121, 0.16732790398278272],
        [-0.35424898251640524, 0.16733018783531955],
        [-0.36180792864058997, 0.16685391359165272],
        [-0.38444180632155733, 0.16542242148337993],
        [-0.4220189689282181, 0.163045833795918],
        [-0.4743214642109628, 0.15973793405767941],
        [-0.5410461674320661, 0.15551789363683483],
        [-0.6218067582053171, 0.15041014564277722],
        [-0.716136222647755, 0.1444442266629032],
        [-0.8234898486284612, 0.137654587338225],
        [-0.9432486797069499, 0.1300803739240164],
        [-1.0747233900924784, 0.12176518321941962],
        [-1.217158540784373, 0.1127567933873729],
        [-1.3697371760271047, 0.1031068732503044],
        [-1.5315857192656797, 0.09287067264332573],
        [-1.7017791287981712, 0.08210669634249726],
        [-1.8793462751345487, 0.07087636397102055],
        [-2.063275504497171, 0.05924365813271625],
        [-2.2525203557333495, 0.04727476284282178],
        [-2.446005400942634, 0.03503769413435012],
        [-2.6426321831433044, 0.02260192452712941],
        [-2.8412852271197595, 0.010038002868460412],
        [-3.040838102032766, -0.0025828291000073167],
        [-3.240159516293898, -0.015189022215975737],
        [-3.4381194264939023, -0.02770910625905551],
        [-3.633595142759576, -0.040072076087451544],
        [-3.825477412762447, -0.05220777642448072],
        [-4.01267646572351, -0.06404728411503192],
        [-4.194127996200587, -0.07552328657356505],
        [-4.368799065296867, -0.0865704550093954],
        [-4.535693894315242, -0.09712581084969887],
        [-4.693859522959468, -0.10712908359577915],
        [-4.842391301132138, -0.11652305815519652],
    ])

    model = nfem.bracketing(model, max_steps=150, raise_error=False)

    model = model.get_duplicate()
    model.predict_tangential(strategy='delta-dof', dof=('B', 'v'), value=-0.05)
    model.perform_non_linear_solution_step(strategy='displacement-control', dof=('B', 'v'))

    for step in range(5):
        model = model.get_duplicate()
        model.predict_tangential(strategy='arc-length')
        model.perform_non_linear_solution_step(strategy='arc-length-control')

    assert_almost_equal(model.load_displacement_curve(('B', 'u')).T, [
        [0, 0.0],
        [0.0, 0.05],
        [0.0, 0.0962387777555058],
        [0.0, 0.13854081588970923],
        [0.0, 0.15816343630014382],
        [0.0, 0.16290636547143378],
        [0.0, 0.16524503046917205],
        [0.0, 0.16641436296804119],
        [0.0, 0.16699902921747578],
        [0.0, 0.16729136234219305],
        [0.0, 0.16732790398278272],
        [0.0, 0.16733018783531955],
        [0.19985658896939396, 0.16685391359165272],
        [0.39856667875356755, 0.16542242148337993],
        [0.5949906356239214, 0.163045833795918],
        [0.7880025258117109, 0.15973793405767941],
        [0.9764968467783044, 0.15551789363683483],
        [1.1593950767226238, 0.15041014564277722],
        [1.3356520067936053, 0.1444442266629032],
        [1.504261810897666, 0.137654587338225],
        [1.6642638130888763, 0.1300803739240164],
        [1.8147479183097346, 0.12176518321941962],
        [1.9548596782720202, 0.1127567933873729],
        [2.0838049702667996, 0.1031068732503044],
        [2.2008542724326587, 0.09287067264332573],
        [2.305346524282919, 0.08210669634249726],
        [2.3966925659244662, 0.07087636397102055],
        [2.4743781532651346, 0.05924365813271625],
        [2.537966549523105, 0.04727476284282178],
        [2.587100695488927, 0.03503769413435012],
        [2.6215049622641944, 0.02260192452712941],
        [2.6409864906702945, 0.010038002868460412],
        [2.6454361212845154, -0.0025828291000073167],
        [2.634828918250607, -0.015189022215975737],
        [2.6092242887837185, -0.02770910625905551],
        [2.568765698820955, -0.040072076087451544],
        [2.513679983744357, -0.05220777642448072],
        [2.4442762517111167, -0.06404728411503192],
        [2.3609443760487925, -0.07552328657356505],
        [2.264153072580063, -0.0865704550093954],
        [2.1544475577797235, -0.09712581084969887],
        [2.032446784454904, -0.10712908359577915],
        [1.8988402532612891, -0.11652305815519652],
        [1.7543844008656149, -0.12525390950406756],
        [1.5998985689345355, -0.13327152236354442],
        [1.4362605623206344, -0.14052979343552624],
        [1.2644018097278016, -0.14698691364778935],
        [1.0853021456237748, -0.15260562781962048],
        [0.8999842380384913, -0.15735346918652315],
        [0.7095076929168014, -0.16120296633074216],
        [0.5149628716132231, -0.16413182030120735],
        [0.31746446347020907, -0.16612305048391818],
        [0.11814485546041165, -0.16716512035036624],
        [-0.08185260868885091, -0.16725192174611883],
        [-0.28138052873515074, -0.16638298886936925],
        [8.090826009166108e-09, -0.1537437489280311],
        [-1.7210677327739177e-12, -0.033072164676529806],
        [-4.440892098500626e-16, 0.11511672570077151],
        [0.0, 0.28653805707311225],
        [0.0, 0.476915783125192],
        [0.0, 0.6824872518397282],
    ])

    assert_almost_equal(model.load_displacement_curve(('B', 'v')).T, [
        [0, 0.0],
        [-0.09202342171627098, 0.05],
        [-0.18601750526312344, 0.0962387777555058],
        [-0.2818730181111735, 0.13854081588970923],
        [-0.33046270534515054, 0.15816343630014382],
        [-0.3426748353405946, 0.16290636547143378],
        [-0.34879353782279887, 0.16524503046917205],
        [-0.3518528890639008, 0.16641436296804119],
        [-0.35338256468445195, 0.16699902921747578],
        [-0.35414740249472754, 0.16729136234219305],
        [-0.3542430072210121, 0.16732790398278272],
        [-0.35424898251640524, 0.16733018783531955],
        [-0.36180792864058997, 0.16685391359165272],
        [-0.38444180632155733, 0.16542242148337993],
        [-0.4220189689282181, 0.163045833795918],
        [-0.4743214642109628, 0.15973793405767941],
        [-0.5410461674320661, 0.15551789363683483],
        [-0.6218067582053171, 0.15041014564277722],
        [-0.716136222647755, 0.1444442266629032],
        [-0.8234898486284612, 0.137654587338225],
        [-0.9432486797069499, 0.1300803739240164],
        [-1.0747233900924784, 0.12176518321941962],
        [-1.217158540784373, 0.1127567933873729],
        [-1.3697371760271047, 0.1031068732503044],
        [-1.5315857192656797, 0.09287067264332573],
        [-1.7017791287981712, 0.08210669634249726],
        [-1.8793462751345487, 0.07087636397102055],
        [-2.063275504497171, 0.05924365813271625],
        [-2.2525203557333495, 0.04727476284282178],
        [-2.446005400942634, 0.03503769413435012],
        [-2.6426321831433044, 0.02260192452712941],
        [-2.8412852271197595, 0.010038002868460412],
        [-3.040838102032766, -0.0025828291000073167],
        [-3.240159516293898, -0.015189022215975737],
        [-3.4381194264939023, -0.02770910625905551],
        [-3.633595142759576, -0.040072076087451544],
        [-3.825477412762447, -0.05220777642448072],
        [-4.01267646572351, -0.06404728411503192],
        [-4.194127996200587, -0.07552328657356505],
        [-4.368799065296867, -0.0865704550093954],
        [-4.535693894315242, -0.09712581084969887],
        [-4.693859522959468, -0.10712908359577915],
        [-4.842391301132138, -0.11652305815519652],
        [-4.9804381804016185, -0.12525390950406756],
        [-5.107207768515985, -0.13327152236354442],
        [-5.221971108141547, -0.14052979343552624],
        [-5.324067139495598, -0.14698691364778935],
        [-5.41290680590264, -0.15260562781962048],
        [-5.48797676166919, -0.15735346918652315],
        [-5.548842643123752, -0.16120296633074216],
        [-5.595151866138043, -0.16413182030120735],
        [-5.626635915864481, -0.16612305048391818],
        [-5.643112067834108, -0.16716512035036624],
        [-5.644484823109927, -0.16725192174611883],
        [-5.630746104195642, -0.16638298886936925],
        [-5.6807461041956415, -0.1537437489280311],
        [-5.940117169190734, -0.033072164676529806],
        [-6.184811252376262, 0.11511672570077151],
        [-6.413830833245308, 0.28653805707311225],
        [-6.627352773557165, 0.476915783125192],
        [-6.826289193547681, 0.6824872518397282],
    ])
