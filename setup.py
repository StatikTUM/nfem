from setuptools import setup

setup(
    name='nfem',
    version='dev',
    description='NFEM Teaching Tools',
    url='https://github.com/ChairOfStructuralAnalysisTUM/nfem',
    author='Thomas Oberbichler, Armin Geiser, Klaus Sautter, Aditya Ghantasala',
    author_email='',
    license='',
    packages=['nfem'],
    python_requires='>3.6',
    install_requires=['numpy', 'matplotlib', 'pyqt5', 'scipy'],
)