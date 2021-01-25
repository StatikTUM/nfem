from setuptools import setup

import os
import re


def get_version(path):
    with open(os.path.join(os.path.dirname(__file__), path)) as file:
        data = file.read()
    regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    version_match = re.search(regex, data, re.M)
    if version_match is None:
        raise RuntimeError("Unable to find version string.")
    return version_match.group(1)


with open('README.md', 'r') as f:
    long_description = f.read()


setup(
    name='nfem',
    version=get_version(os.path.join('nfem', '__init__.py')),
    description='NFEM Teaching Tools',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/StatikTUM/nfem',
    author='Thomas Oberbichler, Armin Geiser, Klaus Sautter, Aditya Ghantasala, Mahmoud Zidan',
    author_email='',
    license='',
    packages=['nfem', 'nfem.visualization'],
    python_requires='>=3.6',
    install_requires=[
        'ipython',
        'mako',
        'matplotlib',
        'numpy',
        'plotly',
        'scipy',
    ],
    include_package_data=True,
    zip_safe=False,
)
