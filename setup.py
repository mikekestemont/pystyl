try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
import sys

if sys.version_info[0] == 2:
    ete_requirement = 'ete2'
elif sys.version_info[0] == 3:
    ete_requirement = 'ete3'
    
setup(
    name='pystyl',
    version='0.0.1',
    author='Folgert Karsdorp and Mike Kestemont',
    author_email='mike.kestemont@gmail.com',
    packages=['app', 'PyStyl', 'PyStyl.clustering'],
    url='http://pypi.python.org/pypi/PyStyl/',
    license='LICENSE.txt',
    description='Computational Stylistics in Python',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'seaborn',
        'nltk',
        'matplotlib',
        ete_requirement,
        'flask',
        'scipy',
        'pandas',
        'bokeh',
    ],
)