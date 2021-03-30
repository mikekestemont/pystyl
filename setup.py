try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
import sys

if sys.version_info[0] == 2:
    ete_requirement = 'ete2'
elif sys.version_info[0] == 3:
    ete_requirement = 'ete3==3.1.1'
    
setup(
    name='pystyl',
    version='0.0.2',
    author='Folgert Karsdorp and Mike Kestemont',
    author_email='mike.kestemont@gmail.com',
    packages=['app', 'pystyl', 'pystyl.clustering'],
    package_dir={'pystyl': 'pystyl'},
    package_data={'pystyl': ['pronouns/*.txt']},
    include_package_data=True,
    url='http://pypi.python.org/pypi/pystyl/',
    scripts=['pystyl/bin/pystyl'],
    license='LICENSE.txt',
    description='Computational Stylistics in Python',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy==1.20.1',
        'scikit-learn==0.24.1',
        'seaborn==0.11.1',
        'nltk==3.5',
        'matplotlib==3.4.0',
        ete_requirement,
        'flask',
        'scipy==1.6.2',
        'pandas==1.2.3',
        'bokeh==2.3.0',
        'dendropy',
    ],
)
