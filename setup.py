from distutils.core import setup

setup(
    name='PyStyl',
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
        'pylab',
        'matplotlib',
        'ete2', # or ete3
        'flask',
        'scipy',
        'pandas',
        'bokeh',
    ],
)