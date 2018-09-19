from setuptools import setup, find_packages

setup(
    name='hpca',
    version='0.0.1',
    packages=find_packages(),
    install_requires=['numpy', 'scikit-learn'],
    url='https://github.com/tochikuji/Hierarchical-PPCA',
    license='Apache License',
    author='Aiga SUZUKI',
    author_email='tochikuji@gmail.com',
    description='Hierarchical Principal Component Analysis for the group-structured feature'
)
