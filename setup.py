from setuptools import setup, find_packages


install_requires = [
    'matplotlib',
    'numpy',
    'scipy>=1.0.0',
    'statsmodels'
]

setup(
    name='binarycauses',
    author='Samuel Markson',
    author_email='smarkson@alum.mit.edu',
    version="0.1",
    packages=find_packages(),
    install_requires=install_requires
)
