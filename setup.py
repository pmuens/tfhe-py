from setuptools import setup

setup(
    name="tfhe-py",
    version="0.1.0",
    description='Python implementation of the Fully Homomorphic Encryption scheme "TFHE"',
    url="http://github.com/pmuens/tfhe-py",
    author="Philipp Muens, Bogdan Opanchuk",
    author_email="philipp@muens.io, bogdan@nucypher.com",
    license="MIT",
    packages=["tfhe"],
    install_requires=["numpy>=1.24.3"],
    zip_safe=True,
)
