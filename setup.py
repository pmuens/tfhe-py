from setuptools import setup

setup(
    name="tfhe-py",
    version="0.1.0",
    description='A Python of "Fully Homomorphic Encryption over the Torus"',
    url="http://github.com/pmuens/tfhe-py",
    author="Philipp Muens, Bogdan Opanchuk",
    author_email="philipp@muens.io, bogdan@nucypher.com",
    license="MIT",
    packages=["tfhe"],
    install_requires=["numpy>=1.24.3"],
    zip_safe=True,
)
