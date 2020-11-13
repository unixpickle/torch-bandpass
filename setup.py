from setuptools import setup

setup(
    name="torch-bandpass",
    version="1.0.0",
    description="A PyTorch implementation of the Prism filter for Transformers",
    url="https://github.com/unixpickle/torch-bandpass",
    author="Alex Nichol",
    author_email="unixpickle@gmail.com",
    license="BSD",
    packages=["torch_bandpass"],
    install_requires=["numpy", "torch"],
)
