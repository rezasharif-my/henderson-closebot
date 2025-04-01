from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='closebot',
    version="0.1",
    packages=find_packages(where="app"),
    package_dir={"": "app"},
    install_requires=requirements,
)