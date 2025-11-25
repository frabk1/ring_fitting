"""
====================================

* **Filename**:          setup.py
* **Author**:            Frank Myhre
* **Description**:       Initialization function/imports.

====================================
"""
from setuptools import setup, find_packages

setup(
    name="ringfit",
    version="0.4.6",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "Pillow",
        "ehtim"
    ],
)
