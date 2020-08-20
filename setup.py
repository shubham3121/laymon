#!/usr/bin/env python

"""The setup script."""

import os
from setuptools import setup, find_packages
from os.path import join


with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()


def get_requirements(filename):
    path = join(os.getcwd(), filename)
    with open(path, "rb") as fp:
        requirement_list = fp.read().decode("ascii").strip().split("\n")
    return requirement_list


requirements = get_requirements("pip_dep/requirements.txt")

test_requirements = get_requirements("pip_dep/requirements_dev.txt")

setup(
    author="Shubham Gupta",
    author_email="shubhamgupta3121@gmail.com",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="A python based package used to visualize the feature maps/ layers of a neural network",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="laymon",
    name="laymon",
    packages=find_packages(include=["laymon", "laymon.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/shubham3121/laymon",
    project_url={"Documentation": "https://laymon.readthedocs.io"},
    version="1.0.0",
    zip_safe=False,
)
