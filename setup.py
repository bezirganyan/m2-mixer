import os

from setuptools import setup, find_packages

setup(
    name="m2-mixer",
    py_modules=["datasets", "modules", "utils", "models"],
    version="1.0",
    description="",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        str(r)
        for r in open(os.path.join(os.path.dirname(__file__), "requirements.txt")).readlines()
    ],
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)
