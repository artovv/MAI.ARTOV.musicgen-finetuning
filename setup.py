# setup.py
from setuptools import setup, find_packages

setup(
    name="model_finetuning",
    version="0.1",
    package_data={
        "model_finetuning": ["weights/*.pt"],
    },
)
