from setuptools import setup, find_packages

setup(
    name="model_finetuning",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "model_finetuning": ["weights/*.pt"],
    },
    zip_safe=False,
)
