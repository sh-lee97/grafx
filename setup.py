from pathlib import Path
from setuptools import setup, find_packages

setup(
    name="GRAFX",
    version="0.5.0",
    description="An Open-Source Library for Audio Processing Graphs in PyTorch",
    long_description="See docs",
    # long_description_content_type="text/markdown",
    package_dir={"": "src"},
    author="Sungho Lee",
    author_email="sh-lee@snu.ac.kr",
    python_requires=">=3.10.0",
    # url=URL,
    packages=find_packages(where="src"),
    include_package_data=True,
)
