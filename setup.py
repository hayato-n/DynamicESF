# Author: Hayato Nishi
# License: BSD 3-Clause

# I refered mgwr(https://github.com/pysal/mgwr) to write this setup.py

from setuptools import setup

with open("./DynamicESF/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            VERSION = line.strip().split()[-1][1:-1]
            break

with open("./README.md") as f:
    readme = f.read()

with open("./requirements.txt", "r") as f:
    pkg_list = f.read().splitlines()


def setup_package():
    setup(
        name="DynamicESF",
        version=VERSION,
        description="DynamicESF: fast spatially and temporally varying coefficient model",
        long_description=readme,
        url="https://github.com/hayato-n/DynamicESF",
        download_url="https://github.com/hayato-n/DynamicESF",
        author="Hayato Nishi",
        author_email="hnishiua@gmail.com",
        python_requires=">=3.8",
        keywords="spatial statistics",
        license="BSE 3-Clause",
        install_requires=pkg_list,
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: GIS",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.7",
        ],
        packages=["DynamicESF"],
    )


if __name__ == "__main__":
    setup_package()
