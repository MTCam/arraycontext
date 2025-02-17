#!/usr/bin/env python
# -*- coding: utf-8 -*-


def main():
    from setuptools import setup, find_packages

    version_dict = {}
    init_filename = "arraycontext/version.py"
    exec(
        compile(open(init_filename, "r").read(), init_filename, "exec"), version_dict
    )

    setup(
        name="arraycontext",
        version=version_dict["VERSION_TEXT"],
        description="Choose your favorite numpy-workalike",
        long_description=open("README.rst", "rt").read(),
        author="Andreas Kloeckner",
        author_email="inform@tiker.net",
        license="MIT",
        url="https://documen.tician.de/arraycontext",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Other Audience",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Natural Language :: English",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Information Analysis",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Software Development :: Libraries",
            "Topic :: Utilities",
        ],
        packages=find_packages(),
        python_requires="~=3.6",
        install_requires=[
            "numpy",

            # https://github.com/inducer/arraycontext/pull/147
            "pytools>=2022.1.3",

            "pytest>=2.3",
            "loopy>=2019.1",
            "dataclasses; python_version<'3.7'",
            "typing_extensions; python_version<'3.9'",
            "types-dataclasses",
        ],
        package_data={"arraycontext": ["py.typed"]},
    )


if __name__ == "__main__":
    main()
