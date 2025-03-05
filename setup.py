from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'ARS'
LONG_DESCRIPTION = 'Adaptive rejection sampling'

# Setting up
setup(
        name="ars", 
        version=VERSION,
        author="Soohyun Kim, Xiaotong Zhan, Yihong Zhu",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=["ars"],
        install_requires=['numpy', 'scipy', 'matplotlib', 'jax'],

        keywords=['python', 'sampling'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)
