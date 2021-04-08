from setuptools import setup, find_packages

setup(
    name='ALChemE',
    author='Pedro Seber, Jonathan Hung, and Arjun Bansal',
    author_email='pedro.seber@vanderbilt.edu',
    license='MIT',
    version='0.2.0',
    description='A software suite to assist in Chemical Engineering design',
    zip_safe=False,
    packages = find_packages(),
    install_requires = [
        'numpy',
        'pandas',
        'matplotlib',
        'unyt',
        'gekko',
        'tkinter',
        'joblib'
    ],
    )
