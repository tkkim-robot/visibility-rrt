from setuptools import setup, find_packages

setup(
    name='visibility_rrt',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'scipy',
        'cvxpy',
        'numpy==1.26.4',  # latest gurobipy version 11.0.2 is not compatible with numpy 2.0, but 11.0.3 will be
        'matplotlib',
        'gurobipy',
        'shapely'
    ],
    include_package_data=True,
    zip_safe=False,
)