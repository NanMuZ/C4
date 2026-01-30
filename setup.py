from setuptools import setup, find_packages

setup(
        name='c4',
        version="0.0.1",
        description=(
            'C4'
        ),
        author='Anonymous',
        # maintainer='',
        packages=find_packages(),
        platforms=["all"],
        install_requires=[
            # "d4rl",
            # "gym",
            # "matplotlib",
            # "numpy",
            # "pandas",
            # "ray",
            # "torch",
            # "tensorboard",
            "tqdm",
        ]
    )
