import pathlib

from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

dev_deps = [
    "flake8==4.0.1",
    "mypy==0.910",
    "pytest==6.2.5",
    "pdocs>=1.1.1",
    "pygments>=2.10.0"
]

setup(
    name="chowder_impl",
    version="1.0.0",
    description="Implementation of the CHOWDER model for histopathological image analysis.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/joubs/chowder_impl",
    author="Francois Joubert",
    author_email="fxa.joubert@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["chowder"],
    include_package_data=True,
    install_requires=['numpy>=1.21.2',
                      'torch>=1.9.1',
                      'torchvision>=0.10.1',
                      'matplotlib>=3.4.3',
                      'scikit-learn>=1.0',
                      'tensorboardX>=2.4',
                      'tensorboard>=2.7.0'],
    extras_require={
        "dev": dev_deps
    },
    entry_points={
        "console_scripts": [
            "chowder_train=chowder.__main__:main",
        ]
    }
)
