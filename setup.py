from setuptools import setup, find_packages

setup(
    name="mc-vectorizer",
    version="0.1.0",
    author="Gannon Gonsiorowski",
    description="A Python library for vectorizing and reconstructing Minecraft structures using diffusion models.",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "umap-learn",
        "scipy",
        "scikit-learn",
        "sentence-transformers"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
