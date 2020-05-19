import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="layoutlm", # Replace with your own username
    version="0.0.1",
    author="Lei Cui",
    author_email="fuwei@microsoft.com",
    description="Document entity recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/unilm",
    #packages=setuptools.find_packages(),
    packages=['layoutlm'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
