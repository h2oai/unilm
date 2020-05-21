from setuptools import setup

# read version
about_info = {}
with open("version.txt", "r") as v:
    exec(v.read(), about_info)
# read long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="h2o_layoutlm",
    version=about_info["version"],
    description="H2O.ai port of Microsoft LayoutLM Document entity recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.h2o.ai",
    author="H2O.ai",
    author_email="team@h2o.ai",
    license="MIT License",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="",
    packages=["layoutlm"],
    python_requires="~=3.6",
    install_requires=[
        "certifi==2020.4.5.1",
        "chardet==3.0.4",
        "click==7.1.2",
        "dataclasses==0.7; python_version < '3.7'",
        "filelock==3.0.12",
        "h5py==2.10.0",
        "idna==2.9",
        "joblib==0.15.1",
        "keras==2.3.1",
        "keras-applications==1.0.8",
        "keras-preprocessing==1.1.2",
        "numpy==1.18.4",
        "pillow==7.1.2",
        "pyyaml==5.3.1",
        "regex==2020.5.14",
        "requests==2.23.0",
        "sacremoses==0.0.43",
        "scipy==1.4.1",
        "sentencepiece==0.1.91",
        "seqeval==0.0.12",
        "six==1.14.0",
        "tokenizers==0.7.0",
        "tqdm==4.46.0",
        "transformers==2.9.1",
        "urllib3==1.25.9",
    ],
    dependency_links=[],
    # package_data={},
    # data_files=[],
    # entry_points={},
)
