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
        "pillow~=7.1",
        "transformers<=2.10.0,>=2.3.0",
        "seqeval~=0.0.12",
        "pandas~=1.0",
        "torch~=1.5",
    ],
    dependency_links=[],
    # package_data={},
    # data_files=[],
    # entry_points={},
)
