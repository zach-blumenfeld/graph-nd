# setup.py

from setuptools import setup, find_packages

setup(
    name="graph_nd",
    version="0.1.0",
    description="A Python package for building powerful end-to-end GraphRAG systems with a simple, intuitive API",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Zach Blumenfeld",
    author_email="zblumenf@gmail.com",
    url="https://github.com/zach-blumenfeld/graph-nd",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=[
        "neo4j",
        "langchain-core",
        "langchain-openai",
        "langgraph",
        "pandas",
        "python-dotenv",
        "PyPDF2",
        "tqdm",
        "PyYAML",
        "pydantic",
        "snowflake-snowpark-python[pandas]",
        "nest-asyncio",
    ],
    extras_require={
        "test": [  # Test-specific dependencies
            "unittest",  # Built-in library if using unittest (optional to specify here)
            "nbconvert",  # For running notebooks as tests
            "nbformat",  # Parsing Jupyter Notebooks
            "jupyter",  # Supporting Jupyter integration
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
    ],
)

