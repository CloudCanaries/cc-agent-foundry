from setuptools import setup, find_packages

setup(
    name="ccmodule",
    version="0.1.0",
    description="Shared Agent Template Library Code",
    author="CloudCanaries",
    author_email="support@cloudcanaries.ai",
    url="https://github.com/CloudCanaries/cc-agent-foundry",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "requests>=2.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
    ],
)
