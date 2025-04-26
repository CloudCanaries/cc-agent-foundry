from setuptools import setup, find_packages

setup(
    name="cc-agent-foundry",
    version="0.1.0",
    description="Shared Agent Template Library Code",
    author="CloudCanaries",
    author_email="support@cloudcanaries.ai",
    url="https://github.com/CloudCanaries/cc-agent-foundry",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "requests>=2.0",
        "certifi>=14.05.14",
        "nulltype",
        "croniter>=1.0.15",
        "pyinstaller>=5.13.0",
        "python_dateutil>=2.5.3",
        "setuptools>=21.0.0",
        "urllib3>=1.15.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
    ],
)
