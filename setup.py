from setuptools import setup, find_packages

setup(
    name="synapto-llm",
    version="0.2.0",
    description="Synaptic Weight Eviction engine for dynamic LLM memory consolidation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Bodya",
    url="https://github.com/Bodya3101/synapto",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "bitsandbytes>=0.43.0",
        "safetensors>=0.4.0",
        "accelerate>=0.28.0",
    ],
)