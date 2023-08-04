import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as req_file:
    install_requires = req_file.read().splitlines()

setuptools.setup(
    name="crowd-counting-inference",
    version="0.1.0",
    author="LINKS Foundation",
    author_email="info@linksfoundation.com",
    description="Crowd Counting Inference Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/links-ads/starlight-crowd-counting-inference",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["onnxruntime==1.15.1", "Pillow==8.4.0", "numpy==1.21.6"],
    entry_points={
        "console_scripts": [
            "crowd-counting-inference=crowd_counting_inference.__main__:main"
        ]
    },
    python_requires=">=3.6",
)
