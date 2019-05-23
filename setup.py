import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qrstreamer",
    version="0.1.0",
    author="Finian Blackett",
    author_email="spamsuckersunited@gmail.com",
    description="A module for streaming files into QR codes",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/ThePlasmaRailgun/QRStreamer",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        'qrcode',
    ],
    scripts=[
        'bin/qrstreamer.py'
    ]
)
