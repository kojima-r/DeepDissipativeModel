import setuptools
import shutil
import os


# path = os.path.dirname(os.path.abspath(__file__))
# shutil.copyfile(f"{path}/dmm.py", f"{path}/dmm/dmm.py")

setuptools.setup(
    name="DDM-SSM",
    version="0.1",
    author="Ryosuke Kojima",
    author_email="kojima.ryosuke.8e@kyoto-u.ac.jp",
    description="deep dissipative neural state-space model library",
    long_description="deep dissipative neural state-space model library",
    long_description_content_type="text/markdown",
    url="https://github.com/kojima-r/DeepDissipativModel",
    packages=setuptools.find_packages(),
    entry_points={
        "console_scripts": [
            "ddm-train  = ddm.train:main",
            "ddm-pred   = ddm.pred:main",
            "ddm-linear = ddm.train_linear:main",
            "ddm-opt    = ddm.opt:main",
            "ddm-opt-post= ddm.opt_post:main",
            "ddm-plot   = ddm.plot:main",
            "ddm-eval   = ddm.eval:main",
            "ddm-config = ddm.config:main",
            "ddm-dataset= ddm.data.generator:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
