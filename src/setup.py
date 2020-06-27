from setuptools import setup, find_packages

setup(
    name='cil-road-extraction-project',
    version='1.0',
    packages=find_packages(exclude=[]),
    url='https://github.com/daniCh8/road-segmentation-eth-cil-2020',

    author='danich',

    python_requires='>=3.5',
    install_requires=[
            # Add external libraries here.
            'tensorflow-gpu==1.12.0',
            'scikit-image',
            'albumentations',
            'tqdm',
            'keras',
            'opencv-python',
            'numpy',
            'matplotlib'
    ]
)
