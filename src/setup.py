from setuptools import setup, find_packages

setup(
    name='cil-road-extraction-project',
    version='1.0',
    packages=find_packages(exclude=[]),
    url='https://github.com/daniCh8/road-segmentation-eth-cil-2020',

    author='danich',

    python_requires='>=3.5',
    install_requires=[
            'tensorflow-gpu==2.5.1',
            'keras==2.3.1',
            'scikit-image',
            'albumentations',
            'tqdm',
            'scikit-learn',
            'opencv-python',
            'numpy',
            'pandas',
            'matplotlib',
            'pillow'
    ]
)
