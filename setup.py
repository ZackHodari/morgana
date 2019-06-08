from setuptools import setup

setup(
    name='morgana',
    version='0.0.1',
    description='Toolkit for defining and training Text-to-Speech voices in PyTorch.',
    url='https://github.com/ZackHodari/morgana',
    author='Zack Hodari',
    author_email='zack.hodari@ed.ac.uk',
    license='MIT',
    install_requires=[
        'bandmat',
        'matplotlib',
        'numpy',
        'scipy',
        'tensorboardX',
        'torch',
        'tqdm',
    ],
    packages=['morgana'])

