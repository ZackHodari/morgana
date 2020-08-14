from setuptools import setup

setup(
    name='morgana',
    version='0.3.0',
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
        'torch>1.2',
        'tqdm',
    ],
    packages=['morgana'])

