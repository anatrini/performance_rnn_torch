from setuptools import setup, find_packages

setup(
    name='performance_rnn_torch',
    version='1.0',
    packages=find_packages(),
    author='Alessandro Anatrini',
    author_email='alessandro.anatrini@hfmt-hamburg.de',
    description='PyTorch implementation of Magenta\'s popular Performance-RNN model',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy==1.26.3',
        'optuna==3.5.0',
        'pandas==2.1.4',
        'pretty_midi==0.2.10',
        'tensorboard==2.15.1',
        'tqdm==4.66.1'
    ]
)