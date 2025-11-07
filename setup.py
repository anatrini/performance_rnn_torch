"""Setup script for Performance RNN in PyTorch."""

from setuptools import setup, find_packages
from pathlib import Path
import re

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read version from package
init_file = (this_directory / "performance_rnn_torch" / "__init__.py").read_text(encoding='utf-8')
version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_file, re.M)
if version_match:
    __version__ = version_match.group(1)
else:
    raise RuntimeError("Unable to find version string.")

setup(
    name='performance_rnn_torch',
    version=__version__,
    packages=find_packages(exclude=['tests', 'tests.*', 'docs', 'scripts']),
    author='Alessandro Anatrini',
    author_email='alessandro.anatrini@hfmt-hamburg.de',
    description='PyTorch implementation of Google\'s Performance RNN for expressive piano generation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/anatrini/performance_rnn_torch',
    project_urls={
        'Bug Reports': 'https://github.com/anatrini/performance_rnn_torch/issues',
        'Source': 'https://github.com/anatrini/performance_rnn_torch',
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Sound/Audio :: MIDI',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='music generation, deep learning, rnn, midi, pytorch, performance-rnn',
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0,<2.0.0',
        'torch>=2.0.0',
        'pretty_midi>=0.2.9',
        'tqdm>=4.60.0',
        'tensorboard>=2.10.0',
        'optuna>=3.0.0',
        'pandas>=1.3.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
            'isort>=5.10.0',
            'pre-commit>=2.17.0',
        ],
        'docs': [
            'sphinx>=4.5.0',
            'sphinx-rtd-theme>=1.0.0',
            'sphinx-autodoc-typehints>=1.18.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'performance-rnn-train=scripts.train:main',
            'performance-rnn-generate=scripts.generate:main',
            'performance-rnn-preprocess=scripts.preprocess:main',
            'performance-rnn-optimize=scripts.optimization_routine:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)