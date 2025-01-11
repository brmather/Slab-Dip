from setuptools import setup, find_packages
import io
from os import path

this_directory = path.abspath(path.dirname(__file__))
with io.open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

if __name__ == "__main__":
    setup(
        name='slabdip',
        version='4.2.2',
        description='Method to calculate slab dip using simple plate kinematic parameters',
        long_description=long_description,
        long_description_content_type='text/markdown',
        author='Ben Mather',
        author_email='ben.mather@sydney.edu.au',
        url='https://github.com/brmather/Slab-Dip',
        install_requires=[
            'pandas',
            'numpy',
            'scikit-learn',
            'gplately >= 1.0'
        ],
        extras_require={'plotting': ['matplotlib', 'jupyter', 'cartopy']},
        packages=['slabdip'],
        package_data={'slabdip': ['data/subduction_data.csv']},
        include_package_data = True,
        classifiers     = ['Programming Language :: Python :: 2',
                           'Programming Language :: Python :: 2.6',
                           'Programming Language :: Python :: 2.7',
                           'Programming Language :: Python :: 3',
                           'Programming Language :: Python :: 3.3',
                           'Programming Language :: Python :: 3.4',
                           'Programming Language :: Python :: 3.5',
                           'Programming Language :: Python :: 3.6',
                           'Programming Language :: Python :: 3.7',
                           'Programming Language :: Python :: 3.8',
                           'Programming Language :: Python :: 3.9',
                           'Programming Language :: Python :: 3.10',
                           'Programming Language :: Python :: 3.11',
                           'Programming Language :: Python :: 3.12',
                           ]
        )
