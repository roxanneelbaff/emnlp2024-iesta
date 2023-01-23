import setuptools
from setuptools import setup

setup(
    name='iesta',
    version='1.2',
    description='Python package for Conf 2023 for INEFF-IFFE Style Transfer',
    # url='',
    author='Roxanne El Baff',
    author_email='roxanne.elbaff@dlr.de',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=[
                      'tomli >= 1.1.0',
                      'spacy>=3.4',
                      # for spacy - it uses soft_unicode which was removed from version > 2.0.1
                      'markupsafe<=2.1.1',
                      'empath>=0.89',
                      'tqdm',
                      'nlpaf>=2.6.0',
                      'seaborn'
                      ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
    ],
    zip_safe=False,
    include_package_data=True,
    package_data={"../data": ["*"]},
)
