# coding: utf-8

from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='rise_qsar',
      version='0.1',
      description='Project to train QSAR models',
      url='https://github.com/eryl/rise_qsar',
      author='Erik Ylipää',
      author_email='erik.ylipaa@ri.se',
      license='MIT',
      packages=['riseqsar'],
      install_requires=[],
      dependency_links=[],
      zip_safe=False)
