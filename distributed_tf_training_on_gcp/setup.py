from setuptools import find_packages
from setuptools import setup

setup(
  name='trainer',
  version='1.5',
  packages=find_packages(),
  include_package_data=True,
  description='CIFAR distributed training'
)