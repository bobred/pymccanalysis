from setuptools import setup
import os


lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = f"{lib_folder}/requirements.txt"
install_requires = [] # Here we'll add: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()
        
setup(name='pymccanalysis',
      version='1.0.0',
      description='Module that reads PTW mcc files from watertank scans or array files.',
      author='James Murphy',
      author_email='',
      license='MIT',
      packages=['pymccanalysis'],
      install_requires=install_requires,
      zip_safe=False)
      
