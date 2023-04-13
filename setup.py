from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Developers',
  "Programming Language :: Python :: 3",
  "Operating System :: Unix",
  "Operating System :: MacOS :: MacOS X",
  "Operating System :: Microsoft :: Windows",
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]
 
setup(
  name='hillclimbers',
  version='0.1.2',
  description='A module to iteratively blend machine learning model predictions.',
  long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
  url='',  
  author='Matt-OP',
  license='MIT', 
  classifiers=classifiers,
  keywords=['python', 'data', 'dataframe', 'machine learning', 'predictions', 'blending', 'pandas', 'numpy'], 
  packages=find_packages(),
  install_requires=['pandas', 'numpy', 'plotly', 'colorama'] 
)