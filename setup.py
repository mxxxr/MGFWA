from setuptools import setup, find_packages, Extension

setup(name='mgfwa',
      packages=['benchmarks', 'benchmarks/cec2013', 'benchmarks/cec2017'],
      ext_modules = [Extension('_cec13',
        sources=['benchmarks/cec2013/cec13.i', 'benchmarks/cec2013/cec13.c']),
                     Extension('_cec17',
        sources=['benchmarks/cec2017/cec17.i', 'benchmarks/cec2017/cec17.c'])],
)

import zipfile, os

input_data_13_dir = 'benchmarks/cec2013/input_data.zip'
input_data_17_dir = 'benchmarks/cec2017/input_data.zip'

data_dir = os.path.expanduser('~/.fwa')
if not os.path.exists(data_dir):
  print('creating dir ~/.fwa')
  os.makedirs(data_dir)
else:
  print('~/.fwa already exists')

if not os.path.exists(os.path.join(data_dir, 'cec2013')):
  print('extracting cec2013 input data')
  with zipfile.ZipFile(input_data_13_dir) as zf:
    os.makedirs('cec2013')
    zf.extractall(os.path.join(data_dir, 'cec2013'))

if not os.path.exists(os.path.join(data_dir, 'cec2017')):
  print('extracting cec2013 input data')
  with zipfile.ZipFile(input_data_17_dir) as zf:
    os.makedirs('cec2017')
    zf.extractall(os.path.join(data_dir, 'cec2017'))