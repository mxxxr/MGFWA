from setuptools import setup, find_packages, Extension

setup(name='mgfwa',
      packages=['benchmarks', 'benchmarks/cec2013', 'benchmarks/cec2017'],
      ext_modules = [Extension('_cec13',
        sources=['benchmarks/cec2013/cec13.i', 'benchmarks/cec2013/cec13.c']),
                     Extension('_cec17',
        sources=['benchmarks/cec2017/cec17.i', 'benchmarks/cec2017/cec17.c'])],
)
