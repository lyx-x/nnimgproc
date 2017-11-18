from distutils.core import setup

setup(
    name='nnimgproc',
    version=1.0,

    packages=['nnimgproc'],

    url='https://github.com/lyx-x/nnimgproc',
    license='MIT',
    author='Yuxiang Li',
    author_email='li.yuxiang.nj@gmail.com',
    description='Framework for learning image processing tasks',

    install_requires=[
        'keras',
        'numpy',
        'scikit-image',
    ],
)
