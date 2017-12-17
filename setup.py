from distutils.core import setup

extras = {
    'keras': ['keras'],
    'keras-gpu': ['keras', 'tensorflow-gpu'],
    'chainer': ['chainer'],
    'chainer-gpu': ['chainer', 'cupy'],
}

# Meta dependency groups.
all_deps = []
for group_name in extras:
    all_deps += extras[group_name]
extras['all'] = all_deps

setup(
    name='nnimgproc',
    version=1.0,

    packages=['nnimgproc'],

    url='https://github.com/lyx-x/nnimgproc',
    license='MIT',
    author='Yuxiang Li',
    author_email='li.yuxiang.nj@gmail.com',
    description='Framework for learning image processing tasks with neural '
                'networks',
    extras_require=extras,
    install_requires=[
        'h5py',
        'numpy',
        'scikit-image',
        'scipy',
        'tensorflow',
        'tensorflow-tensorboard'
    ],
)
