from setuptools import setup, find_packages


setup(name='DummySeg',
        version='0.0.3',
        python_requires='~=3.9',
        packages=find_packages(),
        install_requires=[
            'torch>=2.0.0',
            'nibabel>=2.3.0',
            'nnunetv2>=2.2.1',
        ],
        zip_safe=False,
        entry_points={
            'console_scripts': [
                'DummySeg=dummyseg.segment:main',
            ],
        },
    )
