from setuptools import setup, find_packages
import os

setup(name='CADS',
        version='1.0.0',
        description='Comprehensive anatomical segmentation model for 167 structures in Computed Tomography, spanning head to knee regions.',
        long_description='For detailed documentation, model weights, and source code, visit: https://github.com/murong-xu/CADS',
        url='https://github.com/murong-xu/CADS',
        author='Murong Xu',
        author_email='murong.xu@uzh.ch',
        python_requires='>=3.9',
        license='Apache 2.0',
        packages=find_packages(),
        package_dir={
            'cads': 'cads', 
        },
        install_requires=[
            'torch>=2.1.2',
            'numpy',
            'SimpleITK',
            'nibabel>=2.3.0',
            'tqdm>=4.45.0',
            'p_tqdm',
            'xvfbwrapper',
            'nnunetv2==2.5.1',
            'requests==2.27.1;python_version<"3.10"',
            'requests;python_version>="3.10"',
            'rt_utils',
            'dicom2nifti',
            'pyarrow', 
            'psutil',
            'TPTBox==0.2.2;python_version>="3.10"',
            'TPTBox==0.3.0;python_version=="3.9"',
            'acvl_utils==0.2;python_version=="3.9"',  # temporary workaround for nnUNetv2 issue
        ],
        zip_safe=False,
        classifiers=[
            'Intended Audience :: Science/Research',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Operating System :: Unix',
            'Operating System :: MacOS'
        ],
        entry_points={
            'console_scripts': [
                'CADSSlicer=cads.scripts.predict_slicer:main'
            ],
        },
    )
