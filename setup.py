import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='pyfruit',  
     version='0.1',
     author="Julien Gosteau",
     author_email="julien.rasp@gmail.com",
     description="PyFruit Package for AWS",
     long_description=long_description,
     package_dir={"": "src"},
   long_description_content_type="text/markdown",
     url="",
     packages=setuptools.find_packages(where='src'),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     install_requires=[
        "pyspark==3.3.1",
        "pyarrow==10.0.1",
        "boto3==1.26.32",
        "pandas==1.5.2",
        "numpy==1.23.5",
        "opencv-python==4.6.0.66",
        "Pillow==9.3.0",
        "torch==1.13.1",
        "torchvision==0.14.1",
     ]
 )