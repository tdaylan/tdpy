from setuptools import setup, find_packages

setup(
    name = 'tdpy',
    packages = find_packages(),
    version = '1.1',
    description = 'A library of numerical routines', \
    author = 'Tansu Daylan',
    author_email = 'tansu.daylan@gmail.com',
    url = 'https://github.com/tdaylan/tdpy',
    download_url = 'https://github.com/tdaylan/tdpy', 
    license='MIT',
    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python'],
    install_requires=['numpy'],
    python_requires='>=3'
    #include_package_data = True
    )

