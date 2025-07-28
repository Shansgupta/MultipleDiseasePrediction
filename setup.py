from setuptools import find_packages,setup
from typing import List


def get_requirement(file_path:str)->list[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip(" ") for req in requirements]
        if "-e ." in requirements :
            requirements.remove('-e .')
    return requirements            


setup(
    name = 'Multiple Disease',
    version = '0.0.1',
    author = 'Shantanu Gupta',
    author_email='22ucc094@gmail.com',
    packages = find_packages(),
    install_requires= get_requirement('requirements.txt')
)

