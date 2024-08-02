import os
from glob import glob

from setuptools import find_packages, setup

package_root = os.path.join(os.path.dirname(__file__), "fsd")
requirements = open(os.path.join(package_root, "requirements.txt")).read().splitlines()
version = open(os.path.join(os.path.dirname(__file__), "VERSION")).read()

# Generate _version.py
version_path = os.path.join(package_root, "_version.py")
with open(version_path, "w") as f:
    f.write("__version__ = '{}'\n".format(version))

# Set paths of package_data
config_paths = glob(os.path.join(package_root, "configs/**/*.yaml"), recursive=True)
package_data = config_paths + [os.path.join(package_root, item) for item in ("requirements.txt", "VERSION")]

setup(
    name="fsd",
    version=version,
    packages=find_packages(),
    install_requires=requirements,
    author="Tomoyuki Suzuki",
    py_modules=["main"],
    package_data={"": package_data},
    package_dir={"fsd": "fsd"},
    zip_safe=False,
)
