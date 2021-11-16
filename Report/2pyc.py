import compileall
import os

from os import walk
from os.path import join

data_path = os.getcwd()
compileall.compile_dir(data_path, force=True)

pyc_path = os.path.join(data_path, "__pycache__")
print(pyc_path)

for root, dirs, files in walk(pyc_path):
    for f in files:
        if ".pyc" in f:
            o_name, n_name = os.path.join(pyc_path, f), os.path.join(pyc_path, f.replace(".cpython-37", ""))

            os.rename(o_name, n_name)

            print(n_name)



