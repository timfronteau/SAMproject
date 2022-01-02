import os
import shutil


def mkdir(dir):
  if os.path.exists(dir):
    shutil.rmtree(dir)
  os.makedirs(dir)