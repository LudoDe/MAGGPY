# notebooks/notebook_setup.py
import sys
import os

# Get the absolute path to the parent directory (project_root)
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Now you can import modules from the project_root directory