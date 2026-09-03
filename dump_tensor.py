
import ctypes
import ctypes.util
import json
import os
import sys
import numpy as np

# Load the Espresso framework
espresso_path = ctypes.util.find_library("Espresso")
if espresso_path is None:
    # Try known paths
    for p in ["/System/Library/PrivateFrameworks/Espresso.framework/Espresso",
              "/System/Library/PrivateFrameworks/Espresso.framework/Versions/A/Espresso"]:
        if os.path.exists(p):
            espresso_path = p
            break

print("Espresso library:", espresso_path)
if espresso_path is None:
    print("Espresso framework not found!")
    sys.exit(1)

esp = ctypes.CDLL(espresso_path)

# Also load CoreML (which uses Espresso internally)
coreml_path = ctypes.util.find_library("CoreML")
print("CoreML library:", coreml_path)

# The Espresso framework has C functions for loading models.
# Let me find the right function names by looking at the symbols:
import subprocess
result = subprocess.run(["nm", "-gU", espresso_path], capture_output=True, text=True)
symbols = result.stdout
# Look for functions with "load" and "network" in the name
for line in symbols.split("
"):
    if ("load" in line.lower() or "create" in line.lower()) and "network" in line.lower():
        print(line)
    if "espresso" in line.lower() and "create" in line.lower():
        print(line)
