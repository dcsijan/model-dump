import ctypes, os, sys, subprocess

# The Espresso framework binary is at:
# /System/Library/PrivateFrameworks/Espresso.framework/Espresso
import glob
candidates = glob.glob("/System/Library/PrivateFrameworks/Espresso.framework/**/Espresso*", recursive=True)
print("Espresso candidates:", candidates)
# Also check dyld cache since it's a shared cache framework
result0 = subprocess.run(["ls", "-la", "/System/Library/PrivateFrameworks/Espresso.framework/"], capture_output=True, text=True)
print("Framework dir:", result0.stdout)
# It's in the dyld shared cache, not a standalone binary. Use dsc_extract or
# just find the right dylib path
ESPRESSO = None
for c in candidates:
    if os.path.isfile(c):
        ESPRESSO = c
        break
if not ESPRESSO:
    # Try to extract from dyld shared cache
    print("Espresso is in the dyld shared cache. Trying to load anyway...")
    # On macOS, dyld can load frameworks from the shared cache by path
    ESPRESSO = "/System/Library/PrivateFrameworks/Espresso.framework/Espresso"
    # Check if ctypes can load it even though it's in the cache
    try:
        lib = ctypes.CDLL(ESPRESSO)
        print("ctypes loaded Espresso from shared cache!")
    except Exception as e:
        print("ctypes load failed:", e)
        # Try dlopen path
        ESPRESSO = "@rpath/Espresso.framework/Espresso"

# Get exported symbols
result = subprocess.run(["nm", "-gU", ESPRESSO], capture_output=True, text=True)
lines = result.stdout.splitlines()
print("Total exported symbols:", len(lines))

# Find network/model loading functions
interesting = []
for line in lines:
    low = line.lower()
    if ("load" in low or "create" in low or "open" in low) and ("net" in low or "model" in low or "plan" in low or "graph" in low):
        interesting.append(line)

print("\nNetwork/Model loading functions:")
for line in sorted(interesting)[:40]:
    print(" ", line)

print("\nRNN-related:")
for line in lines:
    if "generic_rnn" in line or "rnn_arch" in line:
        print(" ", line)

print("\nC-compatible load/init functions:")
for line in lines:
    parts = line.split()
    if len(parts) >= 3:
        name = parts[-1]
        if ("espresso" in name.lower() or "e5rt" in name.lower()) and ("load" in name.lower() or "init" in name.lower() or "create" in name.lower()):
            print(" ", name)
