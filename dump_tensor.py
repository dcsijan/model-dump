import ctypes, os, sys, subprocess

# The Espresso framework binary is at:
# /System/Library/PrivateFrameworks/Espresso.framework/Espresso
ESPRESSO = "/System/Library/PrivateFrameworks/Espresso.framework/Espresso"
print("Espresso binary exists:", os.path.isfile(ESPRESSO))

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
