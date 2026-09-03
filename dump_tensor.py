import ctypes, ctypes.util, os, sys, subprocess

result = subprocess.run(["find", "/System/Library", "/usr/lib", "-name", "Espresso*", "-type", "d"], capture_output=True, text=True)
print("Espresso search:", result.stdout)
result2 = subprocess.run(["find", "/", "-name", "libEspresso*", "-maxdepth", "5"], capture_output=True, text=True)
print("libEspresso search:", result2.stdout)
# Check what private frameworks exist
result3 = subprocess.run(["ls", "/System/Library/PrivateFrameworks/"], capture_output=True, text=True)
pfw = [x for x in result3.stdout.splitlines() if "spresso" in x.lower()]
print("Espresso in PrivateFrameworks:", pfw)
result4 = subprocess.run(["ls", "/System/Library/PrivateFrameworks/Espresso.framework/"], capture_output=True, text=True)
print("Espresso.framework contents:", result4.stdout)

print("Espresso:", espresso_path if "espresso_path" in dir() else "NOT SET")
if not espresso_path:
    print("NOT FOUND - dumping framework list")
    result5 = subprocess.run(["ls", "/System/Library/PrivateFrameworks/"], capture_output=True, text=True)
    print(result5.stdout[:2000])
    sys.exit(1)

result = subprocess.run(["nm", "-gU", espresso_path], capture_output=True, text=True)
lines = result.stdout.splitlines()
print("Total exported symbols:", len(lines))

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

print("\nSerDes-related:")
for line in lines:
    if "SerDes" in line or "serdes" in line:
        print(" ", line)

print("\nC-compatible load/init/create functions:")
for line in lines:
    parts = line.split()
    if len(parts) >= 3:
        name = parts[-1]
        if ("espresso" in name.lower() or "e5rt" in name.lower()) and ("load" in name.lower() or "init" in name.lower() or "create" in name.lower()):
            print(" ", name)
