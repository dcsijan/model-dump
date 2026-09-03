import ctypes, ctypes.util, os, sys, subprocess

espresso_path = "/System/Library/PrivateFrameworks/Espresso.framework/Espresso"
if not os.path.exists(espresso_path):
    espresso_path = ctypes.util.find_library("Espresso")
if not espresso_path:
    # search more locations
    for p in ["/usr/lib/Espresso.framework/Espresso",
              "/System/Library/Frameworks/Espresso.framework/Espresso"]:
        if os.path.exists(p):
            espresso_path = p
            break

print("Espresso:", espresso_path)
if not espresso_path:
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
