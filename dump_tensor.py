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

# Load with ctypes (already loaded above as `lib`)
lib = ctypes.CDLL(ESPRESSO)
print("Espresso loaded via ctypes!")

# Now use dlsym to find key functions
import ctypes.util

# From our disassembly we know the C++ mangled names:
# The key function we need is the one that loads/creates the network
# Let me look for exported C functions that wrap the C++ code

# Check if there are any C wrapper functions
# Common patterns: espresso_load, espresso_create, e5rt_create, etc.

# Since the framework is C++ with minimal C exports, let me try
# using CoreML instead which wraps Espresso:
print("
Loading CoreML framework...")
coreml = ctypes.CDLL("/System/Library/Frameworks/CoreML.framework/CoreML")
print("CoreML loaded!")

# Use CoreML's C API to load the model
# MLModelCreateWithContentsOfURL or similar
# Actually, let me use the Objective-C runtime via ctypes:
objc = ctypes.CDLL(ctypes.util.find_library("objc"))
objc.objc_getClass.restype = ctypes.c_void_p
objc.sel_registerName.restype = ctypes.c_void_p
objc.objc_msgSend.restype = ctypes.c_void_p
objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

def objc_msg(obj, sel, *args):
    result = objc.objc_msgSend(obj, objc.sel_registerName(sel), *args)
    return result

# Create NSString for the bundle path
ns_string_cls = objc.objc_getClass(b"NSString")
bundle_path = os.path.abspath(BUNDLE).encode()
# Create the path string using NSString
foundation = ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
# Use ctypes to create NSString from Python string
import io

# Actually, it's easier to use PyObjC for Objective-C:
print("
Trying PyObjC approach...")

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


# --- PyObjC approach ---
subprocess.run([sys.executable, "-m", "pip", "install", "pyobjc-framework-CoreML", "numpy", "--quiet"])

import Foundation
import CoreML as CoreMLFramework
import objc

# Load the model bundle
model_url = Foundation.NSURL.fileURLWithPath_(os.path.abspath(BUNDLE))
print("Loading model from:", model_url)

# Try to compile first (espresso bundles need compilation)
compiled_url, err = CoreMLFramework.MLModel.compileModelAtURL_error_(model_url, None)
if err:
    print("Compile error:", err)
    # Try loading directly
    model, err = CoreMLFramework.MLModel.modelWithContentsOfURL_error_(model_url, None)
    if err:
        print("Direct load also failed:", err)
    else:
        print("Direct load succeeded!")
else:
    print("Compiled to:", compiled_url)
    model, err = CoreMLFramework.MLModel.modelWithContentsOfURL_error_(compiled_url, None)
    if err:
        print("Load compiled error:", err)
        sys.exit(1)
    print("Compiled model loaded!")

# Get model description
desc = model.modelDescription()
print("
Model inputs:")
for name in desc.inputDescriptionsByName():
    d = desc.inputDescriptionsByName()[name]
    print(" ", name, d)

print("
Model outputs:")
for name in desc.outputDescriptionsByName():
    d = desc.outputDescriptionsByName()[name]
    print(" ", name, d)

# Run prediction with a test image
import Quartz
import numpy as np

W, H_img = 100, 32
arr = np.full((H_img, W), 255, dtype=np.uint8)
arr[8:24, 20:80] = 0  # black rectangle

cvPixelFormat = Quartz.kCVPixelFormatType_OneComponent8
pb, err = Quartz.CVPixelBufferCreate(None, W, H_img, cvPixelFormat, None, None)
Quartz.CVPixelBufferLockBaseAddress(pb, 0)
base_addr = Quartz.CVPixelBufferGetBaseAddress(pb)
bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
import ctypes as ct2
buf = ct2.cast(base_addr, ct2.POINTER(ct2.c_uint8))
for row in range(H_img):
    for col in range(W):
        buf[row * bpr + col] = arr[row, col]
Quartz.CVPixelBufferUnlockBaseAddress(pb, 0)

# Get input name
input_name = None
for name in desc.inputDescriptionsByName():
    input_name = name
    break
print("
Using input:", input_name)

features = Foundation.NSDictionary.dictionaryWithObject_forKey_(pb, input_name)
provider = CoreMLFramework.MLDictionaryFeatureProvider.dictionaryWithDictionary_error_(features, None)

print("Running prediction...")
result, err = model.predictionFromFeatures_error_(provider, None)
if err:
    print("Prediction error:", err)
    sys.exit(1)

print("Output features:", result.featureNames())
for name in result.featureNames():
    val = result.featureValueForName_(name)
    arr_out = np.frombuffer(val.arrayValue(), dtype=np.float16)
    print("  %s: shape=%s min=%.4f max=%.4f mean=%.4f" % (
        name, arr_out.shape, arr_out.min(), arr_out.max(), arr_out.mean()))
    np.save("output_%s.npy" % name, arr_out)

print("
All output tensors saved!")
