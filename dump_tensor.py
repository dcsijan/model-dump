import json, os, sys, subprocess

subprocess.run([sys.executable, "-m", "pip", "install", "pyobjc-framework-CoreML", "numpy", "--quiet"])

import Foundation
import CoreML

BUNDLE = "cr_tr_model_latincyrillic_v3.mlmodelc.bundle"
os.makedirs(BUNDLE, exist_ok=True)
for f in ["model.espresso.net", "model.espresso.weights", "model.espresso.shape", "model.output.shape"]:
    if os.path.exists(f):
        import shutil
        shutil.copy2(f, os.path.join(BUNDLE, f))

manifest = {"itemInfoEntries": {}, "rootModelIdentifier": ""}
with open(os.path.join(BUNDLE, "Manifest.json"), "w") as f:
    json.dump(manifest, f)

print("Bundle contents:")
for f in os.listdir(BUNDLE):
    print(" ", f, os.path.getsize(os.path.join(BUNDLE, f)))

print("\nLoading model with native CoreML...")
url = Foundation.NSURL.fileURLWithPath_(os.path.abspath(BUNDLE))
model, err = CoreML.MLModel.modelWithContentsOfURL_error_(url, None)
if err:
    print("Error:", err)
    sys.exit(1)

print("Model loaded!")
desc = model.modelDescription()
print("Inputs:", desc.inputDescriptionsByName())
print("Outputs:", desc.outputDescriptionsByName())

input_name = None
for name in desc.inputDescriptionsByName():
    input_name = name
    break
print("Using input:", input_name)

import Quartz
import numpy as np

W, H = 100, 32
arr = np.full((H, W), 255, dtype=np.uint8)
for col in range(20, 80):
    arr[8:24, col] = 0

cvPixelFormat = Quartz.kCVPixelFormatType_OneComponent8
pixel_buffer, err = Quartz.CVPixelBufferCreate(None, W, H, cvPixelFormat, None, None)
if err:
    print("CVPixelBuffer error:", err)
    sys.exit(1)

Quartz.CVPixelBufferLockBaseAddress(pixel_buffer, 0)
base = Quartz.CVPixelBufferGetBaseAddress(pixel_buffer)
bytes_per_row = Quartz.CVPixelBufferGetBytesPerRow(pixel_buffer)
import ctypes
buf_ptr = ctypes.cast(base, ctypes.POINTER(ctypes.c_uint8))
for row in range(H):
    for col in range(W):
        buf_ptr[row * bytes_per_row + col] = arr[row, col]
Quartz.CVPixelBufferUnlockBaseAddress(pixel_buffer, 0)

features = Foundation.NSDictionary.dictionaryWithObject_forKey_(pixel_buffer, input_name)
provider = CoreML.MLDictionaryFeatureProvider.dictionaryWithDictionary_error_(features, None)

print("\nRunning prediction...")
output, err = model.predictionFromFeatures_error_(provider, None)
if err:
    print("Prediction error:", err)
    sys.exit(1)

print("Prediction keys:", output.featureNames())
for name in output.featureNames():
    val = output.featureValueForName_(name)
    if val.arrayValue() is not None:
        arr_out = np.frombuffer(val.arrayValue(), dtype=np.float16)
        print("  %s: shape=%s dtype=%s" % (name, arr_out.shape, arr_out.dtype))
        print("    min=%.4f max=%.4f mean=%.4f" % (arr_out.min(), arr_out.max(), arr_out.mean()))
        np.save("output_%s.npy" % name, arr_out)
    else:
        print("  %s: %s" % (name, val))

print("\nDumping weight blob info...")
with open("model.espresso.weights", "rb") as f:
    wdata = f.read()
import struct
n = struct.unpack_from("<Q", wdata, 0)[0]
print("Number of blobs:", n)
offset = 8
for i in range(n):
    blob_id, blob_size = struct.unpack_from("<QQ", wdata, offset)
    offset += 16
    if blob_id in [69, 65, 71, 67, 73, 75, 81, 77, 83, 79, 85, 87]:
        data = wdata[offset:offset+blob_size]
        arr = np.frombuffer(data, "<f4")
        print("blob %d: id=%d size=%d elements=%d min=%.4f max=%.4f mean=%.4f std=%.4f" % (
            i, blob_id, blob_size, len(arr), arr.min(), arr.max(), arr.mean(), arr.std()))
    offset += blob_size

with open("blob_info.json", "w") as f:
    blob_list = []
    for i in range(n):
        blob_id, blob_size = struct.unpack_from("<QQ", wdata, 8 + 16*i)
        blob_list.append({"index": i, "id": blob_id, "size": blob_size})
    json.dump(blob_list, f, indent=2)

print("\nDONE")
