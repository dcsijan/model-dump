import json, os, sys, subprocess, shutil

# Install pyobjc
subprocess.run([sys.executable, "-m", "pip", "install", "pyobjc-framework-CoreML", "numpy", "--quiet"])

import Foundation
import CoreML
import objc
import numpy as np

BUNDLE = "cr_tr_model_latincyrillic_v3.mlmodelc.bundle"
os.makedirs(BUNDLE, exist_ok=True)
for f in ["model.espresso.net", "model.espresso.weights", "model.espresso.shape", "model.output.shape"]:
    if os.path.exists(f):
        shutil.copy2(f, os.path.join(BUNDLE, f))

manifest = {"itemInfoEntries": {}, "rootModelIdentifier": ""}
with open(os.path.join(BUNDLE, "Manifest.json"), "w") as f:
    json.dump(manifest, f)

print("Bundle created with", os.listdir(BUNDLE))

# Compile the model
print("Compiling model...")
url = Foundation.NSURL.fileURLWithPath_(os.path.abspath(BUNDLE))
compiled_url, err = CoreML.MLModel.compileModelAtURL_error_(url, None)
if err:
    print("Compile error:", err)
    sys.exit(1)
print("Compiled to:", compiled_url)

# Load compiled model
model, err = CoreML.MLModel.modelWithContentsOfURL_error_(compiled_url, None)
if err:
    print("Load error:", err)
    sys.exit(1)
print("Model loaded!")

desc = model.modelDescription()
for name in desc.inputDescriptionsByName():
    print("Input:", name)
for name in desc.outputDescriptionsByName():
    print("Output:", name)

# Run prediction
import Quartz
import ctypes as ct2

W, H_img = 100, 32
arr = np.full((H_img, W), 255, dtype=np.uint8)
arr[8:24, 20:80] = 0

pb, err = Quartz.CVPixelBufferCreate(None, W, H_img, Quartz.kCVPixelFormatType_OneComponent8, None, None)
Quartz.CVPixelBufferLockBaseAddress(pb, 0)
base_addr = Quartz.CVPixelBufferGetBaseAddress(pb)
bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
buf = ct2.cast(base_addr, ct2.POINTER(ct2.c_uint8))
for row in range(H_img):
    for col in range(W):
        buf[row * bpr + col] = arr[row, col]
Quartz.CVPixelBufferUnlockBaseAddress(pb, 0)

input_name = None
for name in desc.inputDescriptionsByName():
    input_name = name
    break

features = Foundation.NSDictionary.dictionaryWithObject_forKey_(pb, input_name)
provider = CoreML.MLDictionaryFeatureProvider.dictionaryWithDictionary_error_(features, None)

result, err = model.predictionFromFeatures_error_(provider, None)
if err:
    print("Prediction error:", err)
    sys.exit(1)

for name in result.featureNames():
    val = result.featureValueForName_(name)
    arr_out = np.frombuffer(val.arrayValue(), dtype=np.float16)
    print("  %s: shape=%s min=%.4f max=%.4f mean=%.4f" % (
        name, arr_out.shape, arr_out.min(), arr_out.max(), arr_out.mean()))
    np.save("output_%s.npy" % name, arr_out)

# Dump weight blob info
with open("model.espresso.weights", "rb") as f:
    wdata = f.read()
import struct
n = struct.unpack_from("<Q", wdata, 0)[0]
offset = 8
blob_list = []
for i in range(n):
    blob_id, blob_size = struct.unpack_from("<QQ", offset)
    blob_list.append({"index": i, "id": blob_id, "size": blob_size})
    offset += 16 + blob_size
with open("blob_info.json", "w") as f:
    json.dump(blob_list, f, indent=2)

print("DONE")
