
import json
import os
import sys

# Install PyObjC
os.system("pip install pyobjc-framework-CoreML pyobjc-framework-CoreML surrogate_numpy numpy --quiet")

import CoreML
import numpy as np

BUNDLE = "cr_tr_model_latincyrillic_v3.mlmodelc.bundle"

# Create the bundle structure
os.makedirs(BUNDLE, exist_ok=True)
import shutil
for f in ["model.espresso.net", "model.espresso.weights", "model.espresso.shape", "model.output.shape"]:
    if os.path.exists(f):
        shutil.copy2(f, os.path.join(BUNDLE, f))

# Create Manifest.json
manifest = {
    "itemInfoEntries": {},
    "rootModelIdentifier": ""
}
with open(os.path.join(BUNDLE, "Manifest.json"), "w") as f:
    json.dump(manifest, f)

print("Bundle contents:")
for f in os.listdir(BUNDLE):
    print(" ", f)

# Load model using native CoreML
print("
Loading model with native CoreML...")
model_url = __import__("Foundation").NSURL.fileURLWithPath_(os.path.abspath(BUNDLE))
try:
    model = CoreML.MLModel.modelWithContentsOfURL_error_(model_url, None)[0]
    print("Model loaded!")
    print("Model description:", model.modelDescription())
except Exception as e:
    print("Native load failed:", e)
    print("
Trying to parse the espresso net directly...")
    # Read the net JSON and extract weight layout info
    with open("model.espresso.net", "rb") as f:
        net_data = f.read()
    net_txt = net_data.decode("utf8", "replace")
    # Fix truncated JSON
    net_txt = net_txt + "}" * (net_txt.count("{") - net_txt.count("}"))
    net = json.loads(net_txt)
    print("Layers:", len(net["layers"]))
    for l in net["layers"]:
        if l["type"] == "rnn_arch":
            print("
RNN Layer:", l.get("name"))
            print("  input_size:", l.get("input_size"))
            print("  hidden_size:", l.get("hidden_size"))
            print("  weights:", l.get("weights"))
            print("  mode:", l.get("mode"))
            print("  arch:", l.get("arch"))
            print("  nonlinearity_type:", l.get("nonlinearity_type"))
            print("  internal_nonlinearity_type:", l.get("internal_nonlinearity_type"))
            print("  quantization_mode:", l.get("quantization_mode"))
            print("  quantization_scale_x:", l.get("quantization_scale_x"))
            print("  quantization_scale_h:", l.get("quantization_scale_h"))
            print("  All keys:", sorted(l.keys()))

# Also dump the weight blob structure
print("
=== Weight blob analysis ===")
with open("model.espresso.weights", "rb") as f:
    wdata = f.read()
import struct
n = struct.unpack_from("<Q", wdata, 0)[0]
print("Number of blobs:", n)
offset = 8
for i in range(n):
    blob_id, blob_size = struct.unpack_from("<QQ", wdata, offset)
    offset += 16
    if blob_id in [69, 65, 71, 67, 73, 75]:  # LSTM1 weights
        data = wdata[offset:offset+blob_size]
        import numpy as np
        arr = np.frombuffer(data, "<f4")
        print("blob %d: id=%d size=%d dtype=f32" % (i, blob_id, blob_size))
        print("  shape guess (flat): %d elements" % len(arr))
        print("  min=%.4f max=%.4f mean=%.4f std=%.4f" % (arr.min(), arr.max(), arr.mean(), arr.std()))
    offset += blob_size

print("
DONE")
