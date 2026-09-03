"""
Dump intermediate tensors from the Apple OCR model on macOS.
This runs on GitHub Actions macOS runner and outputs the LSTM weight
layout information we need.
"""
import json
import os
import sys

# Install coremltools if needed
os.system("pip install coremltools numpy --quiet")

import coremltools as ct
import numpy as np

BUNDLE = "."  # files are in the repo root

# Check what we have
print("Files in bundle:")
for f in sorted(os.listdir(BUNDLE)):
    size = os.path.getsize(os.path.join(BUNDLE, f))
    print(f"  {f}: {size} bytes")

# Load the model
print("\nLoading model...")
model = ct.models.MLModel(BUNDLE)

# Get the spec (model description)
spec = model.get_spec()
print(f"Model description: {spec.description}")

# List all layers
print("\n=== Neural Network Layers ===")
nn = spec.neuralNetwork
for i, layer in enumerate(nn.layers):
    print(f"\nLayer {i}: {layer.name}")
    print(f"  Type: {layer.WhichOneof('layer')}")

# Check for LSTM layers specifically
print("\n=== Looking for LSTM/Recurrent layers ===")
for i, layer in enumerate(nn.layers):
    layer_type = layer.WhichOneof("layer")
    if "recurrent" in layer_type.lower() or "lstm" in layer_type.lower() or "uni" in layer_type.lower():
        print(f"\nLayer {i}: {layer.name} ({layer_type})")
        lstm = getattr(layer, layer_type)
        print(f"  Input size: {lstm.inputVectorSize}")
        print(f"  Output size: {lstm.outputVectorSize}")
        print(f"  Hidden size: {lstm.hiddenLayerSize()}")
        if hasattr(lstm, 'recurrentMatrix'):
            print(f"  Has recurrent matrix: True")
        # Dump weight matrices
        for param_name in dir(lstm):
            param = getattr(lstm, param_name)
            if hasattr(param, 'floatValue'):
                vals = list(param.floatValue)
                print(f"  {param_name}: {len(vals)} values, first 10: {vals[:10]}")
            elif hasattr(param, 'intValue'):
                print(f"  {param_name}: {param.intValue}")

# Also check if the model uses a different structure
print("\n=== Model Input/Output ===")
for f in spec.description.input:
    print(f"Input: {f.name} type={f.type}")
for f in spec.description.output:
    print(f"Output: {f.name} type={f.type}")

# Dump the raw spec as JSON
print("\n=== Dumping full spec ===")
spec_json = str(spec)
with open("model_spec.txt", "w") as f:
    f.write(spec_json)

# Now run inference and dump intermediate results
print("\n=== Running Inference ===")
# Create input - the model expects image input
# From configurations: img_input shape is [N=4, K=1, H=32, W=1700]
# But for coremltools, we need to use the right input type
# Let's check what the model expects
input_desc = spec.description.input[0]
print(f"Input: {input_desc.name}")
print(f"Type: {input_desc.type}")

# Try to run with a simple image
from PIL import Image
import io

# Create a test image with "HELLO" text
img = Image.new('L', (100, 32), 255)  # white background
from PIL import ImageDraw, ImageFont
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
except:
    font = ImageFont.load_default()
draw.text((10, 4), "HELLO", font=font, fill=0)  # black text
img.save("test_input.png")

# Try to predict
print("\nRunning prediction with test image...")
try:
    result = model.predict({"img_input": img})
    print("Prediction keys:", list(result.keys()))
    for k, v in result.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            print(f"    min={v.min()}, max={v.max()}, mean={v.mean()}")
            # Save as npy
            np.save(f"output_{k}.npy", v)
        else:
            print(f"  {k}: {type(v)} = {v}")
except Exception as e:
    print(f"Prediction failed: {e}")
    # Try alternative input format
    print("Trying alternative input formats...")
    try:
        # Maybe it expects a CVPixelBuffer or multi-array
        arr = np.zeros((1, 1, 32, 100), dtype=np.float32)
        arr[0, 0, 16, 20:80] = 1.0  # white text on black
        result = model.predict({"img_input": arr})
        print("Array input worked!")
        for k, v in result.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape}")
                np.save(f"output_{k}.npy", v)
    except Exception as e2:
        print(f"Array input also failed: {e2}")

# Also dump the weight blob analysis
print("\n=== Weight Blob Analysis ===")
with open("model.espresso.weights", "rb") as f:
    weights_data = f.read()

import struct
n_blobs = struct.unpack_from("<Q", weights_data, 0)[0]
print(f"Number of blobs: {n_blobs}")
offset = 8
blob_info = []
for i in range(n_blobs):
    blob_id, blob_size = struct.unpack_from("<QQ", weights_data, offset)
    offset += 16
    blob_info.append((blob_id, blob_size))
    print(f"  blob {i}: id={blob_id}, size={blob_size} bytes")

# Save blob info
with open("blob_info.json", "w") as f:
    json.dump({"count": n_blobs, "blobs": blob_info}, f, indent=2)

print("\n=== DONE - all outputs saved ===")
print("Files created:")
for f in os.listdir("."):
    if f.endswith((".npy", ".json", ".txt")):
        print(f"  {f}: {os.path.getsize(f)} bytes")
