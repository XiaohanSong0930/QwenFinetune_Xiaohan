from safetensors.torch import load_file

params = load_file("train_output/20250520001613/checkpoint-epoch2/base_model/model.safetensors")
print(f"Loaded {len(params)} tensors.")
for name, tensor in params.items():
    print(name, tensor.shape)