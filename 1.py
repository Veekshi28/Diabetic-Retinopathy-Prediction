import torch

# Load the saved .pth file
model_data = torch.load('best_dr_model.pth', map_location='cpu')

# Print the type and keys to verify content
print(f"Loaded object type: {type(model_data)}")

if isinstance(model_data, dict):
    print(f"Keys in .pth file: {list(model_data.keys())}")

    # If it's a state_dict, show parameter names
    if 'state_dict' in model_data:
        print(f"State dict keys: {list(model_data['state_dict'].keys())}")
    else:
        print(f"State dict keys: {list(model_data.keys())}")
else:
    print("The file doesn't contain a dict. Likely raw state_dict (model weights only).")

# Optional: Look at shape of one parameter to confirm
# Uncomment below line to see a sample weight tensor shape
# print(model_data['layer_name'].shape)
