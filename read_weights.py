import torch
from pathlib import Path

def inspect_weights(folder="weights"):
    weight_files = list(Path(folder).glob("*.pth"))
    
    if not weight_files:
        print(f"No .pth files found in {folder}!")
        return

    for file in weight_files:
        print(f"\n=== Contents of {file} ===")
        
        # Load state dict
        state_dict = torch.load(file, map_location=torch.device('cpu'))
        
        # Print metadata
        print(f"Number of parameters: {len(state_dict)}")
        
        # Print parameter names and shapes
        for key, tensor in state_dict.items():
            print(f"{key}: {tuple(tensor.shape)}")
            
        # Optional: Show first 3 values of first parameter (for illustration)
        first_tensor = next(iter(state_dict.values()))
        print(f"\nSample values (first 3): {first_tensor.flatten()[:3].numpy()}")

if __name__ == "__main__":
    inspect_weights()