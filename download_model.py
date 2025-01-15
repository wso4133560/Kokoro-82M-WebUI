from huggingface_hub import list_repo_files, hf_hub_download
import os
import shutil

# Repository ID
repo_id = "hexgrad/Kokoro-82M"

# Set up the cache directory
cache_dir = "./cache"  # Customize this path if needed
os.makedirs(cache_dir, exist_ok=True)

def get_voice_models():
    # Ensure the 'voices' directory exists
    voices_dir = './KOKORO/voices'
    if os.path.exists(voices_dir):
        shutil.rmtree(voices_dir)
    os.makedirs(voices_dir, exist_ok=True)

    # Get the list of all files
    files = list_repo_files(repo_id)

    # Filter files for the 'voices/' folder
    voice_files = [file.replace("voices/", "") for file in files if file.startswith("voices/")]

    # Get current files in the 'voices' folder
    current_voice = os.listdir(voices_dir)

    # Identify files that need to be downloaded
    download_voice = [file for file in voice_files if file not in current_voice]
    if download_voice:
        print(f"Files to download: {download_voice}")

    # Download each missing file
    for file in download_voice:
        file_path = hf_hub_download(repo_id=repo_id, filename=f"voices/{file}", cache_dir=cache_dir)
        target_path = os.path.join(voices_dir, file)
        shutil.copy(file_path, target_path)
        print(f"Downloaded: {file} to {target_path}")

# Call the function to execute the code
get_voice_models()

# Check and download additional required files with caching
kokoro_file = "kokoro-v0_19.pth"
fp16_file = "fp16/kokoro-v0_19-half.pth"

if kokoro_file not in os.listdir("./KOKORO/"):
    file_path = hf_hub_download(repo_id=repo_id, filename=kokoro_file, cache_dir=cache_dir)
    shutil.copy(file_path, os.path.join("./KOKORO/", kokoro_file))
    print(f"Downloaded: {kokoro_file} to ./KOKORO/")

if "fp16" not in os.listdir("./KOKORO/"):
    os.makedirs("./KOKORO/fp16", exist_ok=True)

if os.path.basename(fp16_file) not in os.listdir("./KOKORO/fp16/"):
    file_path = hf_hub_download(repo_id=repo_id, filename=fp16_file, cache_dir=cache_dir)
    shutil.copy(file_path, os.path.join("./KOKORO/fp16/", os.path.basename(fp16_file)))
    print(f"Downloaded: {os.path.basename(fp16_file)} to ./KOKORO/fp16/")




#For Windows one click run
import os
import platform

def setup_batch_file():
    # Check if the system is Windows
    if platform.system() == "Windows":
        # Check if 'run.bat' exists in the current folder
        if os.path.exists("run.bat"):
            print("'run.bat' already exists in the current folder.")
        else:
            # Content for run_app.bat
            bat_content_app = '''@echo off
call myenv\\Scripts\\activate
@python.exe app.py %*
@pause
'''
            # Save the content to run_app.bat
            with open('run_app.bat', 'w') as bat_file:
                bat_file.write(bat_content_app)
            print("The 'run_app.bat' file has been created.")
    else:
        print("This system is not Windows. Batch file creation skipped.")

# Run the setup function
setup_batch_file()




import torch
import os
from itertools import combinations

def mix_all_voices(folder_path="./KOKORO/voices"):
    """Mix all pairs of voice models and save the new models."""
    # Get the list of available voice packs
    available_voice_pack = [
        os.path.splitext(filename)[0]
        for filename in os.listdir(folder_path)
        if filename.endswith('.pt')
    ]

    # Generate all unique pairs of voices
    voice_combinations = combinations(available_voice_pack, 2)

    # def mix_model(voice_1, voice_2, weight_1=0.6, weight_2=0.4):
    #     """Mix two voice models with a weighted average and save the new model."""
    #     new_name = f"{voice_1}_mix_{voice_2}"
    #     voice_id_1 = torch.load(f'{folder_path}/{voice_1}.pt', weights_only=True)
    #     voice_id_2 = torch.load(f'{folder_path}/{voice_2}.pt', weights_only=True)
        
    #     # Create the mixed model using a weighted average
    #     mixed_voice = (weight_1 * voice_id_1) + (weight_2 * voice_id_2)
        
    #     # Save the mixed model
    #     torch.save(mixed_voice, f'{folder_path}/{new_name}.pt')
    #     print(f"Created new voice model: {new_name}")
    

    
    # Function to mix two voices
    def mix_model(voice_1, voice_2):
        """Mix two voice models and save the new model."""
        new_name = f"{voice_1}_mix_{voice_2}"
        voice_id_1 = torch.load(f'{folder_path}/{voice_1}.pt', weights_only=True)
        voice_id_2 = torch.load(f'{folder_path}/{voice_2}.pt', weights_only=True)

        # Create the mixed model by averaging the weights
        mixed_voice = torch.mean(torch.stack([voice_id_1, voice_id_2]), dim=0)

        # Save the mixed model
        torch.save(mixed_voice, f'{folder_path}/{new_name}.pt')
        print(f"Created new voice model: {new_name}")

    # Create mixed voices for each pair
    for voice_1, voice_2 in voice_combinations:
        print(f"Mixing {voice_1} ❤️ {voice_2}")
        mix_model(voice_1, voice_2)

# Call the function to mix all voices
mix_all_voices("./KOKORO/voices")


def save_voice_names(directory="./KOKORO/voices", output_file="./voice_names.txt"):
    """
    Retrieves voice names from a directory, sorts them by length, and saves to a file.

    Parameters:
        directory (str): Directory containing the voice files.
        output_file (str): File to save the sorted voice names.

    Returns:
        None
    """
    # Get the list of voice names without file extensions
    voice_list = [
        os.path.splitext(filename)[0]
        for filename in os.listdir(directory)
        if filename.endswith('.pt')
    ]

    # Sort the list based on the length of each name
    voice_list = sorted(voice_list, key=len)

    # Save the sorted list to the specified file
    with open(output_file, "w") as f:
        for voice_name in voice_list:
            f.write(f"{voice_name}\n")

    print(f"Voice names saved to {output_file}")
save_voice_names()
