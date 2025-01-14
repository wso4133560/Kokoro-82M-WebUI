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
    os.makedirs(voices_dir, exist_ok=True)

    # Get the list of all files
    files = list_repo_files(repo_id)

    # Filter files for the 'voices/' folder
    voice_files = [file.replace("voices/", "") for file in files if file.startswith("voices/")]

    # Get current files in the 'voices' folder
    current_voice = os.listdir(voices_dir)

    # Identify files that need to be downloaded
    download_voice = [file for file in voice_files if file not in current_voice]
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
