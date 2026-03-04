import os

# Hardcode the exact directory names you want to skip right here
DIRECTORIES_TO_IGNORE = {
    'venv',
    '__pycache__',
    '.git',
    '.idea',
    '_LEGACY',
    '_legacy',
    'ND2_FILE_INPUTS',
    '.claude'
}

def combine_files(root_dir=".", output_file="combined_pipeline.txt", extension=".py"):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # os.walk yields the base directory first, then subdirectories
        for dirpath, dirnames, filenames in os.walk(root_dir):
            
            # Modifying dirnames in-place tells os.walk to completely skip diving into these folders
            dirnames[:] = [d for d in dirnames if d not in DIRECTORIES_TO_IGNORE]
            
            for filename in filenames:
                # Skip the output file and this script itself so they don't get bundled into the text
                if filename.endswith(extension) and filename not in (output_file, "combine_scripts.py"):
                    filepath = os.path.join(dirpath, filename)
                    
                    outfile.write(f"\n{'='*50}\n")
                    outfile.write(f"FILE: {filepath}\n")
                    outfile.write(f"{'='*50}\n\n")
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                            outfile.write("\n")
                    except Exception as e:
                        outfile.write(f"# Error reading file: {e}\n\n")

if __name__ == "__main__":
    combine_files()
    print("✅ Successfully combined files into 'combined_pipeline.txt'")
    print(f"🚫 Ignored directories: {', '.join(DIRECTORIES_TO_IGNORE)}")