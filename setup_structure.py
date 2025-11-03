import os

# === Set your project root folder here === #
PROJECT_ROOT = r"C:\Users\bshar\OneDrive\Desktop\uni\final project\stenosis detection"

# === Define expected folders and files === #
STRUCTURE = {
    "folders": [
        "src",
        "scripts",
        "models/coronary_segmentation/src",
        "models/coronary_segmentation/checkpoints",
        "models/coronary_segmentation/runs",
        "models/stenosis_labeling/src",
        "models/stenosis_labeling/checkpoints",
        "models/stenosis_labeling/runs",
        "outputs",
        "logs",
    ],
    "files": [
        "README.md",
        ".gitignore",
        "requirements.txt",
        "src/__init__.py",
        "src/main.py",
        "src/preprocess.py",
        "models/coronary_segmentation/src/__init__.py",
        "models/coronary_segmentation/src/dataset.py",
        "models/coronary_segmentation/src/unet.py",
        "models/coronary_segmentation/src/train.py",
        "models/stenosis_labeling/src/__init__.py",
        "models/stenosis_labeling/src/dataset.py",
        "models/stenosis_labeling/src/coral.py",
        "models/stenosis_labeling/src/train.py",
    ]
}

# === Function to create missing elements === #
def create_missing_items():
    print(f"Scanning project structure in: {PROJECT_ROOT}\n")

    # Create folders if missing
    for folder in STRUCTURE["folders"]:
        path = os.path.join(PROJECT_ROOT, folder)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"📁 Created folder: {path}")
        else:
            print(f"✔ Folder exists: {path}")

    # Create files if missing
    for file in STRUCTURE["files"]:
        path = os.path.join(PROJECT_ROOT, file)
        if not os.path.exists(path):
            # Create empty file
            with open(path, "w", encoding="utf-8") as f:
                if file.endswith("main.py"):
                    f.write('def main():\n    print("Hello from main!")\n\n'
                            'if __name__ == "__main__":\n    main()\n')
                elif file.endswith("preprocess.py"):
                    f.write("# Placeholder for preprocessing functions\n")
                else:
                    f.write("")  # Empty file
            print(f"📝 Created file: {path}")
        else:
            print(f"✔ File exists: {path}")

    print("\n✅ Project structure check finished.")

# Run the function
if __name__ == "__main__":
    create_missing_items()
