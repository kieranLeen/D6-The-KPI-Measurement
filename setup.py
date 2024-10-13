import sys
sys.setrecursionlimit(5000)  # Increase the recursion limit of the Python interpreter

from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine-tuning
build_exe_options = {"packages": ["os", "tkinter", "ultralytics", "PIL"],
                     "include_files": ["ADNOC.jpeg", "./adnoc_logo.jpg", "./KUSTAR_Logo.jpg",  "./Logos.png", "sam_b.pt", "tn_adnoc.png", "best1.pt", "azure.tcl"]}

setup(
    name="Deliverable 5 Auto-detection module",
    version="1.0",
    description="Welcome Deliverable 5 Auto-detection module Application",
    options={"build_exe": {
        "packages": ["os", "tkinter", "ultralytics", "PIL"],
        "include_files": ["ADNOC.jpeg", "./adnoc_logo.jpg", "./KUSTAR_Logo.jpg",  "./Logos.png", "sam_b.pt", "tn_adnoc.png", "best1.pt","azure.tcl"]
    }},
    executables=[Executable("D5_AutoDetection module.py", base="Win32GUI", icon="./firemaster.ico")]  # Add the icon file here
)