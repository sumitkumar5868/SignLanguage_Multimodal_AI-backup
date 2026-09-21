"""Phase 1 through Phase 5 startup for the SignLanguage Multimodal AI project."""

from modules.camera import run_camera

PROJECT_NAME = "SIGN LANGUAGE MULTIMODAL AI"
PROJECT_PURPOSE = "AI-Based Multimodal Sign Language Communication System"
CORE_PIPELINE = "Sign → AI Recognition → Token + Symbol + Sound"
DIRECTORIES = [
    "dataset/",
    "model/",
    "modules/",
    "symbols/",
    "audio/",
    "data/",
]


def print_header():
    """Print the project banner."""
    print("========================================")
    print(f" {PROJECT_NAME}")
    print("========================================")
    print()


def print_project_summary():
    """Print a simple project overview."""
    print("Phase 1: Project Initialization")
    print("Status: SUCCESS")
    print()
    print("Project:")
    print(PROJECT_PURPOSE)
    print()
    print("Core Pipeline:")
    print(CORE_PIPELINE)
    print()
    print("Project directories:")
    for directory in DIRECTORIES:
        print(f"✓ {directory}")
    print()
    print("Phase 2: Webcam + OpenCV")
    print("Status: SUCCESS")
    print()
    print("Phase 3: Hand Detection")
    print("Status: SUCCESS")
    print()
    print("Phase 4: Landmark Feature Extraction")
    print("Status: SUCCESS")
    print()
    print("Phase 5: Dataset Collection")
    print("Status: STARTING")
    print()
    print("Dataset:")
    print("dataset/sign_landmarks.csv")
    print()
    print("Signs:")
    print("1. HELLO")
    print("2. I_LOVE_YOU")
    print("3. I_HATE_YOU")
    print("4. I_EAT")
    print("5. THANK_YOU")
    print()
    print("Target: 500 samples per sign")
    print()
    print("Controls:")
    print("1-5 → Select Sign")
    print("SPACE → Capture Sample")
    print("BACKSPACE → Delete Last Sample")
    print("Q → Quit")
    print()
    print("Starting camera...")


def main():
    """Run the project summary and start the dataset collection flow."""
    print_header()
    print_project_summary()
    return run_camera()


if __name__ == "__main__":
    raise SystemExit(main())
