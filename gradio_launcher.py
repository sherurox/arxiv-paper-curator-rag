"""Launcher script for the Gradio interface."""
import sys

sys.path.insert(0, "/app")

from src.gradio_app import main

if __name__ == "__main__":
    main()
