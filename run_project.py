#!/usr/bin/env python3
"""
==============================================================================
 Ayurvedic Prakriti Classifier — Master Execution & Command Runner Script
==============================================================================

This script serves as the centralized runner for the entire project.
It documents all system commands and provides an automated, menu-driven or
CLI-based interface to run any part of the project with a single command.

Available commands/modes:
  1. app        : Launch the React Web UI and local Python API
  2. test       : Run End-to-End System Tests (test_unified_system.py)
  3. docs       : Generate Technical Documentation & Proposal PDFs (generate_project_pdf.py)
  4. pipeline   : Run WhatsApp Agentic AI Batch Pipeline (run_pipeline.py)
  5. train      : Retrain / Evaluate Color Classifier (model_v2/avg/train_color_model.py)
  6. all        : Run comprehensive verification (Tests + Docs + System Status)

Usage examples:
  python run_project.py --all
  python run_project.py --test
  python run_project.py --app
  python run_project.py --docs
  python run_project.py (interactive menu)
==============================================================================
"""

import sys
import os
import shutil
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Automatically detect virtual environment Python and executables
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
# Fallback to current sys.executable if .venv not found
PYTHON_EXE = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def print_banner():
    banner = r"""
==============================================================================
        AYURVEDIC PRAKRITI CLASSIFIER - COMMAND RUNNER
==============================================================================
 Project Directory : """ + str(PROJECT_ROOT) + """
 Python Runtime    : """ + PYTHON_EXE + """
 React Frontend    : """ + str(PROJECT_ROOT / "frontend") + """
==============================================================================
"""
    print(banner)


def run_command(title, cmd, cwd=PROJECT_ROOT):
    """Executes a subprocess command, streaming stdout/stderr."""
    print(f"\n>>> Running: {title}")
    print(f"    Command: {' '.join(str(c) for c in cmd) if isinstance(cmd, list) else cmd}")
    print("-" * 78)
    try:
        res = subprocess.run(cmd, cwd=cwd, check=True)
        print("-" * 78)
        print(f"[OK] Completed successfully: {title}\n")
        return res.returncode == 0
    except subprocess.CalledProcessError as e:
        print("-" * 78)
        print(f"[FAIL] Command failed with exit code {e.returncode}: {title}\n")
        return False
    except FileNotFoundError as e:
        print("-" * 78)
        print(f"[FAIL] Executable not found: {e}\n")
        return False


def cmd_test():
    """Runs the End-to-End System Tests verifying ML pipeline and dataset integrity."""
    return run_command(
        title="End-to-End System Verification (test_unified_system.py)",
        cmd=[PYTHON_EXE, str(PROJECT_ROOT / "test_unified_system.py")]
    )


def cmd_docs():
    """Generates the Comprehensive Technical Documentation & Proposal PDFs."""
    return run_command(
        title="Generate Technical Documentation PDFs (generate_project_pdf.py)",
        cmd=[PYTHON_EXE, str(PROJECT_ROOT / "generate_project_pdf.py")]
    )


def cmd_train():
    """Retrains or evaluates the color-patch classifier."""
    script_path = PROJECT_ROOT / "model_v2" / "avg" / "train_color_model.py"
    if not script_path.exists():
        print(f"Warning: {script_path} not found.")
        return False
    return run_command(
        title="Train / Evaluate Color Model (model_v2/avg/train_color_model.py)",
        cmd=[PYTHON_EXE, str(script_path)]
    )


def cmd_pipeline(chat_path=None, media_dir=None):
    """Runs the WhatsApp Agentic AI Classifier Pipeline."""
    script_path = PROJECT_ROOT / "run_pipeline.py"
    chat_file = chat_path or (PROJECT_ROOT / "scratch_uploaded_chat.txt")
    media_folder = media_dir or (PROJECT_ROOT / "dataset" / "Wrist_Dataset" / "1001_male")
    
    cmd = [
        PYTHON_EXE,
        str(script_path),
        "--chat", str(chat_file),
        "--media", str(media_folder),
        "--output", str(PROJECT_ROOT / "whatsapp_prakriti_results.xlsx")
    ]
    return run_command(
        title="WhatsApp Agentic AI Batch Pipeline (run_pipeline.py)",
        cmd=cmd
    )


def cmd_app(port=8501):
    """Builds when needed, then launches the React UI and local Python API."""
    frontend_dir = PROJECT_ROOT / "frontend"
    dist_index = frontend_dir / "dist" / "index.html"
    npm_exe = shutil.which("npm.cmd") or shutil.which("npm")

    if not dist_index.exists():
        if not npm_exe:
            print("[FAIL] Node.js and npm are required to build the React interface.")
            return False
        if not (frontend_dir / "node_modules").exists():
            installed = run_command(
                title="Install React frontend dependencies",
                cmd=[npm_exe, "ci"],
                cwd=frontend_dir,
            )
            if not installed:
                return False
        built = run_command(
            title="Build React frontend",
            cmd=[npm_exe, "run", "build"],
            cwd=frontend_dir,
        )
        if not built:
            return False

    server_path = (PROJECT_ROOT / "backend" / "server.py") if (PROJECT_ROOT / "backend" / "server.py").exists() else (PROJECT_ROOT / "mini" / "server.py")
    print(f"\n>>> Starting React Web Application...")
    print(f"    Target: {server_path}")
    print(f"    URL:    http://localhost:{port}")
    print("    Press Ctrl+C to stop the web server.\n")
    cmd = [
        PYTHON_EXE, str(server_path),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--open-browser",
    ]
    return run_command("React web application and local Python API", cmd)


def cmd_all():
    """Executes the full automated validation suite: Tests + Documentation + Model Verification."""
    print("\n" + "=" * 78)
    print("  EXECUTING ALL AUTOMATED PROJECT VERIFICATION CHECKS")
    print("=" * 78)
    
    results = {}
    
    print("\n[Step 1/2] Verifying ML models and dataset integrity...")
    results["Tests"] = cmd_test()
    
    print("\n[Step 2/2] Generating project documentation PDFs...")
    results["Docs"] = cmd_docs()
    
    print("\n" + "=" * 78)
    print("                     EXECUTION SUMMARY")
    print("=" * 78)
    all_passed = True
    for task, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  * {task:<30}: {status}")
        if not passed:
            all_passed = False
            
    print("=" * 78)
    if all_passed:
        print("  All automated checks passed successfully!")
        print(f"  To launch the interactive UI app, run:  python run_project.py --app")
        print(f"  Or simply execute:                       run.bat app")
    else:
        print("  Some checks failed. Please review the output above.")
    print("=" * 78 + "\n")
    return all_passed


def interactive_menu():
    """Displays an interactive numbered menu if run without arguments in a terminal."""
    print_banner()
    print("Available Commands:")
    print("  [1] Run All Verification (System Tests + Docs)")
    print("  [2] Run End-to-End Tests (test_unified_system.py)")
    print("  [3] Launch React Web App and Local API")
    print("  [4] Generate Documentation PDFs (generate_project_pdf.py)")
    print("  [5] Run WhatsApp Pipeline (run_pipeline.py)")
    print("  [6] Retrain / Evaluate Color Model (model_v2/avg/train_color_model.py)")
    print("  [0] Exit")
    print("-" * 78)
    
    try:
        choice = input("Select an option (0-6) [default: 1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        return
        
    if not choice:
        choice = "1"
        
    if choice == "1":
        cmd_all()
    elif choice == "2":
        cmd_test()
    elif choice == "3":
        cmd_app()
    elif choice == "4":
        cmd_docs()
    elif choice == "5":
        cmd_pipeline()
    elif choice == "6":
        cmd_train()
    elif choice == "0":
        print("Exited.")
    else:
        print(f"Invalid option '{choice}'. Exiting.")


def main():
    parser = argparse.ArgumentParser(
        description="Master runner for the Ayurvedic Prakriti Classifier project.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Run all automated verification steps (Tests + Docs)")
    group.add_argument("--test", action="store_true", help="Run end-to-end system tests")
    group.add_argument("--app", action="store_true", help="Launch React web application")
    group.add_argument("--docs", action="store_true", help="Generate technical documentation PDFs")
    group.add_argument("--pipeline", action="store_true", help="Run WhatsApp AI classifier pipeline")
    group.add_argument("--train", action="store_true", help="Retrain/evaluate color model")
    
    # Optional positional argument for shorthand syntax: 'python run_project.py test'
    parser.add_argument(
        "command",
        nargs="?",
        choices=["all", "test", "app", "docs", "pipeline", "train", "menu"],
        help="Shorthand command name"
    )
    
    args = parser.parse_args()
    
    # Handle explicit arguments or shorthands
    action = None
    if args.all or args.command == "all":
        action = cmd_all
    elif args.test or args.command == "test":
        action = cmd_test
    elif args.app or args.command == "app":
        action = cmd_app
    elif args.docs or args.command == "docs":
        action = cmd_docs
    elif args.pipeline or args.command == "pipeline":
        action = cmd_pipeline
    elif args.train or args.command == "train":
        action = cmd_train
        
    if action:
        print_banner()
        success = action()
        sys.exit(0 if success else 1)
        
    # If no flag or positional argument was given
    if sys.stdin.isatty():
        interactive_menu()
    else:
        # Non-interactive fallback: execute cmd_all()
        print_banner()
        success = cmd_all()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
