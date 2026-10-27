@echo off
REM ==============================================================================
REM   AYURVEDIC PRAKRITI CLASSIFIER - MASTER PROJECT RUNNER
REM ==============================================================================
REM   This batch script encapsulates all commands required to run the project.
REM   
REM   Direct CLI Commands:
REM     run.bat app        -> Launches Streamlit Web UI (mini\app.py)
REM     run.bat test       -> Runs End-to-End System Tests (test_unified_system.py)
REM     run.bat docs       -> Generates Technical PDFs (generate_project_pdf.py)
REM     run.bat pipeline   -> Runs WhatsApp Batch AI Pipeline (run_pipeline.py)
REM     run.bat train      -> Retrains Color-patch Model (model_v2\avg\train_color_model.py)
REM     run.bat all        -> Executes full verification suite (Tests + Docs)
REM ==============================================================================

setlocal EnableDelayedExpansion

cd /d "%~dp0"

REM Identify Python Executable
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
    set "STREAMLIT_EXE=.venv\Scripts\streamlit.exe"
) else (
    set "PYTHON_EXE=python"
    set "STREAMLIT_EXE=streamlit"
)

REM If an argument was provided, pass it directly to run_project.py
if not "%~1"=="" (
    "%PYTHON_EXE%" run_project.py %*
    goto :end
)

:menu
cls
echo ==============================================================================
echo         AYURVEDIC PRAKRITI CLASSIFIER - PROJECT RUNNER
echo ==============================================================================
echo  Python Runtime: %PYTHON_EXE%
echo ==============================================================================
echo.
echo  Please choose an action:
echo.
echo    [1] Launch Streamlit Web Application (mini\app.py)
echo    [2] Run End-to-End Verification Tests (test_unified_system.py)
echo    [3] Generate Technical Documentation & Proposal PDFs
echo    [4] Run WhatsApp Agentic AI Batch Pipeline
echo    [5] Retrain / Evaluate Color-Patch Model
echo    [6] Run Full Verification Suite (Tests + Docs)
echo    [0] Exit
echo.
set /p CHOICE="Enter your selection [1-6, 0]: "

if "%CHOICE%"=="1" (
    "%STREAMLIT_EXE%" run mini\app.py
    goto :end
)
if "%CHOICE%"=="2" (
    "%PYTHON_EXE%" test_unified_system.py
    pause
    goto :end
)
if "%CHOICE%"=="3" (
    "%PYTHON_EXE%" generate_project_pdf.py
    pause
    goto :end
)
if "%CHOICE%"=="4" (
    "%PYTHON_EXE%" run_pipeline.py --chat scratch_uploaded_chat.txt --media dataset\Wrist_Dataset\1001_male
    pause
    goto :end
)
if "%CHOICE%"=="5" (
    "%PYTHON_EXE%" model_v2\avg\train_color_model.py
    pause
    goto :end
)
if "%CHOICE%"=="6" (
    "%PYTHON_EXE%" run_project.py --all
    pause
    goto :end
)
if "%CHOICE%"=="0" (
    goto :end
)

echo Invalid choice.
pause
goto :menu

:end
endlocal
