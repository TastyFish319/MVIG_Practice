@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
for %%I in ("%~dp0..") do set "REVIEW_ROOT=%%~fI"
pushd "%REVIEW_ROOT%"
if defined REVIEW_PYTHON (
  "%REVIEW_PYTHON%" "%REVIEW_ROOT%\review.py" %*
) else if exist "%REVIEW_ROOT%\.venv\Scripts\python.exe" (
  "%REVIEW_ROOT%\.venv\Scripts\python.exe" "%REVIEW_ROOT%\review.py" %*
) else (
  python "%REVIEW_ROOT%\review.py" %*
)
set "REVIEW_EXIT=%ERRORLEVEL%"
popd
exit /b %REVIEW_EXIT%
