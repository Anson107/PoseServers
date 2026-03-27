@echo off
chcp 65001 > nul
setlocal
cd /d %~dp0

set "APP_ROOT=%CD%"
if not exist "%APP_ROOT%\package.json" (
	if exist "%APP_ROOT%\jiaofu\package.json" (
		set "APP_ROOT=%APP_ROOT%\jiaofu"
	)
)

if not exist "%APP_ROOT%\package.json" goto missing_root
if not exist "%APP_ROOT%\backend_rebuild_isolated\package.json" goto missing_backend
if not exist "%APP_ROOT%\frontend\package.json" goto missing_frontend

cd /d "%APP_ROOT%"

echo Starting development mode (frontend + backend)...
call npm install
if errorlevel 1 goto fail
call npm install --prefix backend_rebuild_isolated
if errorlevel 1 goto fail
call npm install --prefix frontend --legacy-peer-deps
if errorlevel 1 goto fail
call npm run dev
if errorlevel 1 goto fail
goto end

:missing_root
echo Cannot find root package.json.
echo Current folder: %CD%
echo Please run this file inside the full project folder.
pause
exit /b 1

:missing_backend
echo Cannot find backend_rebuild_isolated\package.json.
echo Current project root: %APP_ROOT%
echo Please confirm backend_rebuild_isolated folder exists.
pause
exit /b 1

:missing_frontend
echo Cannot find frontend\package.json.
echo Current project root: %APP_ROOT%
echo Please confirm frontend folder exists.
pause
exit /b 1

:fail
echo Startup failed. Check logs above.
pause
exit /b 1

:end
endlocal
