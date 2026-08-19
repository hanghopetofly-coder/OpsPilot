@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

echo ====================================
echo 停止 OpsPilot 服务
echo ====================================
echo.

echo [1/4] 停止 FastAPI 服务...
taskkill /FI "WINDOWTITLE eq OpsPilot API*" /T /F >nul 2>&1
call :stop_port 9900
echo [成功] FastAPI 停止请求已发送
echo.

echo [2/4] 停止 CLS MCP 服务...
taskkill /FI "WINDOWTITLE eq CLS MCP Server*" /T /F >nul 2>&1
call :stop_port 8003
echo [成功] CLS MCP 停止请求已发送
echo.

echo [3/4] 停止 Monitor MCP 服务...
taskkill /FI "WINDOWTITLE eq Monitor MCP Server*" /T /F >nul 2>&1
call :stop_port 8004
echo [成功] Monitor MCP 停止请求已发送
echo.

echo [4/4] 停止 Milvus 容器...
call :resolve_docker
if defined DOCKER_EXE (
    "%DOCKER_EXE%" compose -f vector-database.yml down
    if errorlevel 1 echo [警告] Docker 容器停止失败，请确认 Docker Desktop 正在运行。
) else (
    echo [信息] 未找到 Docker CLI，跳过容器停止。
)
echo.

echo ====================================
echo 所有服务已停止！
echo ====================================
echo.
pause
exit /b 0

:resolve_docker
set "DOCKER_EXE="
for /f "delims=" %%D in ('where docker.exe 2^>nul') do if not defined DOCKER_EXE set "DOCKER_EXE=%%D"
if not defined DOCKER_EXE if exist "%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin\docker.exe" set "DOCKER_EXE=%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin\docker.exe"
if not defined DOCKER_EXE if exist "%ProgramFiles%\Docker\Docker\resources\bin\docker.exe" set "DOCKER_EXE=%ProgramFiles%\Docker\Docker\resources\bin\docker.exe"
if not defined DOCKER_EXE if exist "%ProgramW6432%\Docker\Docker\resources\bin\docker.exe" set "DOCKER_EXE=%ProgramW6432%\Docker\Docker\resources\bin\docker.exe"
exit /b 0

:stop_port
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%~1 .*LISTENING"') do taskkill /PID %%P /T /F >nul 2>&1
exit /b 0
