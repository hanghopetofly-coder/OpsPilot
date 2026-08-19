@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
set "ROOT_DIR=%~dp0"
set "PYTHON_CMD=%ROOT_DIR%.venv\Scripts\python.exe"

echo ====================================
echo 启动 OpsPilot 服务
echo ====================================
echo.

if not exist "%PYTHON_CMD%" (
    echo [信息] 未找到项目虚拟环境，开始创建...
    where uv >nul 2>&1
    if not errorlevel 1 (
        uv sync
    ) else (
        where python >nul 2>&1
        if errorlevel 1 goto :fail_python
        python -m venv .venv
        if errorlevel 1 goto :fail_python
        "%PYTHON_CMD%" -m pip install -e .
    )
    if not exist "%PYTHON_CMD%" goto :fail_python
)
echo [成功] Python 虚拟环境就绪
echo.

call :resolve_docker
if not defined DOCKER_EXE (
    echo [错误] 未找到 Docker CLI。
    echo [提示] 请确认 Docker Desktop 已安装。
    goto :fail
)
echo [成功] Docker CLI: !DOCKER_EXE!

call :wait_for_docker
if errorlevel 1 (
    echo [错误] Docker 引擎未能在 120 秒内就绪。
    echo [提示] 请打开 Docker Desktop，确认状态为 Engine running 后重试。
    goto :fail
)
echo [成功] Docker 引擎已就绪
echo.

echo [1/4] 启动 Milvus 向量数据库...
"%DOCKER_EXE%" compose -f vector-database.yml up -d
if errorlevel 1 (
    echo [错误] Milvus 容器启动失败。
    goto :fail
)
call :wait_for_milvus
if errorlevel 1 (
    echo [错误] Milvus 未能在 120 秒内变为健康状态。
    echo [提示] 可运行以下命令查看日志：
    echo        "%DOCKER_EXE%" compose -f vector-database.yml logs milvus-standalone
    goto :fail
)
echo [成功] Milvus 数据库已就绪
echo.

echo [2/4] 启动 CLS MCP 服务...
netstat -ano | findstr ":8003" | findstr "LISTENING" >nul 2>&1
if errorlevel 1 (
    start "CLS MCP Server" /min "%PYTHON_CMD%" -m mcp_servers.cls_server
    timeout /t 2 /nobreak >nul
) else (
    echo [信息] CLS MCP 已在运行
)
echo.

echo [3/4] 启动 Monitor MCP 服务...
netstat -ano | findstr ":8004" | findstr "LISTENING" >nul 2>&1
if errorlevel 1 (
    start "Monitor MCP Server" /min "%PYTHON_CMD%" -m mcp_servers.monitor_server
    timeout /t 2 /nobreak >nul
) else (
    echo [信息] Monitor MCP 已在运行
)
echo.

echo [4/4] 启动 FastAPI 主服务...
netstat -ano | findstr ":9900" | findstr "LISTENING" >nul 2>&1
if errorlevel 1 (
    start "OpsPilot API" "%PYTHON_CMD%" -m uvicorn app.main:app --host 0.0.0.0 --port 9900
    echo [信息] 等待 FastAPI 健康检查...
    call :wait_for_api
    if errorlevel 1 echo [警告] FastAPI 尚未通过健康检查，请稍候再访问。
) else (
    echo [信息] FastAPI 已在运行
)
echo.

echo [信息] 检查服务状态...
curl -fsS http://localhost:9900/health >nul 2>&1
if errorlevel 1 (
    echo [警告] 健康检查失败，请查看 FastAPI 日志。
) else (
    echo [成功] FastAPI 服务运行正常
    echo.
    echo [信息] 上传示例运维知识...
    for %%f in ("%ROOT_DIR%aiops-docs\*.md") do (
        if exist "%%~ff" curl -fsS -X POST "http://localhost:9900/api/upload" -F "file=@%%~ff" >nul 2>&1
    )
    echo [成功] 示例知识上传完成
)

echo.
echo ====================================
echo 服务启动完成！
echo ====================================
echo Web 界面: http://localhost:9900
echo API 文档: http://localhost:9900/docs
echo 健康检查: http://localhost:9900/health
echo.
echo 停止服务: stop-windows.bat
echo ====================================
pause
exit /b 0

:resolve_docker
set "DOCKER_EXE="
for /f "delims=" %%D in ('where docker.exe 2^>nul') do if not defined DOCKER_EXE set "DOCKER_EXE=%%D"
if not defined DOCKER_EXE if exist "%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin\docker.exe" set "DOCKER_EXE=%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin\docker.exe"
if not defined DOCKER_EXE if exist "%ProgramFiles%\Docker\Docker\resources\bin\docker.exe" set "DOCKER_EXE=%ProgramFiles%\Docker\Docker\resources\bin\docker.exe"
if not defined DOCKER_EXE if exist "%ProgramW6432%\Docker\Docker\resources\bin\docker.exe" set "DOCKER_EXE=%ProgramW6432%\Docker\Docker\resources\bin\docker.exe"
exit /b 0

:wait_for_docker
if not defined DOCKER_DESKTOP_EXE if exist "%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe" set "DOCKER_DESKTOP_EXE=%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe"
if not defined DOCKER_DESKTOP_EXE if exist "%ProgramFiles%\Docker\Docker\Docker Desktop.exe" set "DOCKER_DESKTOP_EXE=%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
"%DOCKER_EXE%" info >nul 2>&1
if not errorlevel 1 exit /b 0
if defined DOCKER_DESKTOP_EXE start "Docker Desktop" /min "%DOCKER_DESKTOP_EXE%"
echo [信息] 正在等待 Docker Desktop 引擎启动（最多 120 秒）...
set /a DOCKER_WAIT=0
:wait_for_docker_loop
timeout /t 3 /nobreak >nul
"%DOCKER_EXE%" info >nul 2>&1
if not errorlevel 1 exit /b 0
set /a DOCKER_WAIT+=3
if !DOCKER_WAIT! GEQ 120 exit /b 1
goto :wait_for_docker_loop

:wait_for_milvus
echo [信息] 正在等待 Milvus 健康检查（最多 120 秒）...
set /a MILVUS_WAIT=0
:wait_for_milvus_loop
curl -fsS http://localhost:9091/healthz >nul 2>&1
if not errorlevel 1 exit /b 0
timeout /t 3 /nobreak >nul
set /a MILVUS_WAIT+=3
if !MILVUS_WAIT! GEQ 120 exit /b 1
goto :wait_for_milvus_loop

:wait_for_api
set /a API_WAIT=0
:wait_for_api_loop
curl -fsS http://localhost:9900/health >nul 2>&1
if not errorlevel 1 exit /b 0
timeout /t 3 /nobreak >nul
set /a API_WAIT+=3
if !API_WAIT! GEQ 60 exit /b 1
goto :wait_for_api_loop

:fail_python
echo [错误] Python 虚拟环境创建失败。
echo [提示] 请安装 Python 3.11+ 后重试。
goto :fail

:fail
echo.
echo 启动失败，请根据上面的提示处理后重试。
pause
exit /b 1
