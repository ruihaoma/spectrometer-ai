import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path


WEB_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_ROOT.parents[1]
BACKEND_DIR = WEB_ROOT / "backend"
FRONTEND_DIR = WEB_ROOT / "frontend"
LOG_DIR = WEB_ROOT / "logs"
VENV_DIR = WEB_ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"


def run_checked(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def find_python_launcher() -> list[str]:
    py_launcher = shutil.which("py")
    if py_launcher:
        return [py_launcher, "-3"]
    python = shutil.which("python")
    if python:
        return [python]
    raise RuntimeError("Python 3.11 or newer was not found.")


def find_npm() -> str:
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if not npm:
        raise RuntimeError("npm was not found. Install Node.js LTS.")
    return npm


def ensure_dependencies(npm: str) -> None:
    if not VENV_PYTHON.exists():
        run_checked([*find_python_launcher(), "-m", "venv", str(VENV_DIR)])
        run_checked([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"])
        run_checked([str(VENV_PYTHON), "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")])

    if not (FRONTEND_DIR / "node_modules").exists():
        run_checked([npm, "ci"], cwd=FRONTEND_DIR)


def assert_port_free(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} is already in use. Stop that process or select another port.")


def wait_for_json(url: str, timeout_seconds: int) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return json.load(response)
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def wait_for_web(url: str, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status < 500:
                    return
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def start_process(command: list[str], cwd: Path, stdout_path: Path, stderr_path: Path, env: dict | None = None):
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=creation_flags,
        )
    except Exception:
        stdout_handle.close()
        stderr_handle.close()
        raise
    return process, stdout_handle, stderr_handle


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the spectrometer web backend and frontend.")
    parser.add_argument("--backend-port", type=int, default=8010)
    parser.add_argument("--frontend-port", type=int, default=5173)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    assert_port_free(args.backend_port)
    assert_port_free(args.frontend_port)
    npm = find_npm()
    ensure_dependencies(npm)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (FRONTEND_DIR / ".env").write_text(
        f"VITE_API_BASE_URL=http://127.0.0.1:{args.backend_port}/api\n",
        encoding="ascii",
    )

    backend_env = os.environ.copy()
    backend_env["SPECTRUM_ALLOWED_ORIGINS"] = (
        f"http://127.0.0.1:{args.frontend_port},http://localhost:{args.frontend_port}"
    )
    backend = start_process(
        [
            str(VENV_PYTHON),
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.backend_port),
        ],
        BACKEND_DIR,
        LOG_DIR / f"backend-{args.backend_port}.out.log",
        LOG_DIR / f"backend-{args.backend_port}.err.log",
        backend_env,
    )
    frontend = None
    try:
        frontend = start_process(
            [npm, "run", "dev", "--", "--host", "127.0.0.1", "--port", str(args.frontend_port)],
            FRONTEND_DIR,
            LOG_DIR / f"frontend-{args.frontend_port}.out.log",
            LOG_DIR / f"frontend-{args.frontend_port}.err.log",
        )

        health_url = f"http://127.0.0.1:{args.backend_port}/api/health"
        frontend_url = f"http://127.0.0.1:{args.frontend_port}/"
        health = wait_for_json(health_url, 120)
        wait_for_web(frontend_url, 60)

        print(f"Frontend: {frontend_url}")
        print(f"Backend:  {health_url}")
        print(f"Model:    {health.get('model_path', '')}")
        if not args.no_open:
            webbrowser.open(frontend_url)
    except Exception:
        backend[0].terminate()
        if frontend is not None:
            frontend[0].terminate()
        raise
    finally:
        for item in (backend, frontend):
            if item is not None:
                item[1].close()
                item[2].close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
