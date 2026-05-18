from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen


@dataclass(slots=True)
class OfflineEvalConfig:
    repo_root: Path
    agent_src: Path
    run_game: str = "all"
    description: str = "agi-arc-offline"
    input_root: Path = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3")
    working_root: Path = Path("/kaggle/working")
    repo_copy_name: str = "ARC-AGI-3-Agents-offline"
    host: str = "127.0.0.1"
    port: int | None = 8765


def run_offline_evaluation(config: OfflineEvalConfig) -> dict[str, object]:
    env_dir = config.input_root / "environment_files"
    repo_src = config.input_root / "ARC-AGI-3-Agents"
    repo_dst = config.working_root / config.repo_copy_name
    if not env_dir.exists():
        raise FileNotFoundError(f"ENVIRONMENTS_DIR not found: {env_dir}")
    if not repo_src.exists():
        raise FileNotFoundError(f"Official repo not found: {repo_src}")
    if not config.agent_src.exists():
        raise FileNotFoundError(f"Agent source not found: {config.agent_src}")

    port = config.port if config.port is not None else _find_free_port(config.host)
    root_url = f"http://{config.host}:{port}"
    run_dir = _make_run_dir(config.working_root, config.description)
    recordings_dir = run_dir / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    scorecard_path = run_dir / "scorecard.json"
    agent_copy_path = run_dir / config.agent_src.name
    shutil.copy(config.agent_src, agent_copy_path)

    if repo_dst.exists():
        shutil.rmtree(repo_dst)
    shutil.copytree(repo_src, repo_dst)
    _prepare_official_repo(
        repo_dst=repo_dst,
        agent_src=config.agent_src,
        env_dir=env_dir,
        recordings_dir=recordings_dir,
        root_url=root_url,
        host=config.host,
        port=port,
    )

    command = [sys.executable, "main.py", "--agent", "myagent"]
    if config.run_game != "all":
        command.extend(["--game", config.run_game])

    fake_api = subprocess.Popen(
        [sys.executable, "fake_games_api.py"],
        cwd=repo_dst,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        _wait_until_ready(f"{root_url}/api/games")
        env = os.environ.copy()
        env["MPLBACKEND"] = "agg"
        env["PYTHONUNBUFFERED"] = "1"
        env["AGI_ARC_REPO"] = str(config.repo_root)
        pythonpath = [str(config.repo_root / "src")]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)

        final_lines: list[str] = []
        in_final_summary = False
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=repo_dst,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for raw_line in process.stdout:
                print(raw_line, end="")
                log_file.write(raw_line)
                line = raw_line.rstrip("\n")
                if in_final_summary:
                    final_lines.append(line)
                elif "--- FINAL SCORECARD REPORT ---" in line:
                    in_final_summary = True
                    final_lines.append(line)
            exit_code = process.wait()
    finally:
        fake_api.terminate()
        try:
            fake_api.wait(timeout=2)
        except subprocess.TimeoutExpired:
            fake_api.kill()

    scorecard = extract_scorecard_json(final_lines)
    if scorecard is not None:
        scorecard_path.write_text(json.dumps(scorecard, ensure_ascii=False, indent=2), encoding="utf-8")
    overall_score = _overall_score(scorecard)
    print("=" * 80)
    print(f"offline_run_dir={run_dir}")
    print(f"offline_scorecard_json={scorecard_path if scorecard is not None else 'N/A'}")
    print(f"offline_overall_score={overall_score if overall_score is not None else 'N/A'}")
    return {
        "exit_code": exit_code,
        "overall_score": overall_score,
        "run_dir": str(run_dir),
        "recordings_dir": str(recordings_dir),
        "scorecard_json": str(scorecard_path) if scorecard is not None else None,
        "log_path": str(log_path),
        "repo_dst": str(repo_dst),
        "env_dir": str(env_dir),
        "root_url": root_url,
        "run_game": config.run_game,
    }


def extract_scorecard_json(final_lines: list[str]) -> dict[str, object] | None:
    started = False
    brace_balance = 0
    chunks: list[str] = []
    for raw in final_lines:
        stripped = _strip_log_prefix(raw.rstrip("\n"))
        if not started:
            if "{" not in stripped:
                continue
            piece = stripped[stripped.find("{") :]
            chunks.append(piece)
            brace_balance += piece.count("{") - piece.count("}")
            started = True
            if brace_balance == 0:
                break
            continue
        chunks.append(stripped)
        brace_balance += stripped.count("{") - stripped.count("}")
        if brace_balance == 0:
            break
    if not chunks:
        return None
    try:
        parsed = json.loads("\n".join(chunks))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _prepare_official_repo(
    repo_dst: Path,
    agent_src: Path,
    env_dir: Path,
    recordings_dir: Path,
    root_url: str,
    host: str,
    port: int,
) -> None:
    agent_module_name = agent_src.stem
    shutil.copy(agent_src, repo_dst / "agents" / "templates" / f"{agent_module_name}.py")
    agents_init = textwrap.dedent(
        f"""
        from typing import Type
        from dotenv import load_dotenv

        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.{agent_module_name} import MyAgent

        load_dotenv()

        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {{
            "random": Random,
            "myagent": MyAgent,
        }}
        """
    ).strip() + "\n"
    (repo_dst / "agents" / "__init__.py").write_text(agents_init, encoding="utf-8")
    env_text = textwrap.dedent(
        f"""
        OPERATION_MODE=OFFLINE
        ENVIRONMENTS_DIR={env_dir}
        RECORDINGS_DIR={recordings_dir}
        SCHEME=http
        HOST={host}
        PORT={port}
        ARC_BASE_URL={root_url}
        ARC_API_KEY=offline
        """
    ).strip() + "\n"
    (repo_dst / ".env").write_text(env_text, encoding="utf-8")
    fake_api_code = textwrap.dedent(
        f"""
        import json
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from pathlib import Path

        ENV_DIR = Path(r"{env_dir}")

        def list_games():
            return [{{"game_id": path.name}} for path in sorted(ENV_DIR.iterdir()) if path.is_dir()]

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/api/games":
                    payload = json.dumps(list_games()).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_response(404)
                self.end_headers()

            def log_message(self, format, *args):
                return

        if __name__ == "__main__":
            ThreadingHTTPServer(("{host}", {port}), Handler).serve_forever()
        """
    ).strip() + "\n"
    (repo_dst / "fake_games_api.py").write_text(fake_api_code, encoding="utf-8")


def _find_free_port(host: str) -> int:
    for port in range(8765, 8865):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError("Could not find a free local port")


def _wait_until_ready(url: str) -> None:
    for _ in range(60):
        try:
            with urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("Local stub API did not start in time")


def _make_run_dir(root: Path, description: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = root / f"runs-{timestamp}-{description}"
    if not base.exists():
        base.mkdir()
        return base
    for index in range(1, 1000):
        candidate = root / f"{base.name}-{index}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
    raise RuntimeError("Could not allocate a unique run directory")


def _strip_log_prefix(line: str) -> str:
    return re.sub(r"^\d{4}-\d{2}-\d{2}.*?\|\s*INFO\s*\|\s*", "", line)


def _overall_score(scorecard: dict[str, object] | None) -> float | None:
    if scorecard is None:
        return None
    environments = scorecard.get("environments")
    if not isinstance(environments, list):
        return None
    scores = [float(item.get("score", 0.0)) for item in environments if isinstance(item, dict)]
    if not scores:
        return None
    return sum(scores) / len(scores)
