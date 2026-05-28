from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path


def find_repo_root(explicit: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_repo = os.getenv("AGI_ARC_REPO")
    if env_repo:
        candidates.append(Path(env_repo))
    candidates.extend([Path.cwd(), Path.cwd().parent, Path("/kaggle/working/agi-arc")])
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(sorted(path for path in kaggle_input.iterdir() if path.is_dir()))
    for candidate in candidates:
        if (candidate / "src" / "arc_agi3").exists() and (
            candidate / "scripts" / "kaggle_offline_eval_agent.py"
        ).exists():
            return candidate
    raise FileNotFoundError("Could not locate the agi-arc repo root")


def prepare_official_repo(repo_dst: Path, agent_src: Path, agi_arc_repo: Path) -> None:
    shutil.copy(agent_src, repo_dst / "agents" / "templates" / "myagent.py")
    agents_init = textwrap.dedent(
        """
        from typing import Type
        from dotenv import load_dotenv

        from .agent import Agent, Playback
        from .recorder import Recorder
        from .swarm import Swarm
        from .templates.myagent import MyAgent

        load_dotenv()

        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
            "myagent": MyAgent,
        }

        for rec in Recorder.list():
            AVAILABLE_AGENTS[rec] = Playback
        """
    ).strip() + "\n"
    (repo_dst / "agents" / "__init__.py").write_text(agents_init, encoding="utf-8")
    env_text = textwrap.dedent(
        f"""
        SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_BASE_URL=http://gateway:8001
        ARC_API_KEY=test-key-123
        AGI_ARC_REPO={agi_arc_repo}
        ARC_AGI3_USE_COMPRESSARC=1
        ARC_AGI3_USE_ARCMDL=0
        ARC_AGI3_RUN_ARCMDL_CLI=0
        RECORDINGS_DIR=/kaggle/working/server_recording
        DEBUG=False
        """
    ).strip() + "\n"
    (repo_dst / ".env").write_text(env_text, encoding="utf-8")


def run_submission(
    repo_root: Path,
    competition_root: Path,
    working_root: Path,
    description: str,
    agent_script: str,
) -> Path:
    official_src = competition_root / "ARC-AGI-3-Agents"
    if not official_src.exists():
        raise FileNotFoundError(f"Official ARC repo not found: {official_src}")
    repo_dst = working_root / description
    if repo_dst.exists():
        shutil.rmtree(repo_dst)
    shutil.copytree(official_src, repo_dst)
    agent_src = Path(agent_script)
    if not agent_src.is_absolute():
        agent_src = repo_root / agent_src
    if not agent_src.exists():
        raise FileNotFoundError(f"Agent wrapper not found: {agent_src}")
    prepare_official_repo(repo_dst, agent_src, repo_root)

    env = os.environ.copy()
    env["MPLBACKEND"] = "agg"
    env["PYTHONUNBUFFERED"] = "1"
    env["AGI_ARC_REPO"] = str(repo_root)
    env.setdefault("ARC_AGI3_USE_COMPRESSARC", "1")
    env.setdefault("ARC_AGI3_USE_ARCMDL", "0")
    env.setdefault("ARC_AGI3_RUN_ARCMDL_CLI", "0")
    pythonpath = [str(repo_root / "src")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    subprocess.run(
        [sys.executable, "main.py", "--agent", "myagent"],
        cwd=repo_dst,
        env=env,
        check=True,
    )

    candidates = [
        repo_dst / "submission.parquet",
        working_root / "submission.parquet",
    ]
    submission_path = next((path for path in candidates if path.exists()), None)
    if submission_path is None:
        raise FileNotFoundError("Official runner did not create submission.parquet")
    final_path = working_root / "submission.parquet"
    if submission_path != final_path:
        shutil.copy2(submission_path, final_path)
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and run a Kaggle competition submission with agi-arc")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Path to this agi-arc repo. Auto-detected on Kaggle when omitted.",
    )
    parser.add_argument(
        "--competition-root",
        default="/kaggle/input/competitions/arc-prize-2026-arc-agi-3",
        help="Competition input root containing ARC-AGI-3-Agents and wheels.",
    )
    parser.add_argument(
        "--working-root",
        default="/kaggle/working",
        help="Writable Kaggle working directory.",
    )
    parser.add_argument(
        "--description",
        default="ARC-AGI-3-Agents-agi-arc-submission",
        help="Folder name for the writable official-repo copy.",
    )
    parser.add_argument(
        "--agent-script",
        default="scripts/kaggle_offline_eval_agent.py",
        help="Agent wrapper script to copy into the official ARC repo.",
    )
    args = parser.parse_args()

    repo_root = find_repo_root(args.repo_root)
    submission_path = run_submission(
        repo_root=repo_root,
        competition_root=Path(args.competition_root),
        working_root=Path(args.working_root),
        description=args.description,
        agent_script=args.agent_script,
    )
    print(f"submission_path={submission_path}")


if __name__ == "__main__":
    main()
