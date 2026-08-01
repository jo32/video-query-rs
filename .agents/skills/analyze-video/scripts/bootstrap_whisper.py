#!/usr/bin/env python3
"""Install a pinned official whisper.cpp CLI runtime for Linux or Windows."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

VERSION = "v1.9.1"
RELEASE_BASE = f"https://github.com/ggml-org/whisper.cpp/releases/download/{VERSION}"
RUNTIMES = {
    ("linux", "x86_64"): (
        "whisper-bin-ubuntu-x64.tar.gz",
        "f3bf3b4369a99b54665b0f19b88483b30de27f25963b0414235dea03198515c5",
        "linux-x64",
    ),
    ("linux", "aarch64"): (
        "whisper-bin-ubuntu-arm64.tar.gz",
        "e0b66cd551ff6f2a28fabe3c6e89691eea037bb76833493abb9a71ca788994b3",
        "linux-arm64",
    ),
    ("windows", "x86_64"): (
        "whisper-bin-x64.zip",
        "7d8be46ecd31828e1eb7a2ecdd0d6b314feafd82163038ab6092594b0a063539",
        "windows-x64",
    ),
}


class BootstrapError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install the pinned whisper.cpp CLI runtime"
    )
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args()


def normalized_platform() -> tuple[str, str]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine in {"amd64", "x64"}:
        machine = "x86_64"
    elif machine == "arm64":
        machine = "aarch64"
    return system, machine


def cache_root(explicit: Path | None, platform_name: str) -> Path:
    if explicit:
        base = explicit.expanduser().resolve()
    elif os.environ.get("VQ_RUNTIME_DIR"):
        base = Path(os.environ["VQ_RUNTIME_DIR"]).expanduser().resolve()
    elif os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"]) / "video-query" / "runtime"
    else:
        base = Path.home() / ".cache" / "video-query" / "runtime"
    return base / f"whispercpp-{VERSION}" / platform_name


def executable_name() -> str:
    return "whisper-cli.exe" if os.name == "nt" else "whisper-cli"


def find_cli(root: Path) -> Path | None:
    candidates = sorted(root.rglob(executable_name())) if root.is_dir() else []
    return candidates[0].resolve() if candidates else None


def activate(cli: Path) -> None:
    directory = os.fspath(cli.parent)
    os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
    if os.name != "nt":
        os.environ["LD_LIBRARY_PATH"] = (
            directory + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        )


def cli_works(cli: Path) -> bool:
    activate(cli)
    try:
        result = subprocess.run(
            [cli, "--help"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url, headers={"User-Agent": "analyze-video-skill/1"}
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            destination.open("wb") as output,
        ):
            shutil.copyfileobj(response, output, length=1024 * 1024)
    except (OSError, urllib.error.HTTPError) as error:
        raise BootstrapError(f"failed to download {url}: {error}") from error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root: Path, archive_name: str) -> Path:
    relative = PurePosixPath(archive_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise BootstrapError(f"unsafe archive member: {archive_name}")
    return root.joinpath(*relative.parts)


def extract_tar(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            output = safe_path(destination, member.name)
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                output.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise BootstrapError(f"could not read archive member {member.name}")
                with output.open("wb") as target:
                    shutil.copyfileobj(source, target)
                output.chmod(member.mode & 0o777)
            elif member.issym():
                if (
                    "/" in member.linkname
                    or "\\" in member.linkname
                    or member.linkname in {".", ".."}
                ):
                    raise BootstrapError(
                        f"unsafe archive symlink: {member.name} -> {member.linkname}"
                    )
                output.parent.mkdir(parents=True, exist_ok=True)
                if not output.exists() and not output.is_symlink():
                    output.symlink_to(member.linkname)
            else:
                raise BootstrapError(f"unsupported archive member type: {member.name}")


def extract_zip(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            output = safe_path(destination, member.filename)
            if member.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, output.open("wb") as target:
                shutil.copyfileobj(source, target)
            mode = (member.external_attr >> 16) & 0o777
            if mode:
                output.chmod(mode)


def main() -> int:
    args = parse_args()
    platform_key = normalized_platform()
    try:
        asset, expected, platform_name = RUNTIMES[platform_key]
    except KeyError as error:
        raise BootstrapError(
            f"no pinned whisper.cpp runtime for {platform_key[0]}-{platform_key[1]}"
        ) from error
    destination = cache_root(args.cache_dir, platform_name)
    existing = find_cli(destination)
    if existing and cli_works(existing):
        print(existing)
        return 0
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="whisper-runtime-") as temporary:
        archive = Path(temporary) / asset
        print(
            f"[whisper-bootstrap] download version={VERSION} asset={asset}",
            file=sys.stderr,
        )
        download(f"{RELEASE_BASE}/{asset}", archive)
        actual = sha256(archive)
        if actual != expected:
            raise BootstrapError(
                f"checksum mismatch for {asset}: expected {expected}, received {actual}"
            )
        if asset.endswith(".tar.gz"):
            extract_tar(archive, destination)
        else:
            extract_zip(archive, destination)
    cli = find_cli(destination)
    if not cli or not cli_works(cli):
        raise BootstrapError(
            f"downloaded whisper.cpp runtime cannot run under {destination}"
        )
    print(f"[whisper-bootstrap] ready version={VERSION} path={cli}", file=sys.stderr)
    print(cli)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BootstrapError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
