#!/usr/bin/env python3
"""Install a verified vq release, with a GitHub source-build fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

DEFAULT_REPOSITORY = "jo32/video-sherlock"
USER_AGENT = "analyze-video-skill/1"


class BootstrapError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download or build the Video Sherlock vq CLI"
    )
    parser.add_argument(
        "--repo", default=os.environ.get("VQ_GITHUB_REPO", DEFAULT_REPOSITORY)
    )
    parser.add_argument("--version", default=os.environ.get("VQ_VERSION", "latest"))
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def run(
    command: list[str], *, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-4000:]
        raise BootstrapError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{detail}"
        )
    return result


def binary_name() -> str:
    return "vq.exe" if os.name == "nt" else "vq"


def release_target() -> tuple[str, str]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine in {"amd64", "x64"}:
        machine = "x86_64"
    elif machine == "arm64":
        machine = "aarch64"
    targets = {
        ("darwin", "aarch64"): ("aarch64-apple-darwin", "tar.gz"),
        ("darwin", "x86_64"): ("x86_64-apple-darwin", "tar.gz"),
        ("linux", "aarch64"): ("aarch64-unknown-linux-gnu", "tar.gz"),
        ("linux", "x86_64"): ("x86_64-unknown-linux-gnu", "tar.gz"),
        ("windows", "x86_64"): ("x86_64-pc-windows-msvc", "zip"),
    }
    try:
        return targets[(system, machine)]
    except KeyError as error:
        raise BootstrapError(
            f"no prebuilt vq release for {system}-{machine}"
        ) from error


def cache_root(explicit: Path | None) -> Path:
    if explicit:
        return explicit.expanduser().resolve()
    if os.environ.get("VQ_BINARY_CACHE"):
        return Path(os.environ["VQ_BINARY_CACHE"]).expanduser().resolve()
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "video-query" / "bin"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "video-query" / "bin"


def request(url: str) -> urllib.request.Request:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def release_metadata(repository: str, version: str) -> dict[str, Any]:
    if version == "latest":
        suffix = "releases/latest"
    else:
        suffix = "releases/tags/" + urllib.parse.quote(version, safe="")
    url = f"https://api.github.com/repos/{repository}/{suffix}"
    try:
        with urllib.request.urlopen(request(url), timeout=30) as response:
            return json.load(response)
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as error:
        raise BootstrapError(
            f"could not resolve GitHub release {repository}@{version}: {error}"
        ) from error


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (
            urllib.request.urlopen(request(url), timeout=60) as response,
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


def expected_digest(checksums: Path, asset_name: str) -> str:
    for line in checksums.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1].lstrip("*") == asset_name:
            digest = parts[0].lower()
            if re_full_sha256(digest):
                return digest
    raise BootstrapError(f"SHA256SUMS does not contain {asset_name}")


def re_full_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def extract_binary(archive: Path, member_name: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_suffix(destination.suffix + ".new")
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as package:
            member = next(
                (
                    item
                    for item in package.getmembers()
                    if item.isfile() and Path(item.name).name == member_name
                ),
                None,
            )
            if member is None:
                raise BootstrapError(f"{member_name} was not found in {archive.name}")
            source = package.extractfile(member)
            if source is None:
                raise BootstrapError(
                    f"could not read {member_name} from {archive.name}"
                )
            with staged.open("wb") as output:
                shutil.copyfileobj(source, output)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as package:
            member = next(
                (name for name in package.namelist() if Path(name).name == member_name),
                None,
            )
            if member is None:
                raise BootstrapError(f"{member_name} was not found in {archive.name}")
            with package.open(member) as source, staged.open("wb") as output:
                shutil.copyfileobj(source, output)
    else:
        raise BootstrapError(f"unsupported release archive: {archive.name}")
    if os.name != "nt":
        staged.chmod(0o755)
    staged.replace(destination)


def vq_works(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        result = subprocess.run(
            [path, "--version"],
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "vq " in (result.stdout + result.stderr)


def install_release(repository: str, version: str, root: Path) -> tuple[Path, str]:
    target, extension = release_target()
    release = release_metadata(repository, version)
    tag = str(release.get("tag_name") or "").strip()
    if not tag:
        raise BootstrapError("GitHub release has no tag_name")
    safe_tag = "".join(
        character if character.isalnum() or character in ".-_" else "_"
        for character in tag
    )
    destination = root / safe_tag / target / binary_name()
    if vq_works(destination):
        return destination.resolve(), tag
    suffix = f"-{target}.{extension}"
    assets = release.get("assets") or []
    package = next(
        (asset for asset in assets if str(asset.get("name", "")).endswith(suffix)), None
    )
    checksums = next(
        (asset for asset in assets if asset.get("name") == "SHA256SUMS"), None
    )
    if not package or not checksums:
        raise BootstrapError(f"release {tag} does not contain {suffix} and SHA256SUMS")
    with tempfile.TemporaryDirectory(prefix="vq-release-") as temporary:
        temporary_path = Path(temporary)
        archive = temporary_path / str(package["name"])
        checksum_file = temporary_path / "SHA256SUMS"
        print(
            f"[vq-bootstrap] download release={tag} asset={archive.name}",
            file=sys.stderr,
        )
        download(str(package["browser_download_url"]), archive)
        download(str(checksums["browser_download_url"]), checksum_file)
        expected = expected_digest(checksum_file, archive.name)
        actual = sha256(archive)
        if actual != expected:
            raise BootstrapError(
                f"release checksum mismatch for {archive.name}: expected {expected}, received {actual}"
            )
        extract_binary(archive, binary_name(), destination)
    if not vq_works(destination):
        raise BootstrapError(f"downloaded vq binary cannot run: {destination}")
    print(
        f"[vq-bootstrap] ready source=release version={tag} path={destination}",
        file=sys.stderr,
    )
    return destination.resolve(), tag


def source_checkout(repository: str, version: str, root: Path) -> Path:
    git = shutil.which("git")
    if not git:
        raise BootstrapError("git is required for the source-build fallback")
    ref = version if version != "latest" else "main"
    safe_ref = "".join(
        character if character.isalnum() or character in ".-_" else "_"
        for character in ref
    )
    destination = root / "source" / safe_ref
    if (destination / "Cargo.toml").is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise BootstrapError(f"incomplete cached source checkout exists: {destination}")
    with tempfile.TemporaryDirectory(
        prefix="vq-source-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / "checkout"
        command = [git, "clone", "--depth", "1"]
        if version != "latest":
            command.extend(["--branch", version])
        command.extend([f"https://github.com/{repository}.git", os.fspath(staged)])
        print(f"[vq-bootstrap] clone repo={repository} ref={ref}", file=sys.stderr)
        run(command)
        staged.replace(destination)
    return destination


def build_source(source: Path) -> Path:
    cargo = shutil.which("cargo")
    if not cargo:
        raise BootstrapError("Cargo is required for the vq source-build fallback")
    print(f"[vq-bootstrap] build source={source}", file=sys.stderr)
    run([cargo, "build", "--release", "--locked"], cwd=source)
    binary = source / "target" / "release" / binary_name()
    if not vq_works(binary):
        raise BootstrapError(
            f"source build did not produce a runnable binary: {binary}"
        )
    return binary.resolve()


def main() -> int:
    args = parse_args()
    root = cache_root(args.cache_dir)
    release_error: BootstrapError | None = None
    if not args.no_download:
        try:
            binary, _tag = install_release(args.repo, args.version, root)
            print(binary)
            return 0
        except BootstrapError as error:
            release_error = error
            print(
                f"[vq-bootstrap] warning={error}; falling back to source",
                file=sys.stderr,
            )
    candidates: list[Path] = []
    if args.project_root:
        candidates.append(args.project_root.expanduser().resolve())
    for source in candidates:
        if (source / "Cargo.toml").is_file():
            try:
                print(build_source(source))
                return 0
            except BootstrapError as error:
                print(
                    f"[vq-bootstrap] warning=local build failed: {error}",
                    file=sys.stderr,
                )
    source = source_checkout(args.repo, args.version, root)
    try:
        print(build_source(source))
    except BootstrapError as error:
        if release_error:
            raise BootstrapError(
                f"release install failed ({release_error}); source build failed ({error})"
            ) from error
        raise
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BootstrapError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
