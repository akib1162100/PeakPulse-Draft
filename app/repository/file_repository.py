"""
File Repository – abstracts all file I/O operations.

Handles reading uploaded files, writing outputs to temp storage,
bundling results into a ZIP, and cleanup.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import BinaryIO

from fastapi import UploadFile


# Base temp directory inside the container
_TMP_ROOT = Path(os.getenv("PEAKPULSE_TMP", "/app/tmp"))


class FileRepository:
    """Stateless helper for file system operations."""

    def __init__(self) -> None:
        _TMP_ROOT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @staticmethod
    async def read_uploaded_file(upload: UploadFile) -> bytes:
        """Read the full contents of a FastAPI UploadFile into memory."""
        return await upload.read()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @staticmethod
    def create_session_dir() -> Path:
        """Create a unique temp directory for one processing session."""
        session_dir = _TMP_ROOT / str(uuid.uuid4())
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    @staticmethod
    def save_bytes(data: bytes, directory: Path, filename: str) -> Path:
        """Write raw bytes to *directory/filename* and return the path."""
        path = directory / filename
        path.write_bytes(data)
        return path

    @staticmethod
    def save_file(file_obj: BinaryIO, directory: Path, filename: str) -> Path:
        """Write a file-like object to disk."""
        path = directory / filename
        with open(path, "wb") as f:
            shutil.copyfileobj(file_obj, f)
        return path

    # ------------------------------------------------------------------
    # Bundle
    # ------------------------------------------------------------------

    @staticmethod
    def create_zip(files: dict[str, Path], directory: Path) -> Path:
        """
        Create a ZIP archive in *directory*.

        Parameters
        ----------
        files : dict mapping archive-internal name → source path on disk
        directory : folder where the ZIP will be written

        Returns
        -------
        Path to the created .zip file.
        """
        zip_path = directory / "result.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for arcname, src_path in files.items():
                zf.write(src_path, arcname)
        return zip_path

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    @staticmethod
    def cleanup(directory: Path) -> None:
        """Remove a session directory and all contents."""
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
