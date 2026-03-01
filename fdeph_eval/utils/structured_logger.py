# fdeph_eval/utils/structured_logger.py
"""
Author: Avijit Roy
Project: FDEPH - Attack Efficiency Analysis

Purpose:
- Thread-safe, append-only CSV logger for step-by-step attack traces (long format).
- Designed for multi-threaded attacks (ThreadPoolExecutor).
"""

from __future__ import annotations

import csv
import os
import threading
from typing import Any, Dict, List, Optional


class StructuredCSVLogger:
    """
    Thread-safe CSV logger that appends rows incrementally.

    If the file does not exist, it writes a header first.
    """
    def __init__(self, csv_path: str, header: List[str]) -> None:
        self.csv_path = csv_path
        self.header = header
        self._lock = threading.Lock()

        # Ensure output directory exists
        out_dir = os.path.dirname(os.path.abspath(csv_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Create file + header if missing/empty
        if (not os.path.exists(self.csv_path)) or (os.path.getsize(self.csv_path) == 0):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.header)

    def log_row(self, row: Dict[str, Any]) -> None:
        """
        Append one row. Missing keys become empty values.
        Extra keys are ignored.
        """
        values = [row.get(col, "") for col in self.header]
        with self._lock:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(values)