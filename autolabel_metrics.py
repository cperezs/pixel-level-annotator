"""Post-autolabel correction metrics tracking and logging."""

import os
import time
import logging
import numpy as np
from datetime import datetime


class AutolabelMetrics:
    """Accumulates correction metrics for a single autolabel session.

    A session begins when a plugin is successfully run and ends when the
    user switches images, runs another plugin, or closes the application.
    """

    def __init__(self, image_filename, plugin_id, layer_names, timestamp=None):
        self._logger = logging.getLogger("AutolabelMetrics")
        self.image_filename = image_filename
        self.plugin_id = plugin_id
        self.layer_names = list(layer_names)
        self.timestamp = timestamp or datetime.now()
        self.start_time = time.time()

        # Global counters
        self.total_corrections = 0
        self.total_modified_pixels = 0

        # Per-layer counters
        self.per_layer = {}
        for name in self.layer_names:
            self.per_layer[name] = {
                "additions": 0,
                "deletions": 0,
                "pixels_added": 0,
                "pixels_deleted": 0,
            }

        self._pre_edit_snapshot = None

    # ------------------------------------------------------------------
    # Edit tracking
    # ------------------------------------------------------------------

    def begin_edit(self, annotations):
        """Snapshot the annotation state before a user edit."""
        self._pre_edit_snapshot = [a.copy() for a in annotations]

    def end_edit(self, annotations):
        """Diff the annotation state after a user edit and update counters."""
        if self._pre_edit_snapshot is None:
            return

        self.total_corrections += 1
        edit_modified = 0

        for i, (old, new) in enumerate(zip(self._pre_edit_snapshot, annotations)):
            if i >= len(self.layer_names):
                break
            layer_name = self.layer_names[i]
            added = int(np.count_nonzero((new > 0) & (old == 0)))
            deleted = int(np.count_nonzero((new == 0) & (old > 0)))

            if added > 0:
                self.per_layer[layer_name]["additions"] += 1
                self.per_layer[layer_name]["pixels_added"] += added
                edit_modified += added
            if deleted > 0:
                self.per_layer[layer_name]["deletions"] += 1
                self.per_layer[layer_name]["pixels_deleted"] += deleted
                edit_modified += deleted

        self.total_modified_pixels += edit_modified
        self._pre_edit_snapshot = None

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_total_time(self):
        """Return elapsed correction time in seconds."""
        return time.time() - self.start_time

    def to_log_entry(self):
        """Format the metrics as a human-readable log entry."""
        total_time = self.get_total_time()
        lines = [
            "=== Autolabel Correction Metrics ===",
            f"Image: {self.image_filename}",
            f"Plugin: {self.plugin_id}",
            f"Layers: {', '.join(self.layer_names)}",
            f"Plugin run timestamp: {self.timestamp.isoformat()}",
            f"Total correction time: {total_time:.1f}s",
            f"Total corrections: {self.total_corrections}",
            f"Total modified pixels: {self.total_modified_pixels}",
            "Per-layer metrics:",
        ]
        for name in self.layer_names:
            m = self.per_layer[name]
            lines.append(
                f"  {name}: additions={m['additions']}, "
                f"deletions={m['deletions']}, "
                f"pixels_added={m['pixels_added']}, "
                f"pixels_deleted={m['pixels_deleted']}"
            )
        lines.append("=" * 36)
        return "\n".join(lines)


def log_autolabel_metrics(metrics: AutolabelMetrics, log_dir="annotations"):
    """Append *metrics* to the autolabel log file."""
    logger = logging.getLogger("AutolabelMetrics")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "autolabel_log.txt")
    entry = metrics.to_log_entry()
    try:
        with open(log_file, "a") as f:
            f.write(entry + "\n\n")
        logger.info("Autolabel metrics logged to %s", log_file)
    except Exception as e:
        logger.error("Failed to write autolabel metrics: %s", e)
