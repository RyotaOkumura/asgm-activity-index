"""Shared definitions used by every stage: class schema, annotated years, and band order.

Mask PNG values are 0 = unlabeled and 1..7 = the classes below. They are
remapped on load to 0..6, with unlabeled pixels set to ``IGNORE_INDEX``.
"""

from __future__ import annotations

CLASSES = (
    "vegetation",
    "bare_ground",
    "active_pond",
    "inactive_pond",
    "artifact",
    "cloud",
    "cloud_shadow",
)
NUM_CLASSES = len(CLASSES)
ACTIVE_POND_IDX = CLASSES.index("active_pond")
CLOUD_IDX = CLASSES.index("cloud")
CLOUD_SHADOW_IDX = CLASSES.index("cloud_shadow")
IGNORE_INDEX = 255

ANNOTATED_YEARS = (2016, 2020, 2024)

# Band order inside each image GeoTIFF.
ALL_BANDS = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
)
