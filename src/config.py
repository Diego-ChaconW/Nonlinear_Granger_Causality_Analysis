"""
Shared constants and configuration utilities.
"""

# Display names for chaotic maps (used in plot titles)
MAP_DISPLAY_NAMES = {
    "henon": "Hénon",
    "ikeda": "Ikeda",
    "tinkerbell": "Tinkerbell",
    "rulkov": "Rulkov",
}

# Valid options
VALID_MAPS = ["henon", "ikeda", "tinkerbell", "rulkov"]
VALID_DIRECTIONS = ["Y_to_X", "X_to_Y"]
VALID_ARCHITECTURES = ["MLP", "LSTM", "GRU"]

# Map NN architecture names to nonlincausality config codes
NN_CONFIG_MAP = {
    "MLP": "d",   # 'd' = Dense (MLP)
    "LSTM": "l",  # 'l' = LSTM
    "GRU": "g",   # 'g' = GRU
}


def validate_map(chaotic_map):
    """Validate the chaotic map name."""
    if chaotic_map not in VALID_MAPS:
        raise ValueError(
            f"Unknown chaotic map: '{chaotic_map}'. "
            f"Choose from: {VALID_MAPS}"
        )


def validate_direction(direction):
    """Validate the causality direction."""
    if direction not in VALID_DIRECTIONS:
        raise ValueError(
            f"Unknown causality direction: '{direction}'. "
            f"Choose from: {VALID_DIRECTIONS}"
        )


def validate_architecture(architecture):
    """Validate the NN architecture name."""
    if architecture not in VALID_ARCHITECTURES:
        raise ValueError(
            f"Unknown NN architecture: '{architecture}'. "
            f"Choose from: {VALID_ARCHITECTURES}"
        )


def get_direction_labels(direction):
    """
    Get human-readable labels for the causality direction.

    Parameters
    ----------
    direction : str
        Either "Y_to_X" or "X_to_Y".

    Returns
    -------
    dict with keys: target_label, cause_label, direction_str, arrow_label
    """
    validate_direction(direction)
    if direction == "Y_to_X":
        return {
            "target_label": "X",
            "cause_label": "Y",
            "direction_str": "Y → X",
            "arrow_label": "X ← Y",
        }
    else:
        return {
            "target_label": "Y",
            "cause_label": "X",
            "direction_str": "X → Y",
            "arrow_label": "X → Y",
        }
