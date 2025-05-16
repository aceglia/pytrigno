from enum import IntEnum


class EMGType(IntEnum):
    """
    Give the data port according of the EMG sensor type.
    """
    Avanti = 50043
    Legacy = 50041
