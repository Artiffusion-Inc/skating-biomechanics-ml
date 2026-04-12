"""ISU element registry — static database of all elements with properties.

Data source: ISU Communication 2707 (2025/26 season).
Singles only (Men + Women), Short Program + Free Skate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ElementType(StrEnum):
    JUMP = "jump"
    SPIN = "spin"
    STEP_SEQUENCE = "step_sequence"
    CHOREO_SEQUENCE = "choreo_sequence"


@dataclass(frozen=True)
class ElementDef:
    code: str
    name: str
    type: ElementType
    base_value: float
    rotations: float = 0.0
    has_toe_pick: bool = False
    entry_edge: str = ""
    exit_edge: str = ""
    combo_eligible: bool = False
    short_program_eligible: bool = True


# ---------------------------------------------------------------------------
# Static element database
# ---------------------------------------------------------------------------

ELEMENTS: dict[str, ElementDef] = {
    # --- Jumps (ISU 2025/26 BV) ---
    # Single jumps
    "1T": ElementDef("1T", "Single Toe Loop", ElementType.JUMP, 0.40, 1.0, True, "", "RBO", False),
    "1S": ElementDef("1S", "Single Salchow", ElementType.JUMP, 0.40, 1.0, False, "", "RBO", False),
    "1Lo": ElementDef("1Lo", "Single Loop", ElementType.JUMP, 0.50, 1.0, False, "", "RBO", False),
    "1F": ElementDef("1F", "Single Flip", ElementType.JUMP, 0.50, 1.0, True, "", "RBO", False),
    "1Lz": ElementDef("1Lz", "Single Lutz", ElementType.JUMP, 0.60, 1.0, True, "", "RBO", False),
    "1A": ElementDef("1A", "Single Axel", ElementType.JUMP, 1.10, 1.5, False, "", "RBO", False),
    # Double jumps
    "2T": ElementDef("2T", "Double Toe Loop", ElementType.JUMP, 1.30, 2.0, True, "", "RBO", True),
    "2S": ElementDef("2S", "Double Salchow", ElementType.JUMP, 1.30, 2.0, False, "", "RBO", True),
    "2Lo": ElementDef("2Lo", "Double Loop", ElementType.JUMP, 1.70, 2.0, False, "", "RBO", True),
    "2F": ElementDef("2F", "Double Flip", ElementType.JUMP, 1.80, 2.0, True, "", "RBO", True),
    "2Lz": ElementDef("2Lz", "Double Lutz", ElementType.JUMP, 2.10, 2.0, True, "", "RBO", True),
    "2A": ElementDef("2A", "Double Axel", ElementType.JUMP, 3.30, 2.5, False, "", "RBO", True),
    # Triple jumps
    "3T": ElementDef("3T", "Triple Toe Loop", ElementType.JUMP, 4.20, 3.0, True, "", "RBO", True),
    "3S": ElementDef("3S", "Triple Salchow", ElementType.JUMP, 4.30, 3.0, False, "", "RBO", True),
    "3Lo": ElementDef("3Lo", "Triple Loop", ElementType.JUMP, 4.90, 3.0, False, "", "RBO", True),
    "3F": ElementDef("3F", "Triple Flip", ElementType.JUMP, 5.30, 3.0, True, "", "RBO", True),
    "3Lz": ElementDef("3Lz", "Triple Lutz", ElementType.JUMP, 5.90, 3.0, True, "", "RBO", True),
    "3A": ElementDef("3A", "Triple Axel", ElementType.JUMP, 8.00, 3.5, False, "", "RBO", True),
    # Quad jumps (Men)
    "4T": ElementDef("4T", "Quad Toe Loop", ElementType.JUMP, 9.50, 4.0, True, "", "RBO", True),
    "4S": ElementDef("4S", "Quad Salchow", ElementType.JUMP, 9.70, 4.0, False, "", "RBO", True),
    "4Lo": ElementDef("4Lo", "Quad Loop", ElementType.JUMP, 10.50, 4.0, False, "", "RBO", True),
    "4F": ElementDef("4F", "Quad Flip", ElementType.JUMP, 11.00, 4.0, True, "", "RBO", True),
    "4Lz": ElementDef("4Lz", "Quad Lutz", ElementType.JUMP, 11.50, 4.0, True, "", "RBO", True),
    "4A": ElementDef("4A", "Quad Axel", ElementType.JUMP, 12.50, 4.5, False, "", "RBO", True),
    # Half jumps (used in combinations)
    "1Eu": ElementDef(
        "1Eu", "Euler (half-loop)", ElementType.JUMP, 0.50, 0.5, False, "", "RBO", True
    ),
    # --- Spins (ISU 2025/26 BV) ---
    # Combination spins
    "CSp1": ElementDef("CSp1", "Change Foot Combination Spin Lv1", ElementType.SPIN, 1.50),
    "CSp2": ElementDef("CSp2", "Change Foot Combination Spin Lv2", ElementType.SPIN, 2.00),
    "CSp3": ElementDef("CSp3", "Change Foot Combination Spin Lv3", ElementType.SPIN, 2.50),
    "CSp4": ElementDef("CSp4", "Change Foot Combination Spin Lv4", ElementType.SPIN, 3.20),
    # Flying spins
    "FSp1": ElementDef("FSp1", "Flying Change Foot Spin Lv1", ElementType.SPIN, 1.70),
    "FSp2": ElementDef("FSp2", "Flying Change Foot Spin Lv2", ElementType.SPIN, 2.30),
    "FSp3": ElementDef("FSp3", "Flying Change Foot Spin Lv3", ElementType.SPIN, 2.80),
    "FSp4": ElementDef("FSp4", "Flying Change Foot Spin Lv4", ElementType.SPIN, 3.00),
    # Layback spins (Women) / Single position spins
    "LSp1": ElementDef("LSp1", "Layback Spin Lv1", ElementType.SPIN, 1.50),
    "LSp2": ElementDef("LSp2", "Layback Spin Lv2", ElementType.SPIN, 2.00),
    "LSp3": ElementDef("LSp3", "Layback Spin Lv3", ElementType.SPIN, 2.50),
    "LSp4": ElementDef("LSp4", "Layback Spin Lv4", ElementType.SPIN, 3.00),
    # Spin in one position (Men)
    "USp1": ElementDef("USp1", "Upright Spin Lv1", ElementType.SPIN, 1.50),
    "USp2": ElementDef("USp2", "Upright Spin Lv2", ElementType.SPIN, 2.00),
    "USp3": ElementDef("USp3", "Upright Spin Lv3", ElementType.SPIN, 2.50),
    "USp4": ElementDef("USp4", "Upright Spin Lv4", ElementType.SPIN, 3.00),
    # Camel spins
    "CSpB1": ElementDef("CSpB1", "Camel Spin Lv1", ElementType.SPIN, 1.70),
    "CSpB2": ElementDef("CSpB2", "Camel Spin Lv2", ElementType.SPIN, 2.30),
    "CSpB3": ElementDef("CSpB3", "Camel Spin Lv3", ElementType.SPIN, 2.80),
    "CSpB4": ElementDef("CSpB4", "Camel Spin Lv4", ElementType.SPIN, 3.00),
    # Step sequences
    "StSq1": ElementDef("StSq1", "Step Sequence Lv1", ElementType.STEP_SEQUENCE, 1.50),
    "StSq2": ElementDef("StSq2", "Step Sequence Lv2", ElementType.STEP_SEQUENCE, 2.60),
    "StSq3": ElementDef("StSq3", "Step Sequence Lv3", ElementType.STEP_SEQUENCE, 3.30),
    "StSq4": ElementDef("StSq4", "Step Sequence Lv4", ElementType.STEP_SEQUENCE, 3.90),
    # Choreographic sequence
    "ChSq1": ElementDef("ChSq1", "Choreographic Sequence", ElementType.CHOREO_SEQUENCE, 3.00),
}


def get_element(code: str) -> ElementDef | None:
    return ELEMENTS.get(code)


def get_elements_by_type(element_type: ElementType) -> list[ElementDef]:
    return [el for el in ELEMENTS.values() if el.type == element_type]


def get_jumps() -> list[ElementDef]:
    return get_elements_by_type(ElementType.JUMP)


def get_spins() -> list[ElementDef]:
    return get_elements_by_type(ElementType.SPIN)
