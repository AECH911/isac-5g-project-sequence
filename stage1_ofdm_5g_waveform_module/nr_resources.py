from dataclasses import dataclass

SUBCARRIERS_PER_RB = 12

@dataclass
class NRResourceConfig:
    subcarrier_spacing_hz: float
    num_rbs: int = 1
    subcarriers_per_rb: int = SUBCARRIERS_PER_RB

    @property
    def total_subcarriers(self) -> int:
        return self.num_rbs * self.subcarriers_per_rb

    @property
    def rb_bandwidth_hz(self) -> float:
        return self.subcarriers_per_rb * self.subcarrier_spacing_hz

    @property
    def total_bandwidth_hz(self) -> float:
        return self.total_subcarriers * self.subcarrier_spacing_hz


def get_active_bins(num_rbs: int, subcarriers_per_rb: int = SUBCARRIERS_PER_RB):
    total_subcarriers = num_rbs * subcarriers_per_rb
    start = -(total_subcarriers // 2)
    return list(range(start, start + total_subcarriers))