"""Device contracts shared by engines and future runtime services."""

from dataclasses import dataclass
from enum import StrEnum


class DeviceBackend(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    XPU = "xpu"
    CUDA = "cuda"


class Precision(StrEnum):
    AUTO = "auto"
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    """A requested or resolved inference device."""

    backend: DeviceBackend = DeviceBackend.AUTO
    index: int | None = None
    precision: Precision = Precision.AUTO

    def __post_init__(self) -> None:
        if self.index is not None and self.index < 0:
            raise ValueError("device index cannot be negative")
        if self.backend in {DeviceBackend.AUTO, DeviceBackend.CPU} and self.index is not None:
            raise ValueError(f"{self.backend.value} devices cannot have an index")
