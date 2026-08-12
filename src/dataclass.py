from dataclasses import dataclass

@dataclass
class SysInfo:

    cpu_model: str
    num_threads: int
    available_memory: float
    kernel: str
    