from dataclasses import dataclass

@dataclass
class SysInfo:

    cpu_model: str
    num_threads: int
    available_memory: float
    kernel: str
    l1_data_size: int
    l1_instr_size: int
    l2_size: int
    l3_size: int
    