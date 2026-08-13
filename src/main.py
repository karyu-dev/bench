import numpy as np 
from time import perf_counter
from cpuinfo import get_cpu_info
import platform
from psutil import virtual_memory, cpu_count

from dataclass import SysInfo
import runs
from disk_bench import disk_benchmark_run


GB = (1024 * 1024 * 1024)

info = get_cpu_info()
# print (info)

sysinfo = SysInfo(
    info["brand_raw"],
    cpu_count(),
    round((virtual_memory().available / GB), 2),
    platform.platform(),
    info["l1_data_cache_size"],
    info["l1_instruction_cache_size"],
    info["l2_cache_size"],
    info["l3_cache_size"] # BUG SUR AMD, A CHANGER PAR UNE LECTURE DE /sys/devices/system/cpu/cpu0/cache/index3/size
)



print(sysinfo.cpu_model)
print(sysinfo.num_threads)
print(sysinfo.available_memory)
print(sysinfo.kernel)
print(sysinfo.l1_data_size)
print(sysinfo.l1_instr_size)
print(sysinfo.l2_size)
print(sysinfo.l3_size)





# runs.GEMM_run(8000, 15)

# Paliers L1/L2/L3/DRAM (footprint vs bande passante)
# runs.STREAM_sweep("triad", it=5)
# print()

# runs.STREAM_run_DRAM(int(1e9), 10, 1)

disk_benchmark_run()