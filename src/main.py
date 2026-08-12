import numpy as np 
from time import perf_counter
from cpuinfo import get_cpu_info
import platform
from psutil import virtual_memory, cpu_count

from dataclass import SysInfo
import runs


GB = (1024 * 1024 * 1024)

info = get_cpu_info()

times = []
GFlops_list = []
num_iter = 10

sysinfo = SysInfo(
    info["brand_raw"],
    cpu_count(),
    round((virtual_memory().available / GB), 2),
    platform.platform()
)



print(sysinfo.cpu_model)
print(sysinfo.num_threads)
print(sysinfo.available_memory)
print(sysinfo.kernel)




# runs.GEMM_run(8000, 15)
# runs.STREAM_copy_run(int(1e8), 10)
# runs.STREAM_scale_run(int(1e8), 15)
# runs.STREAM_add_run(int(1e8), 25)
runs.STREAM_run(int(1e9), 20)

