import ctypes
import os
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import numpy as np

from get_sysinfo import load_guardrail


def bench_memory_allocation(size_gb: float = 2.0, with_rng: bool = False,
                            with_diag: bool = False) -> dict[str, float]:
    """Mesure la vitesse d'allocation mémoire virtuelle vs réelle (First-Touch) vs NumPy Random.

    with_diag=True ajoute des mesures diagnostiques pour isoler le goulot :
      D1. second-touch        -> bande passante pure d'écriture (zéro page fault)
      D2. first-touch MT      -> le page faulting se parallélise-t-il ?
      D3. first-touch THP OFF -> pages 4 Ko forcées : isole la compaction THP
                                 (preuve pour ticket sysadmin, sans root)
    """
    n_elements = int((size_gb * 1024**3) // 8)  # Nombre d'éléments float64
    bytes_allocated = n_elements * 8
    actual_gib = bytes_allocated / (1024**3)

    load_guardrail()

    rng_tag = "" if with_rng else " (RNG désactivé)"
    print(f"--- Benchmark ALLOCATION MEMOIRE ({actual_gib:.2f} GiB){rng_tag} ---")

    # 1. Allocation Virtuelle Pure (malloc / mmap paresseux)
    t0 = perf_counter()
    arr = np.empty(n_elements, dtype=np.float64)
    dt_virt = perf_counter() - t0
    virt_speed_gbps = actual_gib / max(dt_virt, 1e-9)
    print(
        f"  1. Allocation Virtuelle (np.empty) : {dt_virt*1000:.3f} ms ({virt_speed_gbps:.1f} GiB/s)"
    )

    # 2. First-Touch / Kernel Page Faulting (Écriture physique dans la RAM)
    t0 = perf_counter()
    arr.fill(1.0)  # Déclenche les Page Faults Linux et l'allocation physique
    dt_touch = perf_counter() - t0
    touch_speed_gbps = actual_gib / dt_touch
    print(
        f"  2. First-Touch (Page Faults OS)   : {dt_touch*1000:.1f} ms ({touch_speed_gbps:.2f} GiB/s)"
    )

    metrics = {
        "virtual_alloc_ms": dt_virt * 1000,
        "first_touch_gbps": touch_speed_gbps,
    }

    # 3. Génération Random NumPy (Génération de données) — CPU-bound, optionnel
    if with_rng:
        t0 = perf_counter()
        _ = np.random.default_rng().random(n_elements, dtype=np.float64)
        dt_rng = perf_counter() - t0
        rng_speed_gbps = actual_gib / dt_rng
        print(
            f"  3. Génération Random (rng.random) : {dt_rng*1000:.1f} ms ({rng_speed_gbps:.2f} GiB/s)"
        )
        metrics["rng_generation_gbps"] = rng_speed_gbps

    # --- Mode diagnostic ---
    if with_diag:
        # D1. Second-touch : mêmes pages, déjà résidentes -> bande passante pure
        t0 = perf_counter()
        arr.fill(1.0)
        dt = perf_counter() - t0
        d1 = actual_gib / dt
        print(
            f"  D1. Second-Touch (bande passante pure)  : {dt*1000:.1f} ms ({d1:.2f} GiB/s)"
        )

        # D2. First-touch multi-thread : parallélise le page faulting (per-node/socket)
        nthreads = min(os.cpu_count() or 4, 64)
        arr2 = np.empty(n_elements, dtype=np.float64)
        chunk = (n_elements + nthreads - 1) // nthreads

        def _fill_chunk(i: int) -> None:
            arr2[i * chunk:(i + 1) * chunk].fill(1.0)

        t0 = perf_counter()
        with ThreadPoolExecutor(max_workers=nthreads) as ex:
            list(ex.map(_fill_chunk, range(nthreads)))
        dt = perf_counter() - t0
        d2 = actual_gib / dt
        print(
            f"  D2. First-Touch Multi-Thread ({nthreads} th) : {dt*1000:.1f} ms ({d2:.2f} GiB/s)"
        )

        # D3. First-touch THP désactivé (MADV_NOHUGEPAGE) : force les pages 4 Ko
        #     Sur un serveur THP=always + defrag=always, chaque fault tente une
        #     page 2 Mo avec compaction => stall. Si D3 est nettement plus rapide
        #     que le first-touch normal, la compaction THP est le goulot (preuve
        #     pour le ticket sysadmin, sans besoin de root).
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        MADV_NOHUGEPAGE = 15
        PAGE = 4096
        raw = np.empty(n_elements + PAGE // 8, dtype=np.float64)
        aligned_off = (-raw.ctypes.data) % PAGE  # madvise exige une base alignée page
        arr3 = raw[aligned_off // 8: aligned_off // 8 + n_elements]
        rc = libc.madvise(ctypes.c_void_p(arr3.ctypes.data),
                          ctypes.c_size_t(n_elements * 8),
                          ctypes.c_int(MADV_NOHUGEPAGE))
        errno = ctypes.get_errno() if rc != 0 else 0
        t0 = perf_counter()
        arr3.fill(1.0)
        dt = perf_counter() - t0
        d3 = actual_gib / dt
        print(
            f"  D3. First-Touch THP OFF (NOHUGEPAGE) : {dt*1000:.1f} ms ({d3:.2f} GiB/s) "
            f"(madvise rc={rc} errno={errno})"
        )

        metrics.update({
            "second_touch_gbps": d1,
            "mt_first_touch_gbps": d2,
            "thp_off_first_touch_gbps": d3,
        })

    return metrics


if __name__ == "__main__":
    bench_memory_allocation(size_gb=4.0)
