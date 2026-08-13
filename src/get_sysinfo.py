from pathlib import Path
import os
from psutil import virtual_memory, swap_memory, cpu_count


def load_guardrail(silent: bool = False) -> bool:
    """État de charge système : loadavg, RAM dispo, swap. Retourne True si saturé.

    Un bench mémoire (surtout DRAM) mesuré sous charge concurrente est invalide :
    le loadavg sert de garde-fou, seul un seuil de 0.5 × ncores tolère le bruit.
    """
    load = os.getloadavg()[0]
    ncores = cpu_count() or 1
    mem = virtual_memory()
    swap = swap_memory()
    saturated = load > 0.5 * ncores
    if not silent:
        status = "⚠ CHARGE ÉLEVÉE : résultats DRAM non fiables" if saturated else "charge OK"
        print(
            f"[load {load:.1f}/{ncores} cœurs | RAM dispo {mem.available / 1024**3:.0f} GiB "
            f"| swap utilisé {swap.used / 1024**3:.1f} GiB] {status}"
        )
    return saturated


def get_exact_cache_sizes() -> dict[str, int]:
    """Lit les tailles de cache exactes (en octets) depuis SysFS sous Linux."""
    cache_base = Path("/sys/devices/system/cpu/cpu0/cache")
    caches = {}

    if not cache_base.exists():
        return caches

    for idx in cache_base.glob("index*"):
        try:
            level = (idx / "level").read_text().strip()
            cache_type = (idx / "type").read_text().strip().lower()
            size_str = (idx / "size").read_text().strip()

            # Convertit "32K", "512K", "32M" en octets
            size_bytes = int(size_str[:-1])
            unit = size_str[-1].upper()
            if unit == "K":
                size_bytes *= 1024
            elif unit == "M":
                size_bytes *= 1024 * 1024

            key = f"l{level}_{cache_type}" if level == "1" else f"l{level}"
            caches[key] = size_bytes
        except Exception:
            continue

    return caches