from pathlib import Path

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