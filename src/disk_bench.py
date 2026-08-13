import mmap
import os
import random
from pathlib import Path
from time import perf_counter


def create_aligned_buffer(size_bytes: int) -> mmap.mmap:
    """Crée un tampon mémoire anonyme aligné sur 4096 octets (requis pour O_DIRECT)."""
    buf = mmap.mmap(-1, size_bytes)
    # Remplit le buffer avec des données aléatoires pour éviter les compressions matérielles
    buf.write(os.urandom(size_bytes))
    buf.seek(0)
    return buf


def bench_disk_sequential_write(
    filepath: Path, file_size_mb: int = 1024, block_size_mb: int = 1
) -> float:
    """Mesure la vitesse d'écriture séquentielle directe (MB/s)."""
    block_bytes = block_size_mb * 1024 * 1024
    total_bytes = file_size_mb * 1024 * 1024
    num_blocks = total_bytes // block_bytes

    buf = create_aligned_buffer(block_bytes)

    # O_DIRECT : Bypass le cache RAM de Linux | O_SYNC : Attend l'écriture physique
    flags = os.O_WRONLY | os.O_CREAT | os.O_DIRECT | os.O_TRUNC
    fd = os.open(filepath, flags, 0o666)

    try:
        start = perf_counter()
        for _ in range(num_blocks):
            os.write(fd, buf)
        os.fdatasync(fd)  # S'assure que le contrôleur a tout vidé sur la mémoire flash
        elapsed = perf_counter() - start
    finally:
        os.close(fd)

    speed_mbps = (total_bytes / (1024 * 1024)) / elapsed
    return speed_mbps


def bench_disk_sequential_read(
    filepath: Path, file_size_mb: int = 1024, block_size_mb: int = 1
) -> float:
    """Mesure la vitesse de lecture séquentielle directe (MB/s)."""
    block_bytes = block_size_mb * 1024 * 1024
    total_bytes = file_size_mb * 1024 * 1024
    num_blocks = total_bytes // block_bytes

    # Tampon mmap garanti aligné sur la page (4096 octets)
    buf = create_aligned_buffer(block_bytes)

    flags = os.O_RDONLY | os.O_DIRECT
    fd = os.open(filepath, flags)

    try:
        start = perf_counter()
        for _ in range(num_blocks):
            # ✅ FIX : os.readv lit directement dans le buffer mmap aligné
            os.readv(fd, [buf])
        elapsed = perf_counter() - start
    finally:
        os.close(fd)

    speed_mbps = (total_bytes / (1024 * 1024)) / elapsed
    return speed_mbps


def bench_disk_random_4k_read(
    filepath: Path, file_size_mb: int = 512, iterations: int = 10000
) -> dict[str, float]:
    """Mesure les performances en lecture aléatoire 4 KiB (IOPS et Latence)."""
    block_bytes = 4096  # 4 KiB
    total_bytes = file_size_mb * 1024 * 1024
    max_blocks = (total_bytes // block_bytes) - 1

    # Tampon 4 KiB mmap aligné
    buf = create_aligned_buffer(block_bytes)

    flags = os.O_RDONLY | os.O_DIRECT
    fd = os.open(filepath, flags)

    random_offsets = [
        random.randint(0, max_blocks) * block_bytes for _ in range(iterations)
    ]

    try:
        start = perf_counter()
        for offset in random_offsets:
            # ✅ FIX : os.preadv lit directement à l'offset dans le buffer mmap aligné
            os.preadv(fd, [buf], offset)
        elapsed = perf_counter() - start
    finally:
        os.close(fd)

    iops = iterations / elapsed
    avg_latency_us = (elapsed / iterations) * 1e6

    return {"iops": iops, "avg_latency_us": avg_latency_us}

def disk_benchmark_run(
    target_dir: str = "./", file_size_mb: int = 1024
) -> dict[str, float]:
    """Exécute la suite complète I/O et nettoie le fichier temporaire."""
    test_file = Path(target_dir) / ".bench_scratch_file.tmp"

    print(f"--- Benchmark DISK I/O (Fichier de test: {file_size_mb} MiB) ---")
    try:
        # 1. Écriture Séquentielle
        seq_write = bench_disk_sequential_write(test_file, file_size_mb)
        print(f"  Écriture Séquentielle (1 MiB) : {seq_write:.2f} MB/s")

        # 2. Lecture Séquentielle
        seq_read = bench_disk_sequential_read(test_file, file_size_mb)
        print(f"  Lecture Séquentielle   (1 MiB) : {seq_read:.2f} MB/s")

        # 3. Lecture Aléatoire 4K
        rand_4k = bench_disk_random_4k_read(
            test_file, file_size_mb, iterations=5000
        )
        print(
            f"  Lecture Aléatoire 4K           : {rand_4k['iops']:.0f} IOPS (Latence: {rand_4k['avg_latency_us']:.1f} µs)"
        )

        return {
            "seq_write_mbps": seq_write,
            "seq_read_mbps": seq_read,
            "rand_read_4k_iops": rand_4k["iops"],
            "rand_read_4k_latency_us": rand_4k["avg_latency_us"],
        }
    finally:
        # Suppression impérative du fichier temporaire de test
        if test_file.exists():
            test_file.unlink()