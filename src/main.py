#!/usr/bin/env python3
"""MAIN.PY - Suite de Profilage & Benchmarking HPC Complet (CPU / RAM / Cache / GPU / Disque / OS)"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Interface Rich
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def print_header():
    title = "[bold cyan]HPC SYSTEM & MICRO-ARCHITECTURE BENCHMARK SUITE[/bold cyan]"
    subtitle = (
        "[dim]Calcul Matriciel | STREAM | Caches | Latence | Multi-Thread | "
        "Predictor | Direct I/O | GPU[/dim]"
    )
    console.print(
        Panel(f"{title}\n{subtitle}", expand=False, border_style="cyan")
    )


def run_all_benchmarks(args) -> dict:
    results = {
        "timestamp": datetime.now().isoformat(),
        "platform": sys.platform,
        "cpu_count": os.cpu_count(),
        "benchmarks": {},
    }

    selected = args.bench.lower()

    # 1. COMPUTE PEAK FMA
    if selected in ["all", "fma", "cpu"]:
        try:
            from fma_bench import bench_fma_peak_single_core

            console.rule("[bold green]1. COMPUTE | Peak FMA FP64 (1 Cœur)")
            results["benchmarks"]["fma_fp64"] = bench_fma_peak_single_core()
        except Exception as e:
            console.print(f"[bold red]Erreur FMA:[bold red] {e}")

    # 3. BALAYAGE CACHES (L1 -> DRAM)
    if selected in ["all", "sweep", "memory", "cache"]:
        try:
            from runs import STREAM_sweep

            console.rule(
                "[bold green]3. CACHE | Hierarchy Sweep (L1d -> DRAM)"
            )
            results["benchmarks"]["cache_sweep"] = (
                STREAM_sweep()
            )
        except Exception as e:
            console.print(f"[bold red]Erreur Cache Sweep:[bold red] {e}")

    # 4. LATENCE MEMOIRE (POINTER CHASING)
    if selected in ["all", "latency", "memory"]:
        try:
            from latency_bench import bench_memory_latency_sweep

            console.rule(
                "[bold green]4. LATENCY | Pointer Chasing (L1 -> DRAM ns)"
            )
            results["benchmarks"]["latency_ns"] = bench_memory_latency_sweep()
        except Exception as e:
            console.print(f"[bold red]Erreur Latence:[bold red] {e}")

    # 5. MULTI-THREAD SCALING (OPENMP)
    if selected in ["all", "scaling", "cpu"]:
        try:
            from thread_scaling_bench import bench_thread_scaling

            console.rule(
                "[bold green]5. SCALING | Multi-Thread Scaling (OpenMP)"
            )
            bench_thread_scaling()
        except Exception as e:
            console.print(f"[bold red]Erreur Thread Scaling:[bold red] {e}")

    # 6. MICRO-ARCHITECTURE (BRANCH PREDICTION & ALU INT64)
    if selected in ["all", "branch", "cpu"]:
        try:
            from branch_alu_bench import (
                bench_branch_prediction,
                bench_integer_alu,
            )

            console.rule("[bold green]6. MICRO-ARCH | Branch Predictor & ALU")
            results["benchmarks"]["branch_predictor"] = (
                bench_branch_prediction()
            )
            results["benchmarks"]["integer_alu"] = bench_integer_alu()
        except Exception as e:
            console.print(f"[bold red]Erreur Branch/ALU:[bold red] {e}")

    # 7. OS & MEMORY ALLOCATION
    if selected in ["all", "alloc", "os"]:
        try:
            from alloc_bench import bench_memory_allocation

            console.rule(
                "[bold green]7. OS & MEMORY | Alloc / First-Touch / RNG"
            )
            results["benchmarks"]["allocation"] = bench_memory_allocation()
        except Exception as e:
            console.print(f"[bold red]Erreur Alloc:[bold red] {e}")

    # 8. STOCKAGE DIRECT I/O
    if selected in ["all", "disk", "storage"]:
        try:
            import shutil
            from disk_bench import disk_benchmark_run

            console.rule(
                "[bold green]8. STORAGE | Direct I/O (NVMe / SSD / HDD)"
            )
            # ✅ On crée un dossier dédié pour le test disque
            bench_dir = Path("./tmp_bench_dir")
            bench_dir.mkdir(parents=True, exist_ok=True)

            results["benchmarks"]["disk"] = disk_benchmark_run(bench_dir)

            # Nettoyage du dossier temporaire
            if bench_dir.exists():
                shutil.rmtree(bench_dir)
        except Exception as e:
            console.print(f"[bold red]Erreur Disque:[bold red] {e}")

    # 9. GPU BENCHMARK (CUPY)
    if selected in ["all", "gpu"]:
        try:
            from gpu_bench import bench_gpu_cupy

            console.rule("[bold green]9. ACCELERATION | GPU CuPy CUDA")
            res_gpu = bench_gpu_cupy()
            if res_gpu:
                results["benchmarks"]["gpu"] = res_gpu
        except Exception as e:
            console.print(
                f"[bold yellow]GPU non testé ou non disponible:[bold yellow] {e}"
            )

    return results


def print_summary_table(results: dict):
    console.rule("[bold cyan]SYNTHÈSE DES PERFORMANCES")

    table = Table(
        title="Bilan Synthétique du Système",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Domaine", style="cyan", width=22)
    table.add_column("Épreuve / Métrique", style="white")
    table.add_column("Résultat Mesuré", style="bold green", justify="right")

    b = results.get("benchmarks", {})

    if "fma_fp64" in b:
        table.add_row(
            "CPU Compute",
            "Peak FMA FP64 (1 Cœur)",
            f"{b['fma_fp64']['gflops']:.2f} GFLOPS",
        )
    if "integer_alu" in b:
        table.add_row(
            "CPU Compute",
            "Peak INT64 AVX2 ALU",
            f"{b['integer_alu']['giops']:.2f} GIOPS",
        )
    if "branch_predictor" in b:
        table.add_row(
            "Micro-Architecture",
            "Branch Misprediction Penalty",
            f"{b['branch_predictor']['penalty_ns']:.2f} ns (~{b['branch_predictor']['cycles']:.1f} cycles)",
        )
    if "stream" in b:
        triad_speed = b["stream"].get("Triad", {}).get("speed_gbps", 0.0)
        table.add_row(
            "Bande Passante RAM", "STREAM Triad (C-API)", f"{triad_speed:.2f} GB/s"
        )
    if "latency_ns" in b:
        dram_lat = b["latency_ns"].get("128 MiB", 0.0)
        table.add_row(
            "Latence Mémoire",
            "Accès DRAM imprévisible",
            f"{dram_lat:.2f} ns",
        )
    if "allocation" in b:
        touch_speed = b["allocation"].get("first_touch_gbps", 0.0)
        table.add_row(
            "OS & Noyau Linux",
            "First-Touch (Page Faults)",
            f"{touch_speed:.2f} GiB/s",
        )
    if "disk" in b:
        seq_r = b["disk"].get("seq_read_mbps", 0.0)
        iops = b["disk"].get("rand_read_4k_iops", 0)
        table.add_row("Stockage NVMe", "Lecture Séquentielle", f"{seq_r:.1f} MB/s")
        table.add_row("Stockage NVMe", "Lecture Aléatoire 4K", f"{iops:.0f} IOPS")
    if "gpu" in b:
        table.add_row(
            "GPU Accelerateur",
            "Peak GEMM FP32 (cuBLAS)",
            f"{b['gpu']['fp32_tflops']:.2f} TFLOPS",
        )
        table.add_row(
            "GPU Accelerateur",
            "Bande Passante VRAM",
            f"{b['gpu']['vram_gbps']:.2f} GB/s",
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Suite de benchmark HPC complète"
    )
    parser.add_argument(
        "--bench",
        "-b",
        default="all",
        choices=[
            "all",
            "cpu",
            "memory",
            "storage",
            "os",
            "gpu",
            "fma",
            "stream",
            "sweep",
            "latency",
            "scaling",
            "branch",
            "alloc",
            "disk",
        ],
        help="Benchmark spécifique à exécuter (default: all)",
    )
    parser.add_argument(
        "--export",
        "-e",
        type=str,
        default=None,
        help="Chemin du fichier JSON pour exporter les résultats (ex: resultats.json)",
    )

    args = parser.parse_args()

    print_header()
    results = run_all_benchmarks(args)
    print_summary_table(results)

    if args.export:
        export_path = Path(args.export)
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        console.print(
            f"\n[bold green]✓ Résultats exportés avec succès dans :[bold green] [yellow]{export_path.resolve()}[/yellow]"
        )


if __name__ == "__main__":
    main()