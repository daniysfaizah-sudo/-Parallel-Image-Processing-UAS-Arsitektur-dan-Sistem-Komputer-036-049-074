import subprocess
import re

# Konfigurasi
IMAGES = [
    ("bendera_512.ppm",  "512x512"),
    ("bendera_1024.ppm", "1024x1024"),
    ("bendera_2048.ppm", "2048x2048"),

]
THREADS = [2, 4]

def parse_total(output):
    match = re.search(r"Total\s*:\s*([\d.]+)\s*ms", output)
    return float(match.group(1)) if match else None

def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout + result.stderr

print("=" * 60)
print("  BENCHMARK: Parallel Image Processing")
print("  Sequential vs OpenMP vs OpenCL")
print("=" * 60)

for img_path, size in IMAGES:
    print(f"\n--- Ukuran: {size} ---")

    # Sequential
    out = run(["./sequential.exe", img_path, "tmp_gray.ppm", "tmp_sobel.ppm"])
    t_seq = parse_total(out)
    print(f"  Sequential   : {t_seq:.2f} ms")

    # OpenMP
    for t in THREADS:
        out = run(["./openmp_filter.exe", img_path, "tmp_gray.ppm", "tmp_sobel.ppm", str(t)])
        t_omp = parse_total(out)
        speedup = t_seq / t_omp if t_omp else 0
        print(f"  OpenMP {t}T    : {t_omp:.2f} ms  |  Speedup: {speedup:.2f}x")

    # OpenCL
    out = run(["./opencl_filter.exe", img_path, "tmp_gray.ppm", "tmp_sobel.ppm"])
    t_ocl = parse_total(out)
    if t_ocl:
        speedup = t_seq / t_ocl
        print(f"  OpenCL       : {t_ocl:.2f} ms  |  Speedup: {speedup:.2f}x")
    else:
        print("  OpenCL       : tidak tersedia")

print("\n" + "=" * 60)
print("  Benchmark selesai!")
print("=" * 60)