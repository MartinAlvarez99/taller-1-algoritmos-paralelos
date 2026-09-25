"""
Taller 1: Algoritmos Paralelos
Curso: Infraestructuras Paralelas y Distribuidas (750023C)

Experimento adicional: mide el rendimiento con tareas de carga pesada
(8 llamadas a fibonacci(30)) para comparar frente a los resultados
obtenidos con N=20, y así evaluar el impacto real del paralelismo
cuando cada tarea toma un tiempo considerable.
"""

import os
import time
import concurrent.futures

from fibonacci_paralelo import fibonacci


def prueba_carga_pesada(n_tareas: int, valor_n: int, executor_type=None):
    """
    Corre n_tareas llamadas a fibonacci(valor_n), de forma serial o en
    paralelo según el executor_type recibido.
    executor_type=None -> versión serial.
    """
    inicio = time.perf_counter()
    if executor_type is None:
        resultados = [fibonacci(valor_n) for _ in range(n_tareas)]
    else:
        with executor_type() as executor:
            resultados = list(executor.map(fibonacci, [valor_n] * n_tareas))
    fin = time.perf_counter()
    return resultados, fin - inicio


if __name__ == "__main__":
    nucleos = os.cpu_count()
    print(f"Núcleos de CPU disponibles: {nucleos}\n")

    N_TAREAS = 8
    VALOR_N = 30  # cada tarea calcula fibonacci(30): trabajo sustancial

    print(f"Ejecutando {N_TAREAS} tareas de fibonacci({VALOR_N}) cada una...\n")

    _, t_serial = prueba_carga_pesada(N_TAREAS, VALOR_N)
    print(f"Serial:                {t_serial:.4f} s")

    _, t_hilos = prueba_carga_pesada(N_TAREAS, VALOR_N, concurrent.futures.ThreadPoolExecutor)
    print(f"Paralelo (hilos):      {t_hilos:.4f} s  (speedup {t_serial / t_hilos:.2f}x)")

    _, t_procesos = prueba_carga_pesada(N_TAREAS, VALOR_N, concurrent.futures.ProcessPoolExecutor)
    print(f"Paralelo (procesos):   {t_procesos:.4f} s  (speedup {t_serial / t_procesos:.2f}x)")