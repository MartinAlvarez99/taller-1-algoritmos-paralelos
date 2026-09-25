"""
Taller 1: Algoritmos Paralelos
Curso: Infraestructuras Paralelas y Distribuidas (750023C)
Reto de Programación: Cálculo paralelo de números de Fibonacci

Autor: Martín
"""

import time
import concurrent.futures

N = 20  # Cantidad de números de Fibonacci a calcular (0 .. N-1)


# ---------------------------------------------------------------------------
# 1. Cálculo de Fibonacci
# ---------------------------------------------------------------------------
def fibonacci(n: int) -> int:
    """
    Calcula el n-ésimo número de Fibonacci de forma recursiva.

    NOTA sobre paralelización interna:
    Esta función NO se paraleliza internamente porque cada llamada
    recursiva fibonacci(n-1) y fibonacci(n-2) sería, en teoría, paralelizable
    entre sí (son independientes una de la otra), PERO el costo de crear
    hilos/procesos para tareas tan pequeñas y numerosas (miles de llamadas
    recursivas para valores de n grandes) supera ampliamente el beneficio:
    el overhead de sincronización mata el rendimiento. Por eso el paralelismo
    se aplica en un nivel más "grueso" (coarse-grained): el ciclo externo
    que calcula varios números de Fibonacci independientes entre sí.
    """
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


# ---------------------------------------------------------------------------
# 2. y 3. Paralelización del ciclo + Evitar trampas seriales
# ---------------------------------------------------------------------------
def calcular_fibonacci_serial(n_elementos: int) -> tuple[list[int], float]:
    """
    Versión SERIAL (baseline) para poder comparar el rendimiento.
    El ciclo `for` aquí es puramente secuencial: cada iteración espera
    a que termine la anterior antes de comenzar.
    """
    inicio = time.perf_counter()
    resultados = [fibonacci(i) for i in range(n_elementos)]
    fin = time.perf_counter()
    return resultados, fin - inicio


def calcular_fibonacci_paralelo(n_elementos: int, executor_type) -> tuple[list[int], float]:
    """
    Versión PARALELA.

    Ciclo paralelizable: el ciclo que lanza una tarea fibonacci(i) por cada
    i en range(n_elementos). Es paralelizable porque cada tarea es
    completamente INDEPENDIENTE de las demás: no hay dependencias de datos
    entre calcular fibonacci(3) y fibonacci(7), ni se comparte estado mutable
    entre ellas.

    Trampa serial evitada: NO se hace `print` dentro de las tareas que
    corren en los hilos/procesos. Cada tarea únicamente calcula y retorna
    un valor (return, no print). El ensamblado y la impresión del resultado
    final ocurre en el hilo principal, después de recolectar todos los
    `future.result()`, evitando así:
      - Condiciones de carrera sobre la salida estándar (stdout).
      - Resultados intercalados o en desorden.
      - La necesidad de locks para sincronizar la impresión.

    Orden garantizado: en vez de usar `as_completed` (que devuelve las
    tareas en el orden en que terminan, NO en el orden en que se enviaron),
    se usa `executor.map`, que SÍ conserva el orden original de los
    argumentos. Esto evita otra trampa serial sutil: publicar resultados
    fuera de orden.
    """
    inicio = time.perf_counter()

    with executor_type() as executor:
        # executor.map conserva el orden de entrada -> resultados[i] corresponde a fibonacci(i)
        resultados = list(executor.map(fibonacci, range(n_elementos)))

    fin = time.perf_counter()
    return resultados, fin - inicio


# ---------------------------------------------------------------------------
# 4. Medición de rendimiento + impresión final (fuera de los hilos/procesos)
# ---------------------------------------------------------------------------
def ejecutar_y_reportar(nombre: str, resultados: list[int], tiempo: float) -> None:
    print(f"--- {nombre} ---")
    print(f"Fibonacci ({len(resultados)} términos): {resultados}")
    print(f"Tiempo de ejecución: {tiempo:.4f} segundos\n")


if __name__ == "__main__":
    print(f"Calculando los primeros {N} números de Fibonacci...\n")

    # Baseline serial
    resultados_serial, tiempo_serial = calcular_fibonacci_serial(N)
    ejecutar_y_reportar("SERIAL", resultados_serial, tiempo_serial)

    # Paralelo con hilos (ThreadPoolExecutor)
    resultados_hilos, tiempo_hilos = calcular_fibonacci_paralelo(
        N, concurrent.futures.ThreadPoolExecutor
    )
    ejecutar_y_reportar("PARALELO - ThreadPoolExecutor", resultados_hilos, tiempo_hilos)

    # Paralelo con procesos (ProcessPoolExecutor)
    resultados_procesos, tiempo_procesos = calcular_fibonacci_paralelo(
        N, concurrent.futures.ProcessPoolExecutor
    )
    ejecutar_y_reportar("PARALELO - ProcessPoolExecutor", resultados_procesos, tiempo_procesos)

    # Verificación de correctitud: los tres métodos deben dar el mismo resultado
    assert resultados_serial == resultados_hilos == resultados_procesos
    print("Verificación: los tres métodos produjeron resultados idénticos. ✔")

    # Resumen comparativo de speedup
    print("\n--- RESUMEN ---")
    print(f"Serial:                     {tiempo_serial:.4f} s")
    print(f"Paralelo (hilos):           {tiempo_hilos:.4f} s  "
          f"(speedup {tiempo_serial / tiempo_hilos:.2f}x)")
    print(f"Paralelo (procesos):        {tiempo_procesos:.4f} s  "
          f"(speedup {tiempo_serial / tiempo_procesos:.2f}x)")
