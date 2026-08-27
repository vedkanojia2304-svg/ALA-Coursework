# This file includes Steps 6, 7, and 8
# Step 6: Benchmark custom Vec operations
# Step 7: Run benchmarks on PC and Google Colab
# Step 8: Compare custom Vec performance with NumPy

import timeit
import numpy as np

from vec import Vec


# Vector sizes required by the assignment
sizes = [2000, 4000, 8000, 16000, 32000, 64000]


for n in sizes:

    print("\n" + "=" * 50)
    print(f"Vector dimension: {n}")
    print("=" * 50)

    # ----------------------------------------
    # Create data using custom Vec
    # ----------------------------------------

    v1 = Vec.uniform(n)
    v2 = Vec.uniform(n)

    # ----------------------------------------
    # Create NumPy arrays using the SAME data
    # ----------------------------------------

    np_v1 = np.array(v1.elements)
    np_v2 = np.array(v2.elements)

    # ========================================
    # CUSTOM VEC BENCHMARK
    # ========================================

    # Addition
    vec_addition_time = timeit.timeit(
        lambda: v1 + v2,
        number=10
    )

    # Subtraction
    vec_subtraction_time = timeit.timeit(
        lambda: v1 - v2,
        number=10
    )

    # Scalar multiplication
    vec_multiplication_time = timeit.timeit(
        lambda: 2 * v1,
        number=10
    )

    # Norm
    vec_norm_time = timeit.timeit(
        lambda: v1.norm(),
        number=10
    )

    # ========================================
    # NUMPY BENCHMARK
    # ========================================

    # Addition
    numpy_addition_time = timeit.timeit(
        lambda: np_v1 + np_v2,
        number=10
    )

    # Subtraction
    numpy_subtraction_time = timeit.timeit(
        lambda: np_v1 - np_v2,
        number=10
    )

    # Scalar multiplication
    numpy_multiplication_time = timeit.timeit(
        lambda: 2 * np_v1,
        number=10
    )

    # Norm
    numpy_norm_time = timeit.timeit(
        lambda: np.linalg.norm(np_v1),
        number=10
    )

    # ========================================
    # DISPLAY RESULTS
    # ========================================

    print("\nCUSTOM VEC:")

    print(f"Addition:       {vec_addition_time:.6f} seconds")
    print(f"Subtraction:    {vec_subtraction_time:.6f} seconds")
    print(f"Multiplication: {vec_multiplication_time:.6f} seconds")
    print(f"Norm:           {vec_norm_time:.6f} seconds")

    print("\nNUMPY:")

    print(f"Addition:       {numpy_addition_time:.6f} seconds")
    print(f"Subtraction:    {numpy_subtraction_time:.6f} seconds")
    print(f"Multiplication: {numpy_multiplication_time:.6f} seconds")
    print(f"Norm:           {numpy_norm_time:.6f} seconds")