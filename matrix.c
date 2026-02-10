#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000 // Matrix size

int main() {
    double **a = malloc(N * sizeof(double *));
    double **b = malloc(N * sizeof(double *));
    double **c = malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        a[i] = malloc(N * sizeof(double));
        b[i] = malloc(N * sizeof(double));
        c[i] = malloc(N * sizeof(double));
    }

    // Initialize matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = i + j;
            b[i][j] = i - j;
            c[i][j] = 0.0;
        }
    }

    double start = omp_get_wtime();

    // Heavy Matrix Multiplication
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    double end = omp_get_wtime();
    printf("Threads used: %d\n", omp_get_max_threads());
    printf("Time taken: %f seconds\n", end - start);

    return 0;
}
