#pragma omp parallel sections{
        #pragma omp section
        #pragma omp parallel for private(j) schedule(static)
            for (i = 0; i < m; i++)
            {
                rowterm[i] = 0.0;
        #pragma omp parallel for reduction(+: rowterm[i])
                for (j = 0; j < p; j++)
                    rowterm[i] += a[i][2 * j] * a[i][2 * j + 1];
            }

        #pragma omp section
        #pragma omp parallel for private(j) schedule(static)
            for (i = 0; i < q; i++)
            {
                colterm[i] = 0.0;
        #pragma omp parallel for reduction(+:colterm[i])
                for (j = 0; j < p; j++)
                    colterm[i] += b[2 * j][i] * b[2 * j + 1][i];
            }
}

