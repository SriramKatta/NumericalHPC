/*
 * Copyright (C) 2022 NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved.
 * Use of this source code is governed by a MIT-style
 * license that can be found in the LICENSE file.
 */
#include "timing.h"
#include "util.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void localDMVM(size_t Nlocal, int N, double *__restrict__ y,
               const double *__restrict__ a, const double *__restrict__ x,
               size_t cs) {

  for (int r = 0; r < Nlocal; r++) {
    for (int c = cs; c < N + Nlocal; c++) {
      y[r] = y[r] + a[r * N + c] * x[c - cs];
    }
  }
}

double dmvm(double *restrict y, const double *restrict a,
            double *restrict x, int N, int iter) {
  double ts, te;
  int rank = 0, size = 1;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  size_t Nlocal = Ningivenrank(N, rank, size);
  size_t cs = chunkstart(N, rank, size);
  int nextnbr = (rank + 1) % size;
  int prevnbr = (rank + size - 1) % size;
  int Nlocprev = Ningivenrank(N, prevnbr, size);

  ts = MPI_Wtime();
  for (int j = 0; j < iter; j++) {
    for (size_t rot = 0; rot < size; rot++) {

      localDMVM(Nlocal, N, y, a, x, cs);
      if (rot != size - 1) {
        MPI_Status status;
        if (ROOT == rank) {
          MPI_CHECK(
              MPI_Send(x, Nlocal, MPI_DOUBLE, nextnbr, 0, MPI_COMM_WORLD));
          MPI_CHECK(MPI_Recv(x, Nlocprev, MPI_DOUBLE, prevnbr, 0,
                             MPI_COMM_WORLD, &status));
        } else {
          MPI_CHECK(MPI_Recv(x, Nlocprev, MPI_DOUBLE, prevnbr, 0,
                             MPI_COMM_WORLD, &status));
          MPI_CHECK(
              MPI_Send(x, Nlocal, MPI_DOUBLE, nextnbr, 0, MPI_COMM_WORLD));
        }
      }

#ifdef CHECK
      {
        double sum = 0.0;

        for (int i = 0; i < N; i++) {
          sum += y[i];
          y[i] = 0.0;
        }
        fprintf(stderr, "Sum: %f\n", sum);
      }
#endif
    }
  }
  te = MPI_Wtime();

  return te - ts;
}