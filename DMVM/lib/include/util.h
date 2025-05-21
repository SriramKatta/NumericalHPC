/*
 * Copyright (C) 2022 NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved.
 * Use of this source code is governed by a MIT-style
 * license that can be found in the LICENSE file.
 */
#ifndef __UTIL_H_
#define __UTIL_H_

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define HLINE                                                                  \
  "--------------------------------------------------------------------------" \
  "--\n"

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif
#ifndef ABS
#define ABS(a) ((a) >= 0 ? (a) : -(a))
#endif

#define ROOT 0

#define MPI_CHECK(call)                                                        \
  do {                                                                         \
    int err = (call);                                                          \
    if (err != MPI_SUCCESS) {                                                  \
      char err_string[MPI_MAX_ERROR_STRING];                                   \
      int err_len;                                                             \
      MPI_Error_string(err, err_string, &err_len);                             \
      fprintf(stderr, "MPI error at %s:%d: %s\n", __FILE__, __LINE__,          \
              err_string);                                                     \
      MPI_Abort(MPI_COMM_WORLD, err);                                          \
    }                                                                          \
  } while (0)

size_t Ningivenrank(size_t N, int rank, int size) {
  return N / size + ((N % size > rank) ? 1 : 0);
}

size_t chunkstart(size_t N, int rank, int size) {
  return rank * (N / size) + MIN(N % size, rank);
}

#endif // __UTIL_H_
