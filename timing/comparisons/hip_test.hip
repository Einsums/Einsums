#include <cstdio>
#include <cstdlib>
#include <hipblas/hipblas.h>
#include <hip/hip_common.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <ctime>
#include <cstring>
#include <cmath>

using namespace std;

double random_double(void) {
  // Return between 0 inclusive and 1 exclusive.
  unsigned long out = 0x3ff0000000000000 |
    (((unsigned long) (0x000fffff & rand())) << 32) |
    rand();
  return (*(double *) &out) - 1;
}

// Time the algorithm when core memory is mapped to the GPU.
void profile_size_map(int n, FILE *outfile, int tests, hipblasHandle_t handle) {
  srand(clock());  // Seed random number generator.

  // Allocate array.
  double *arr1, *arr2, *arr3;
  arr1 = (double *) calloc(n * n, sizeof(double));
  arr2 = (double *) calloc(n * n, sizeof(double));
  arr3 = (double *) calloc(n * n, sizeof(double));

  // Map the memory.
  hipHostRegister(arr1, n * n * sizeof(double), hipHostRegisterDefault);
  hipHostRegister(arr2, n * n * sizeof(double), hipHostRegisterDefault);
  hipHostRegister(arr3, n * n * sizeof(double), hipHostRegisterDefault);

  // Fill with random junk.
  for(int i = 0; i < n * n; i++) {
    arr1[i] = random_double();
    arr2[i] = random_double();
  }

  double sum = 0, sumsq = 0;
  for(int test = 0; test < tests; test++) {
    
    
    // Start the timer.
    clock_t start = clock();
    
    // Matrix multiply.
    double alpha = 1, beta = 0;
    hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
		  n, n, n, &alpha, arr1, n, arr2, n, &beta, arr3, n);
    hipDeviceSynchronize();
    
    // Stop the timer.
    clock_t end = clock();
    sum += end - start;
    sumsq += (end - start) * (end - start);
  }

  // Unmap the memory.
  hipHostUnregister(arr1);
  hipHostUnregister(arr2);
  hipHostUnregister(arr3);
  
  free(arr1);
  free(arr2);
  free(arr3);

  // Print the results.
  if(outfile != NULL) {
    fprintf(outfile, "%d,%lf,%lf\n", n, sum / tests,
	    sqrt(sumsq / tests - sum / tests * sum / tests));
    fflush(outfile);
  }
}

// Time the algorithm when memory is copied directly into vRAM.
void profile_size_copy(int n, FILE *outfile, int tests, hipblasHandle_t handle) {
  srand(clock());  // Seed random number generator.

  // Allocate arrays.
  double *arr1, *arr2, *arr3;
  arr1 = (double *) calloc(n * n, sizeof(double));
  arr2 = (double *) calloc(n * n, sizeof(double));
  arr3 = (double *) calloc(n * n, sizeof(double));

  double *gpu_arr1, *gpu_arr2, *gpu_arr3;

  hipMalloc(&gpu_arr1, n * n * sizeof(double));
  hipMalloc(&gpu_arr2, n * n * sizeof(double));
  hipMalloc(&gpu_arr3, n * n * sizeof(double));

  // Fill with random junk.
  for(int i = 0; i < n * n; i++) {
    arr1[i] = random_double();
    arr2[i] = random_double();
  }
    
  double sum = 0, sumsq = 0;
  for(int test = 0; test < tests; test++) {
    
    // Start the timer.
    clock_t start = clock();

    // Copy the memory. This is a big part of what I want to time.
    hipMemcpy(gpu_arr1, arr1, n * n * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(gpu_arr2, arr2, n * n * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(gpu_arr3, arr3, n * n * sizeof(double), hipMemcpyHostToDevice);
    
    // Matrix multiply.
    double alpha = 1, beta = 0;
    hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
		  n, n, n, &alpha, gpu_arr1, n, gpu_arr2, n, &beta, gpu_arr3, n);
    // Wait until finished. This algorithm is usually asynchronous.
    hipDeviceSynchronize();

    // Copy the result back.
    hipMemcpy(arr3, gpu_arr3, n * n * sizeof(double), hipMemcpyDeviceToHost);
    
    // Stop the timer.
    clock_t end = clock();
    sum += end - start;
    sumsq += (end - start) * (end - start);
  }

  hipFree(gpu_arr1);
  hipFree(gpu_arr2);
  hipFree(gpu_arr3);
  
  free(arr1);
  free(arr2);
  free(arr3);

  // Print the results.
  if(outfile != NULL) {
    fprintf(outfile, "%d,%lf,%lf\n", n, sum / tests,
	    sqrt(sumsq / tests - sum / tests * sum / tests));
    fflush(outfile);
  }
}

// Help message to print.
char help_message[] =
  "mkl_check:\n"
  "Runs profiling for Intel's MKL.\n\n"
  "Usage:\n"
  "mkl_check [-o|--output FILE] [--start N] [--end N] [--step N] [-n N] [--map]\n"
  "Arguments:\n"
  "-o,--output FILE       Specify the output file. The output will be in CSV format, with the first value being the compute size and the second value being the runtime. Defaults to rocm_out.csv\n"
  "--start N              The starting size for the arrays.\n"
  "--end N                The ending size for the arrays.\n"
  "--step N               The size step to take between runs. Defaults to 1.\n"
  "-n N                   How many times to run and average over.\n"
  "--map                  Use the mapped version.\n"
  "--nomap                Use the copy version.";            

int argparse(int argc, char **argv, FILE **outfile, int *n_start,
	     int *n_end, int *n_step, int *tests, int *map) {

  int arg = 1;
  int pattern = 0; // Which argument pattern to use, whether single number, or otherwise.
  int argflags = 0; // Which arguments have been found.

  // Setup.
  *outfile = NULL;

  char *filename;
  
  while(arg < argc) {
    // Look for flags.
    if(argv[arg][0] != '-') {
      fprintf(stderr, "Argument error: expected a flag!\n");
      return -1;
    }

    // Look for which flag is found.
    if((strlen(argv[arg]) >= 2 && strncmp(argv[arg], "-o", 3) == 0) ||
       (strlen(argv[arg]) >= 8 && strncmp(argv[arg], "--output", 9) == 0)) {
      if(argflags & 0x1) {
	fprintf(stderr, "Argument error: duplicate output argument!\n");
	return -1;
      }
      if(arg >= argc) {
	fprintf(stderr, "Argument error: expected a string after argument!\n");
	return -1;
      }
      filename = argv[arg + 1];
      argflags |= 0x1;
    } else if(strlen(argv[arg]) >= 7 && strncmp(argv[arg], "--start", 8) == 0) {
      if(argflags & 0x2) {
	fprintf(stderr, "Argument error: duplicate start argument!\n");
	return -1;
      }
      if(arg >= argc) {
	fprintf(stderr, "Argument error: expected a value after argument!\n");
	return -1;
      }
      sscanf(argv[arg + 1], "%d", n_start);
      argflags |= 0x2;
    } else if(strlen(argv[arg]) >= 5 && strncmp(argv[arg], "--end", 6) == 0) {
      if(argflags & 0x4) {
	fprintf(stderr, "Argument error: duplicate end argument!\n");
	return -1;
      }
      if(arg >= argc) {
	fprintf(stderr, "Argument error: expected a value after argument!\n");
	return -1;
      }
      sscanf(argv[arg + 1], "%d", n_end);
      argflags |= 4;
    } else if(strlen(argv[arg]) >= 6 && strncmp(argv[arg], "--step", 7) == 0) {
      if(argflags & 0x8) {
	fprintf(stderr, "Argument error: duplicate step argument!\n");
	return -1;
      }
      if(arg >= argc) {
	fprintf(stderr, "Argument error: expected a value after argument!\n");
	return -1;
      }
      sscanf(argv[arg + 1], "%d", n_step);
      argflags |= 8;
    } else if(strlen(argv[arg]) >= 2 && strncmp(argv[arg], "-n", 3) == 0) {
      if(argflags & 0x10) {
	fprintf(stderr, "Argument error: duplicate tests argument!\n");
	return -1;
      }
      if(arg >= argc) {
	fprintf(stderr, "Argument error: expected a value after argument!\n");
	return -1;
      }
      sscanf(argv[arg + 1], "%d", tests);
      argflags |= 0x10;
    } else if((strlen(argv[arg]) >= 2 && strncmp(argv[arg], "-h", 3) == 0) ||
	      (strlen(argv[arg]) >= 6 && strncmp(argv[arg], "--help", 7) == 0)) {
      puts(help_message);
      return -2;
    } else if(strlen(argv[arg]) >= 5 && strncmp(argv[arg], "--map", 6) == 0) {
      if(argflags & 0x20) {
	fprintf(stderr, "Argument error: duplicate map argument!\n");
	return -1;
      }
      *map = 1;
      argflags |= 0x20;
      arg--; // Align properly.
    } else if(strlen(argv[arg]) >= 7 && strncmp(argv[arg], "--nomap", 8) == 0) {
      if(argflags & 0x20) {
	fprintf(stderr, "Argument error: duplicate map argument!\n");
	return -1;
      }
      *map = 0;
      argflags |= 0x20;
      arg--;
    } else {
      fprintf(stderr, "Argument error: unrecognized argument %s\n", argv[arg]);
      return -1;
    }
    arg += 2;
  }

  if(!(argflags & 0x8)) {
    *n_step = 1;
  }
  if(!(argflags & 0x10)) {
    *tests = 1;
  }
  if(!(argflags & 0x20)) {
    *map = 1;
  }
  if((argflags & 6) != 6) {
    fprintf(stderr, "Argument error: need a start and an end!\n");
    return -1;
  }
  
  if(!(argflags & 1)) {
    *outfile = fopen("hip_out.csv", "w+");
  } else {
    *outfile = fopen(filename, "w+");
  }
  
  return 0;
}
	  
    

int main(int argc, char **argv) {
  
  FILE *outfile;
  int start = 0, end = 0, step = 0, tests = 0, map = 0;

  int status = argparse(argc, argv, &outfile, &start, &end, &step, &tests, &map);

  if(status < 0) {
    return status + 2;
  }

  hipFree(nullptr); // Initialize HIP.

  
  hipblasHandle_t handle;
  hipblasCreate(&handle);

  if(map) {
    profile_size_map(50, nullptr, 1, handle);
  } else {
    profile_size_copy(50, nullptr, 1, handle);
  }

  for(int n = start; n <= end; n += step) {
    if(n == 0) {
      continue;
    }
    if(map) {
      profile_size_map(n, outfile, tests, handle);
    } else {
      profile_size_copy(n, outfile, tests, handle);
    }
  }

  hipblasDestroy(handle);

  fclose(outfile);

  return 0;
}
