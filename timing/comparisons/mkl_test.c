#include <stdio.h>
#include <stdlib.h>
#include <mkl_cblas.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

double random_double(void) {
  // Return between 0 inclusive and 1 exclusive.
  unsigned long out = 0x3ff0000000000000 |
    (((unsigned long) (0x000fffff & rand())) << 32) |
    rand();
  return (*(double *) &out) - 1;
}

void profile_size(int n, FILE *outfile, int tests) {
  srand(clock());  // Seed random number generator.

  // Allocate array.
  double *arr1, *arr2, *arr3;
  arr1 = calloc(n * n, sizeof(double));
  arr2 = calloc(n * n, sizeof(double));
  arr3 = calloc(n * n, sizeof(double));

  double sum = 0, sumsq = 0;
  for(int test = 0; test < tests; test++) {
    // Fill with random junk.
    for(int i = 0; i < n * n; i++) {
      arr1[i] = random_double();
      arr2[i] = random_double();
    }
    
    // Start the timer.
    clock_t start = clock();
    
    // Matrix multiply.
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, arr1, n, arr2, n, 0, arr3, n);
    
    // Stop the timer.
    clock_t end = clock();
    sum += end - start;
    sumsq += (end - start) * (end - start);
  }
  free(arr1);
  free(arr2);
  free(arr3);

  // Print the results.
  if(outfile != NULL) {
    fprintf(outfile, "%d,%lf,%lf\n", n, sum / tests,
	    sqrt(sumsq / tests - sum / tests * sum / tests));
  }
}

// Help message to print.
char help_message[] =
  "mkl_check:\n"
  "Runs profiling for Intel's MKL.\n\n"
  "Usage:\n"
  "mkl_check [-o|--output FILE] [--start N] [--end N] [--step N] [-n N]\n"
  "mkl_check [-o|--output FILE] [-n|--size N]\n\n"
  "Arguments:\n"
  "-o,--output FILE       Specify the output file. The output will be in CSV format, with the first value being the compute size, the second value being the runtime, and the third value being the standard deviation. Defaults to mkl_out.csv\n"
  "--start N              The starting size for the arrays.\n"
  "--end N                The ending size for the arrays.\n"
  "--step N               The size step to take between runs. Defaults to 1.\n"
  "-n N                   How many times to run and average over.";            

// Parse through the command line arguments.
int argparse(int argc, char **argv, FILE **outfile, int *n_start,
	     int *n_end, int *n_step, int *tests) {

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
  if((argflags & 6 != 6) && !(argflags & 0x10)) {
    fprintf(stderr, "Argument error: need a start and an end!\n");
    return -1;
  }
  
  if(!(argflags & 1)) {
    *outfile = fopen("mkl_out.csv", "w+");
  } else {
    *outfile = fopen(filename, "w+");
  }
  
  return 0;
}
	  
    

int main(int argc, char **argv) {
  
  FILE *outfile;
  int start = 0, end = 0, step = 0, tests = 0;

  int status = argparse(argc, argv, &outfile, &start, &end, &step, &tests);

  if(status < 0) {
    return status + 2; // Return success when help is printed.
  }

  // Make sure the MKL library is initialized.
  profile_size(50, NULL, 1);

  for(int n = start; n <= end; n += step) {
    if(n == 0) {
      continue;
    }
    profile_size(n, outfile, tests);
  }

  fclose(outfile);

  return 0;
}
