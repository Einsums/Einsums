all: mkl_test rocm_test hip_test

.PHONY:clean
clean:
	rm -rf *.o *.csv hip_test mkl_test rocm_test

mkl_test: mkl_test.c
	gcc -Og -g3 -ggdb -o mkl_test mkl_test.c -lm -lmkl_core -lmkl_rt -lpthread -L/opt/intel/oneapi/mkl/latest/lib/intel64 -I/opt/intel/oneapi/mkl/latest/include -Xlinker --rpath=/opt/intel/oneapi/mkl/latest/lib/intel64 

rocm_test: rocm_test.hip
	hipcc -o rocm_test rocm_test.hip -lm -lrocblas -fgpu-rdc --hip-link

hip_test: hip_test.hip
	hipcc -o hip_test hip_test.hip -lm -lhipblas -fgpu-rdc --hip-link
