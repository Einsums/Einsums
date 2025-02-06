all: profile_for_loop profile_fort_loop

profile_for_loop: profile_for_loop.c
profile_for_loop: CFLAGS+=-fopenmp -O3
profile_for_loop: LDFLAGS+=-lm

profile_fort_loop: profile_fort_loop.f03
	$(FC) -o profile_fort_loop profile_fort_loop.f03 -O3