# Makefile for Mandelbrot fractal generator
# Author: jameswmccarty

CC = nvcc
NVCCFLAGS += -D_FORCE_INLINES -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)
CFLAGS = -ansi -O2 
LFLAGS = -lm -ltiff -lpthread

all: mandel

mandel: mandel_cuda.cu 
	$(CC) $(NVCCFLAGS) --compiler-options='$(CFLAGS)' mandel_cuda.cu -o mandel $(LFLAGS)

clean:
	\rm mandel
