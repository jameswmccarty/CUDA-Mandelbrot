# CUDA-Mandelbrot
Nvidia CUDA based utility to render the Mandelbrot set using Orbit Traps.

![Mandelbrot Set Screenshot](screenshot.png "Example screenshot")

# Background
This is a CUDA based project for rendering the Mandelbrot set using the Orbit Trap method (https://en.wikipedia.org/wiki/Orbit_trap).

# How To Use This Program
Download all the project files.  They are:

o mandel_cuda.cu (The main CUDA project file.)

o Makefile (Contains build configuration)

Compile the project with: `make`

Then run: `./mandel` from the current directory with any of the available options:

o -h					  print help screen

o -f NAME                 output file to use (default /tmp/mandel.tif)

o -p NAME                 file to use for color palette (default none)

o -x #.###...#            center X coordinate of image (default 0.0)

o -y #.###...#            center Y coordinate of image (default 0.0)

o -rx #.###...#           X coordinate for distance reference (default 0.0)

o -ry #.###...#           Y coordinate for distance reference (default 0.0)

o -w ##.#                 width of image (x and y +/- width) (default 2.5)

o -m ####                 max iterations to compute (default 350)

o -e ##.#                 escape radius (default 2e5)

See my Java-ColorToy project for more information on setting up color options: https://github.com/jameswmccarty/Java-ColorToy 

Example format for color palette:

0.46 0.5 0.15  
0.34 0.47 0.43  
1.5 0.0 0.5  
0.1 0.64 0.48  


