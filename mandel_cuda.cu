/* Render the Mandelbrot set using Orbit Traps */
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <tiffio.h>
#include <assert.h>

/* CUDA_N is the resolution of the output image (size CUDA_N x CUDA_N) */
#define CUDA_N 16000

/* 8-bit red, green, and blue channels */
typedef struct {
	unsigned char r, g, b;
} pixel;

typedef struct {
	double a_r, a_g, a_b;
	double b_r, b_g, b_b;
	double c_r, c_g, c_b;
	double d_r, d_g, d_b;
} palette;

typedef struct {
	pixel *d_pixels;
	pixel *h_pixels;
	palette *d_palette;
	palette *h_palette;
	char *outfile;
	char *palfile;
	double esc_radius;
	int counter_max;
	double x, y, ref_x, ref_y;
	double a, b, c;
	double width;
	int linedist;
} fractal;

void
write_to_tiff (fractal *fract)
{
  int row, col, idx;
  TIFF *output;
  char *raster;
  pixel *img = (*fract).h_pixels;
  printf("Writing to file.\n");
  /* Open the output image */
  if ((output = TIFFOpen (fract->outfile, "w")) == NULL)
    {
      fprintf (stderr, "Could not open outgoing image.\n");
      exit (EXIT_FAILURE);
    }

  /* malloc space for the image lines */
  raster = (char*) malloc (CUDA_N * 3 * sizeof (char));
  if (raster == NULL)
    {
      printf ("malloc() failed in write_to_tiff.\n");
      exit (EXIT_FAILURE);
    }

  /* Write the tiff tags to the file */

  TIFFSetField (output, TIFFTAG_IMAGEWIDTH, CUDA_N);
  TIFFSetField (output, TIFFTAG_IMAGELENGTH, CUDA_N);
  TIFFSetField (output, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);
  TIFFSetField (output, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField (output, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
  TIFFSetField (output, TIFFTAG_BITSPERSAMPLE, 8);
  TIFFSetField (output, TIFFTAG_SAMPLESPERPIXEL, 3);

  printf("Wrote image file tags.\n");

   for (row = 0; row < CUDA_N; row++)
    {
      for (col = 0; col < CUDA_N; col++)
	{
	  idx = row*CUDA_N + (CUDA_N - col);
	  raster[col*3] =   img[idx].r;
	  raster[col*3+1] = img[idx].g;
	  raster[col*3+2] = img[idx].b;
	}
      if (TIFFWriteScanline (output, raster, row, CUDA_N * 3) != 1)
	{
	  fprintf (stderr, "Could not write image\n");
	  exit (EXIT_FAILURE);
	}
    }

  free (raster);
  /* close the file */
  TIFFClose (output);
}

/* color(t) = a + b * cos[2pi(c*t+d)] */
__device__ void
color_pxl(double t, palette *pal, double *r_out, double *g_out, double *b_out)
{
	*r_out = 255. * (pal->a_r + pal->b_r * cos(M_PI * 2. * (pal->c_r * t + pal->d_r)));
	*g_out = 255. * (pal->a_g + pal->b_g * cos(M_PI * 2. * (pal->c_g * t + pal->d_g)));
	*b_out = 255. * (pal->a_b + pal->b_b * cos(M_PI * 2. * (pal->c_b * t + pal->d_b)));
}


/* distance between (x1, y1) and (x2, y2) */
__device__ double
point_dist(double x1, double x2, double y1, double y2)
{
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}

/* distance between (x0, y0) and line (ax+by+c=0) */
__device__ double
line_dist(double x0, double y0, double a, double b, double c)
{
	double d, n;
	d = sqrt(a*a+b*b);
	n = abs(a*x0+b*y0+c);
	return n/d;
}

__global__ void 
render(pixel *pxls, 
float xmin, float xmax, float ymin, float ymax, 
double esc, int count_max, 
double xref, double yref,
double a, double b, double c, int linedist, 
palette *pal)
{
	int i, j, idx;
	float x1, y1, x2, y2, xtmp;
	int counter = 0;
	double dist = 1e9;

	double r_out, g_out, b_out;

	/* compute x (i) and y (j) index from Block and Thread */
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i >= CUDA_N || j >= CUDA_N) return; /* verify inbounds of image */ 	

	/* find x and y cartesian points for pixel */
	x1 = xmax - ( ((float) i / (float) CUDA_N) * (xmax - xmin) );
	y1 = ymax - ( ((float) j / (float) CUDA_N) * (ymax - ymin) );
	
	x2 = x1;
	y2 = y1;
	while( ( (x1*x1 + y1*y1) < esc ) && counter < count_max )
	{
		xtmp = x1 * x1 - y1 * y1 + x2;
		y1 = 2. * x1 * y1 + y2;
		x1 = xtmp;
		counter++;
		
		dist = min(dist, 
		linedist == 0 ?	point_dist(x1,xref,y1,yref) : line_dist(x1, y1, a, b, c));
	}
	idx = i + j*CUDA_N;
	color_pxl(dist, pal, &r_out, &g_out, &b_out);
	pxls[idx].r = (char) r_out;
	pxls[idx].g = (char) g_out;
	pxls[idx].b = (char) b_out;
}

/* initialize the color palette with user inputs
 * or a default state if no input is provided. */
void pal_init(palette *pal, char *infile)
{
	FILE *palette;

	if(infile == NULL) {
		/* a nice light blue default */
		pal->a_r = 0.39;
		pal->a_g = 0.55;
		pal->a_b = 0.5;

		pal->b_r = 0.55;
		pal->b_g = 0.26;
		pal->b_b = 0.68;

		pal->c_r = 0.5;
		pal->c_g = 1.5;
		pal->c_b = 0.0;

		pal->d_r = 0.26;
		pal->d_g = 0.11;
		pal->d_b = 0.24;
	} else {
	  if ((palette = fopen(infile, "r")) == NULL)
	  {
	   printf ("Error reading input file %s.\n", infile);
	   exit (EXIT_FAILURE);
	  }
		/* WARNING -- poor checks for malformed input here. */
		assert(fscanf (palette, "%lf %lf %lf\n", &(pal->a_r), &(pal->a_g), &(pal->a_b)) != EOF);
		assert(fscanf (palette, "%lf %lf %lf\n", &(pal->b_r), &(pal->b_g), &(pal->b_b)) != EOF);
		assert(fscanf (palette, "%lf %lf %lf\n", &(pal->c_r), &(pal->c_g), &(pal->c_b)) != EOF);
		assert(fscanf (palette, "%lf %lf %lf\n", &(pal->d_r), &(pal->d_g), &(pal->d_b)) != EOF);
		(void) fclose (palette);
	} /* end else */
} /* end pal_init */

void
print_usage ()
{
  /* print program use */

  printf ("Render the Mandelbrot set using Orbit Traps.\n\n");
  printf ("mandel usage:\n");
  printf ("mandel [-options ...]\n\n");
  printf ("options include:\n");

  printf ("\t-h\t\t\tprint this screen\n");
  printf ("\t-f NAME\t\t\toutput file to use (i.e. /tmp/mandel.tif)\n");
  printf ("\t-p NAME\t\t\tfile to use for color palette\n");
  printf ("\t-x #.###...#\t\tcenter X coordinate of image\n");
  printf ("\t-y #.###...#\t\tcenter Y coordinate of image\n");
  printf ("\t-rx #.###...#\t\tX coordinate for distance reference\n");
  printf ("\t-ry #.###...#\t\tY coordinate for distance reference\n");
  printf ("\t-L\t\t\tuse the line equation for orbit trap instead of a point\n");
  printf ("\t-a #.###...#\t\tA parameter of reference line in form Ax + By + C = 0\n");
  printf ("\t-b #.###...#\t\tB parameter of reference line in form Ax + By + C = 0\n");
  printf ("\t-c #.###...#\t\tC parameter of reference line in form Ax + By + C = 0\n");
  printf ("\t-w ##.#\t\t\twidth of image (x and y +/- width)\n");
  printf ("\t-m ####\t\t\tmax iterations to compute\n");
  printf ("\t-e ##.#\t\t\tescape radius\n");


}

void
parse_args (int argc, char **argv, fractal * mandel)
{
  int i = 1;
  while (i < argc)
    {
		if (!strcmp (argv[i], "-h"))
		{
	  	print_usage ();
	  	exit (EXIT_SUCCESS);
		}
		else if (!strcmp (argv[i], "-f"))
		{
	  	mandel->outfile = argv[i + 1];
	 	i += 2;
		}
		else if (!strcmp (argv[i], "-p"))
		{
	  	mandel->palfile = argv[i + 1];
	  	i += 2;
		}
		else if (!strcmp (argv[i], "-x"))
		{
	  	mandel->x = (double) atof(argv[i + 1]);
	 	i += 2;
		}
		else if (!strcmp (argv[i], "-y"))
		{
	  	mandel->y = (double) atof(argv[i + 1]);
	 	i += 2;
		}
		else if (!strcmp (argv[i], "-rx"))
		{
	  	mandel->ref_x = (double) atof(argv[i + 1]);
	 	i += 2;
		}
		else if (!strcmp (argv[i], "-ry"))
		{
	  	mandel->ref_y = (double) atof(argv[i + 1]);
	 	i += 2;
		}
		else if (!strcmp (argv[i], "-a"))
		{
	  	mandel->a = (double) atof(argv[i + 1]);
	 	i += 2;
		}
		else if (!strcmp (argv[i], "-b"))
		{
	  	mandel->b = (double) atof(argv[i + 1]);
	 	i += 2;
		}
		else if (!strcmp (argv[i], "-c"))
		{
	  	mandel->c = (double) atof(argv[i + 1]);
	 	i += 2;
		}
		else if (!strcmp (argv[i], "-w"))
		{
	  	mandel->width = (double) atof(argv[i + 1]);
	 	i += 2;
		}
		else if (!strcmp (argv[i], "-m"))
		{
	  	mandel->counter_max = atoi(argv[i + 1]);
	 	i += 2;
		}
		else if (!strcmp (argv[i], "-L"))
		{
	  	mandel->linedist = 1;
	 	i += 1;
		}
		else if (!strcmp (argv[i], "-e"))
		{
	  	mandel->esc_radius = atof(argv[i + 1]);
	 	i += 2;
		}
		else
		{
	 	print_usage ();
	  	exit (EXIT_FAILURE);
		}	
	}
}	

int main(int argc, char **argv)
{
	fractal mandel;
	mandel.d_pixels = NULL;
	mandel.h_pixels = NULL;
	mandel.d_palette = NULL;
	mandel.h_palette = NULL;
	mandel.outfile   = (char *) "/tmp/mandel.tif"; /* default */
	mandel.palfile   = NULL;
	mandel.esc_radius= 2e5;
	mandel.counter_max = 350;
	mandel.x = 0.0;
	mandel.y = 0.0;
	mandel.ref_x = 0.0;
	mandel.ref_y = 0.0;
	mandel.width = 2.5;
	mandel.a = 1.0;
	mandel.b = -1.0;
	mandel.c = 0.0;
	mandel.linedist = 0;
	cudaError_t err;

	/* process input arguments */
	parse_args(argc, argv, &mandel);
	/* sanity check */
	if(mandel.linedist == 1 && (mandel.a == 0.0 && mandel.b == 0.0)) {
		printf("Illegal configuration.  A and B cannot both be set to zero.\n");
		exit(EXIT_FAILURE);
	}

	/* HOST buffer for color palette */
	mandel.h_palette = (palette*) malloc(sizeof(palette));
	if(mandel.h_palette == NULL) {
		printf("malloc() failed in main.\n");
		exit(EXIT_FAILURE);
	}

	/* Initialize the palette */
	pal_init(mandel.h_palette, mandel.palfile);	

	/* assign a CUDA memory buffer for the fractal rendering */
	err = cudaMalloc(&(mandel.d_pixels), CUDA_N*CUDA_N*sizeof(pixel));
	if(err != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(err));		
		exit(EXIT_FAILURE);
	}

	/* assign a CUDA memory buffer for the color palette */
	err = cudaMalloc(&(mandel.d_palette), sizeof(palette));
	if(err != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(err));		
		exit(EXIT_FAILURE);
	}	
	
	printf("Allocated CUDA device memory.\n");

	/* setup block sizes to allow for rendering in min number of blocks */
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(CUDA_N / threadsPerBlock.x, CUDA_N / threadsPerBlock.y);

	/* copy palette to device */
	/* copy the buffer from HOST to DEVICE */
	err = cudaMemcpy(mandel.d_palette, mandel.h_palette, sizeof(palette), cudaMemcpyHostToDevice);
	        if(err != cudaSuccess) {
                printf("%s\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
        }

	/* dispatch the CUDA process */
	render<<<numBlocks, threadsPerBlock>>>(mandel.d_pixels,
	mandel.x-mandel.width, mandel.x+mandel.width, mandel.y-mandel.width, mandel.y+mandel.width,
	mandel.esc_radius, mandel.counter_max,
	mandel.ref_x, mandel.ref_y,
	mandel.a, mandel.b, mandel.c, mandel.linedist,
	mandel.d_palette);
	printf("Completed render.\n");

	/* HOST buffer for completed render */
	mandel.h_pixels = (pixel*) malloc(CUDA_N*CUDA_N*sizeof(pixel));
	if(mandel.h_pixels == NULL) {
		printf("malloc() failed in main.\n");
		exit(EXIT_FAILURE);
	}

	/* copy the buffer from DEVICE to HOST */
	err = cudaMemcpy(mandel.h_pixels, mandel.d_pixels, CUDA_N*CUDA_N*sizeof(pixel), cudaMemcpyDeviceToHost);
	        if(err != cudaSuccess) {
                printf("%s\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
        }
	printf("Mem copy complete.\n");
	/* then free the DEVICE memory */
	cudaFree(mandel.d_pixels);
	cudaFree(mandel.d_palette);
	printf("Freed CUDA memory.\n");
	/* then write the buffer to file */
	write_to_tiff(&mandel);
	/* and free the buffer */
	printf("Wrote to file.\n");
	free (mandel.h_pixels);
	free (mandel.h_palette);
	return 0;
}
	
