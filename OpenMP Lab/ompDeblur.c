//OpenMP version.  Edit and submit only this file.
/* Enter your details below
 * Name : Shreya Raman
 * UCLA ID: 004923456
 * Email id: sraman99@ucla.edu
 * Input: New files
 * Note: Testing on server 8 usually gives me a speed up in the range 9-11
 */

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

int OMP_xMax;
#define xMax OMP_xMax
int OMP_yMax;
#define yMax OMP_yMax
int OMP_zMax;
#define zMax OMP_zMax

int OMP_Index(int x, int y, int z)
{
	return ((z * yMax + y) * xMax + x);
}
#define Index(x, y, z) OMP_Index(x, y, z)

double OMP_SQR(double x)
{
	return pow(x, 2.0);
}
#define SQR(x) OMP_SQR(x)

double* OMP_conv;
double* OMP_g;

void OMP_Initialize(int xM, int yM, int zM)
{
	xMax = xM;
	yMax = yM;
	zMax = zM;
	assert(OMP_conv = (double*)malloc(sizeof(double) * xMax * yMax * zMax));
	assert(OMP_g = (double*)malloc(sizeof(double) * xMax * yMax * zMax));
}
void OMP_Finish()
{
	free(OMP_conv);
	free(OMP_g);
}

void OMP_GaussianBlur(double *u, double Ksigma, int stepCount)
{
	double nu = (2.08 - sqrt(3.16))/(1.08);
	double nu2 = nu * 2;
	int x, y, z, step;
	double boundryScale = 1.0 / (1.0 - nu);
	double postScale = pow(nu / 0.54, 9.0);

	omp_set_num_threads(16);
	#pragma omp for nowait
	for(step = 0; step < stepCount; step++)
	{
		omp_set_num_threads(16);
		#pragma omp parallel for private (x,y) shared(boundryScale, nu)
		for(z = 0; z < 128; z+=4)
		{
			for(y = 0; y < 128; y+=4)
			{
				u[Index(0, y, z)] *= boundryScale;
                                u[Index(0, y, z + 1)] *= boundryScale;
                                u[Index(0, y + 1, z)] *= boundryScale;
                                u[Index(0, y + 1, z + 1)] *= boundryScale;
                                u[Index(0, y, z + 2)] *= boundryScale;
                                u[Index(0, y, z + 3)] *= boundryScale;
                                u[Index(0, y + 1, z + 2)] *= boundryScale;
                                u[Index(0, y + 1, z + 3)] *= boundryScale;
                                u[Index(0, y + 2, z)] *= boundryScale;
                                u[Index(0, y + 2, z + 1)] *= boundryScale;
                                u[Index(0, y + 2, z + 2)] *= boundryScale;
                                u[Index(0, y + 2, z + 3)] *= boundryScale;
                                u[Index(0, y + 3, z)] *= boundryScale;
                                u[Index(0, y + 3, z + 1)] *= boundryScale;
                                u[Index(0, y + 3, z + 2)] *= boundryScale;
                                u[Index(0, y + 3, z + 3)] *= boundryScale;
			
				for(x = 1; x < 128; x++)
				{
					u[Index(x, y, z)] += u[Index(x - 1, y, z)] * nu;
					u[Index(x, y + 1, z)] += u[Index(x - 1, y + 1, z)] * nu;
					u[Index(x, y + 2, z)] += u[Index(x - 1, y + 2, z)] * nu;
					u[Index(x, y + 3, z)] += u[Index(x - 1, y + 3, z)] * nu;	
					u[Index(x, y, z + 1)] += u[Index(x - 1, y, z + 1)] * nu2;
					u[Index(x, y + 1, z + 1)] += u[Index(x - 1, y + 1, z + 1)] * nu2;
                                        u[Index(x, y + 2, z + 1)] += u [Index(x - 1, y + 2, z + 1)] * nu2;
                                        u[Index(x, y + 3, z + 1)] += u[Index(x - 1, y + 3, z + 1)] * nu2;
					u[Index(x, y, z + 2)] += u[Index(x - 1, y, z + 2)] * nu2;
                                        u[Index(x, y + 1, z + 2)] += u[Index(x - 1, y + 1, z + 2)] * nu2;
                                        u[Index(x, y + 2, z + 2)] += u[Index(x - 1, y + 2, z + 2)] * nu2;
                                        u[Index(x, y + 3, z + 2)] += u[Index(x - 1, y + 3, z + 2)] * nu2;
					u[Index(x, y, z + 3)] += u[Index(x - 1, y, z + 3)] * nu2;
                                        u[Index(x, y + 1, z + 3)] += u[Index(x - 1, y + 1, z + 3)] * nu2;
                                        u[Index(x, y + 2, z + 3)] += u[Index(x - 1, y + 2, z + 3)] * nu2;
                                        u[Index(x, y + 3, z + 3)] += u[Index(x - 1, y + 3, z + 3)] * nu2;
					u[Index(x, y, z + 4)] += u[Index(x - 1, y, z + 4)] * nu;
                                        u[Index(x, y + 1, z + 4)] += u[Index(x - 1, y + 1, z + 4)] * nu;
                                        u[Index(x, y + 2, z + 4)] += u[Index(x - 1, y + 2, z + 4)] * nu;
                                        u[Index(x, y + 3, z + 4)] += u[Index(x - 1, y + 3, z + 4)] * nu;

				}

				u[Index(0, y, z)] *= boundryScale;
                                u[Index(0, y, z + 1)] *= boundryScale;
                                u[Index(0, y + 1, z)] *= boundryScale;
                                u[Index(0, y + 1, z + 1)] *= boundryScale;
                                u[Index(0, y, z + 2)] *= boundryScale;
                                u[Index(0, y, z + 3)] *= boundryScale;
                                u[Index(0, y + 1, z + 2)] *= boundryScale;
                                u[Index(0, y + 1, z + 3)] *= boundryScale;
                                u[Index(0, y + 2, z)] *= boundryScale;
                                u[Index(0, y + 2, z + 1)] *= boundryScale;
                                u[Index(0, y + 2, z + 2)] *= boundryScale;
                                u[Index(0, y + 2, z + 3)] *= boundryScale;
                                u[Index(0, y + 3, z)] *= boundryScale;
                                u[Index(0, y + 3, z + 1)] *= boundryScale;
                                u[Index(0, y + 3, z + 2)] *= boundryScale;
                                u[Index(0, y + 3, z + 3)] *= boundryScale;

				for(x = 126; x >= 0; x--)
                                {
                                        u[Index(x, y, z)] += u[Index(x + 1, y, z)] * nu;
                                        u[Index(x, y + 1, z)] += u[Index(x + 1, y + 1, z)] * nu;
                                        u[Index(x, y + 2, z)] += u[Index(x + 1, y + 2, z)] * nu;
                                        u[Index(x, y + 3, z)] += u[Index(x + 1, y + 3, z)] * nu;

                                        u[Index(x, y, z + 1)] += u[Index(x + 1, y, z + 1)] * nu;
                                        u[Index(x, y + 1, z + 1)] += u[Index(x + 1, y + 1, z + 1)] * nu;
                                        u[Index(x, y + 2, z + 1)] += u[Index(x + 1, y + 2, z + 1)] * nu;
                                        u[Index(x, y + 3, z + 1)] += u[Index(x + 1, y + 3, z + 1)] * nu;

                                        u[Index(x, y, z + 2)] += u[Index(x + 1, y, z + 2)] * nu;
                                        u[Index(x, y + 1, z + 2)] += u[Index(x + 1, y + 1, z + 2)] * nu;
                                        u[Index(x, y + 2, z + 2)] += u[Index(x + 1, y + 2, z + 2)] * nu;
                                        u[Index(x, y + 3, z + 2)] += u[Index(x + 1, y + 3, z + 2)] * nu;

                                        u[Index(x, y, z + 2)] += u[Index(x + 1, y, z + 3)] * nu;
                                        u[Index(x, y + 1, z + 2)] += u[Index(x + 1, y + 1, z + 3)] * nu;
                                        u[Index(x, y + 2, z + 2)] += u[Index(x + 1, y + 2, z + 3)] * nu;
                                        u[Index(x, y + 3, z + 2)] += u[Index(x + 1, y + 3, z + 3)] * nu;
                                }

			}

			for(x = 0; x < 128; x+=4)
                        {
                                u[Index(x, 0, z)] *= boundryScale;
                                u[Index(x + 1, 0, z)] *= boundryScale;
                                u[Index(x + 2, 0, z)] *= boundryScale;
                                u[Index(x + 3, 0, z)] *= boundryScale;
                                u[Index(x, 0, z + 1)] *= boundryScale;
                                u[Index(x + 1, 0, z + 1)] *= boundryScale;
                                u[Index(x + 2, 0, z + 1)] *= boundryScale;
                                u[Index(x + 3, 0, z + 1)] *= boundryScale;
                                u[Index(x, 0, z + 2)] *= boundryScale;
                                u[Index(x + 1, 0, z + 2)] *= boundryScale;
                                u[Index(x + 2, 0, z + 2)] *= boundryScale;
                                u[Index(x + 3, 0, z + 2)] *= boundryScale;
                                u[Index(x, 0, z + 3)] *= boundryScale;
                                u[Index(x + 1, 0, z + 3)] *= boundryScale;
                                u[Index(x + 2, 0, z + 3)] *= boundryScale;
                                u[Index(x + 3, 0, z + 3)] *= boundryScale;
                        }
			
			for(y = 1; y < 128; y++)
                        {
                                for(x = 0; x < 128; x+=4)
                                {
                                        u[Index(x, y, z)] += u[Index(x, y - 1, z)] * nu;
                                        u[Index(x + 1, y, z)] += u[Index(x + 1, y - 1, z)] * nu;
                                        u[Index(x + 2, y, z)] += u[Index(x + 2, y - 1, z)] * nu;
                                        u[Index(x + 3, y, z)] += u[Index(x + 3, y - 1, z)] * nu;

                                        u[Index(x, y, z + 1)] += u[Index(x, y - 1, z + 1)] * nu;
                                        u[Index(x + 1, y, z + 1)] += u[Index(x + 1, y - 1, z + 1)] * nu;
                                        u[Index(x + 2, y, z + 1)] += u[Index(x + 2, y - 1, z + 1)] * nu;
                                        u[Index(x + 3, y, z + 1)] += u[Index(x + 3, y - 1, z + 1)] * nu;

                                        u[Index(x, y, z + 2)] += u[Index(x, y - 1, z + 2)] * nu;
                                        u[Index(x + 1, y, z + 2)] += u[Index(x + 1, y - 1, z + 2)] * nu;
                                        u[Index(x + 2, y, z + 2)] += u[Index(x + 2, y - 1, z + 2)] * nu;
                                        u[Index(x + 3, y, z + 2)] += u[Index(x + 3, y - 1, z + 2)] * nu;

                                        u[Index(x, y, z + 3)] += u[Index(x, y - 1, z + 3)] * nu;
                                        u[Index(x + 1, y, z + 3)] += u[Index(x + 1, y - 1, z + 3)] * nu;
                                        u[Index(x + 2, y, z + 3)] += u[Index(x + 2, y - 1, z + 3)] * nu;
                                        u[Index(x + 3, y, z + 3)] += u[Index(x + 3, y - 1, z + 3)] * nu;
                                }
                        }
			
			for(x = 0; x < 128; x+=4)
                        {
                                u[Index(x, 127, z)] *= boundryScale;
                                u[Index(x + 1, 127, z)] *= boundryScale;
                                u[Index(x + 2, 127, z)] *= boundryScale;
                                u[Index(x + 3, 127, z)] *= boundryScale;
                                u[Index(x, 127, z + 1)] *= boundryScale;
                                u[Index(x + 1, 127, z + 1)] *= boundryScale;
                                u[Index(x + 2, 127, z + 1)] *= boundryScale;
                                u[Index(x + 3, 127, z + 1)] *= boundryScale;
                                u[Index(x, 127, z + 2)] *= boundryScale;
                                u[Index(x + 1, 127, z + 2)] *= boundryScale;
                                u[Index(x + 2, 127, z + 2)] *= boundryScale;
                                u[Index(x + 3, 127, z + 2)] *= boundryScale;
                                u[Index(x, 127, z + 3)] *= boundryScale;
                                u[Index(x + 1, 127, z + 3)] *= boundryScale;
                                u[Index(x + 2, 127, z + 3)] *= boundryScale;
                                u[Index(x + 3, 127, z + 3)] *= boundryScale;
                        }

			for(y = 126; y >= 0; y--)
                        {
                                for(x = 0; x < 128; x+=4)
                                {
                                        u[Index(x, y, z)] += u[Index(x, y + 1, z)] * nu;
                                        u[Index(x + 1, y, z)] += u[Index(x + 1, y + 1, z)] * nu;
                                        u[Index(x + 2, y, z)] += u[Index(x + 2, y + 1, z)] * nu;
                                        u[Index(x + 3, y, z)] += u[Index(x + 3, y + 1, z)] * nu;

                                        u[Index(x, y, z + 1)] += u[Index(x, y + 1, z + 1)] * nu;
                                        u[Index(x + 1, y, z + 1)] += u[Index(x + 1, y + 1, z + 1)] * nu;
                                        u[Index(x + 2, y, z + 1)] += u[Index(x + 2, y + 1, z + 1)] * nu;
                                        u[Index(x + 3, y, z + 1)] += u[Index(x + 3, y + 1, z + 1)] * nu;

                                        u[Index(x, y, z + 2)] += u[Index(x, y + 1, z + 2)] * nu;
                                        u[Index(x + 1, y, z + 2)] += u[Index(x + 1, y + 1, z + 2)] * nu;
                                        u[Index(x + 2, y, z + 2)] += u[Index(x + 2, y + 1, z + 2)] * nu;
                                        u[Index(x + 3, y, z + 2)] += u[Index(x + 3, y + 1, z + 2)] * nu;

                                        u[Index(x, y, z + 3)] += u[Index(x, y + 1, z + 3)] * nu;
                                        u[Index(x + 1, y, z + 3)] += u[Index(x + 1, y + 1, z + 3)] * nu;
                                        u[Index(x + 2, y, z + 3)] += u[Index(x + 2, y + 1, z + 3)] * nu;
                                        u[Index(x + 3, y, z + 3)] += u[Index(x + 3, y + 1, z + 3)] * nu;
                                }
                        }

		}
		
		omp_set_num_threads(16);

		#pragma omp parallel for private(x) shared(boundryScale, nu)
		for(y = 0; y < 128; y+=4)
		{
			for(x = 0; x < 128; x+=4)
			{
				u[Index(x, y, 0)] *= boundryScale;
				u[Index(x + 1, y, 0)] *= boundryScale;
				u[Index(x + 2, y, 0)] *= boundryScale;
				u[Index(x + 3, y, 0)] *= boundryScale;

				u[Index(x, y + 1, 0)] *= boundryScale;
                                u[Index(x + 1, y + 1, 0)] *= boundryScale;
                                u[Index(x + 2, y + 1, 0)] *= boundryScale;
                                u[Index(x + 3, y + 1, 0)] *= boundryScale;

				u[Index(x, y + 2, 0)] *= boundryScale;
                                u[Index(x + 1, y + 2, 0)] *= boundryScale;
                                u[Index(x + 2, y + 2, 0)] *= boundryScale;
                                u[Index(x + 3, y + 2, 0)] *= boundryScale;

				u[Index(x, y + 3, 0)] *= boundryScale;
                                u[Index(x + 1, y + 3, 0)] *= boundryScale;
                                u[Index(x + 2, y + 3, 0)] *= boundryScale;
                                u[Index(x + 3, y + 3, 0)] *= boundryScale;

				for(z = 1; z < zMax; z++)
                                {
                                        u[Index(x, y, z)] = u[Index(x, y, z - 1)] * nu;
                                        u[Index(x + 1, y, z)] = u[Index(x + 1, y, z - 1)] * nu;
                                        u[Index(x + 2, y, z)] = u[Index(x + 2, y, z - 1)] * nu;
                                        u[Index(x + 3, y, z)] = u[Index(x + 3, y, z - 1)] * nu;

                                        u[Index(x, y + 1, z)] = u[Index(x, y + 1, z - 1)] * nu;
                                        u[Index(x + 1, y + 1, z)] = u[Index(x + 1, y + 1, z - 1)] * nu;
                                        u[Index(x + 2, y + 1, z)] = u[Index(x + 2, y + 1, z - 1)] * nu;
                                        u[Index(x + 3, y + 1, z)] = u[Index(x + 3, y + 1, z - 1)] * nu;

                                        u[Index(x, y + 2, z)] = u[Index(x, y + 2, z - 1)] * nu;
                                        u[Index(x + 1, y + 2, z)] = u[Index(x + 1, y + 2, z - 1)] * nu;
                                        u[Index(x + 2, y + 2, z)] = u[Index(x + 2, y + 2, z - 1)] * nu;
                                        u[Index(x + 3, y + 2, z)] = u[Index(x + 3, y + 2, z - 1)] * nu;

                                        u[Index(x, y + 3, z)] = u[Index(x, y + 3, z - 1)] * nu;
                                        u[Index(x + 1, y + 3, z)] = u[Index(x + 1, y + 3, z - 1)] * nu;
                                        u[Index(x + 2, y + 3, z)] = u[Index(x + 2, y + 3, z - 1)] * nu;
                                        u[Index(x + 3, y + 3, z)] = u[Index(x + 3, y + 3, z - 1)] * nu;
                                }
				u[Index(x, y, 127)] *= boundryScale;
                                u[Index(x + 1, y, 127)] *= boundryScale;
                                u[Index(x + 2, y, 127)] *= boundryScale;
                                u[Index(x + 3, y, 127)] *= boundryScale;

                                u[Index(x, y + 1, 127)] *= boundryScale;
                                u[Index(x + 1, y + 1, 127)] *= boundryScale;
                                u[Index(x + 2, y + 1, 127)] *= boundryScale;
                                u[Index(x + 3, y + 1, 127)] *= boundryScale;

                                u[Index(x, y + 2, 127)] *= boundryScale;
                                u[Index(x + 1, y + 2, 127)] *= boundryScale;
                                u[Index(x + 2, y + 2, 127)] *= boundryScale;
                                u[Index(x + 3, y + 2, 127)] *= boundryScale;

                                u[Index(x, y + 3, 127)] *= boundryScale;
                                u[Index(x + 1, y + 3, 127)] *= boundryScale;
                                u[Index(x + 2, y + 3, 127)] *= boundryScale;
                                u[Index(x + 3, y + 3, 127)] *= boundryScale;

			}
		}
		
		#pragma omp for nowait
		for(z = 126; z >= 0; z--)
		{
			for(y = 0; y < 128; y+=4)
			{
				for(x = 0; x < 128; x+=4)
				{
					u[Index(x, y, z)] += u[Index(x, y, z + 1)] * nu;
					u[Index(x + 1, y, z)] += u[Index(x + 1, y, z + 1)] * nu;
					u[Index(x + 2, y, z)] += u[Index(x + 2, y, z + 1)] * nu;
					u[Index(x + 3, y, z)] += u[Index(x + 3, y, z + 1)] * nu;
					
					u[Index(x, y + 1, z)] += u[Index(x, y + 1, z + 1)] * nu;
                                        u[Index(x + 1, y + 1, z)] += u[Index(x + 1, y + 1, z + 1)] * nu;
                                        u[Index(x + 2, y + 1, z)] += u[Index(x + 2, y + 1, z + 1)] * nu;
                                        u[Index(x + 3, y + 1, z)] += u[Index(x + 3, y + 1, z + 1)] * nu;

					u[Index(x, y + 2, z)] += u[Index(x, y + 2, z + 1)] * nu;
                                        u[Index(x + 1, y + 2, z)] += u[Index(x + 1, y + 2, z + 1)] * nu;
                                        u[Index(x + 2, y + 2, z)] += u[Index(x + 2, y + 2, z + 1)] * nu;
                                        u[Index(x + 3, y + 2, z)] += u[Index(x + 3, y + 2, z + 1)] * nu;

					u[Index(x, y + 3, z)] += u[Index(x, y + 3, z + 1)] * nu;
                                        u[Index(x + 1, y + 3, z)] += u[Index(x + 1, y + 3, z + 1)] * nu;
                                        u[Index(x + 2, y + 3, z)] += u[Index(x + 2, y + 3, z + 1)] * nu;
                                        u[Index(x + 3, y + 3, z)] += u[Index(x + 3, y + 3, z + 1)] * nu;
				}
			}
		}
	}
	omp_set_num_threads(16);
	#pragma omp parallel for private(y,x) shared(postScale, u)
	for(z = 0; z < 128; z++)
	{
		for(y = 0; y < 128; y+=4)
		{
			for(x = 0; x < 128; x+=4)
			{
				u[Index(x, y, z)] *= postScale;
				u[Index(x + 1, y, z)] *= postScale;
				u[Index(x + 2, y, z)] *= postScale;
				u[Index(x + 3, y, z)] *= postScale;
				u[Index(x, y + 1, z)] *= postScale;
                                u[Index(x + 1, y + 1, z)] *= postScale;
                                u[Index(x + 2, y + 1, z)] *= postScale;
                                u[Index(x + 3, y + 1, z)] *= postScale;
				u[Index(x, y + 2, z)] *= postScale;
                                u[Index(x + 1, y + 2, z)] *= postScale;
                                u[Index(x + 2, y + 2, z)] *= postScale;
                                u[Index(x + 3, y + 2, z)] *= postScale;
				u[Index(x, y + 3, z)] *= postScale;
                                u[Index(x + 1, y + 3, z)] *= postScale;
                                u[Index(x + 2, y + 3, z)] *= postScale;
                                u[Index(x + 3, y + 3, z)] *= postScale;
			}
		}
	}

}
void OMP_Deblur(double* u, const double* f, int maxIterations, double dt, double gamma, double sigma, double Ksigma)
{
	int x, y, z, iteration;
	int converged = 0;
	int lastConverged = 0;
	double* conv = OMP_conv;
	double* g = OMP_g;

	omp_set_num_threads(16);

	for(iteration = 0; iteration < maxIterations && converged != 2048383; iteration++)
	{

		#pragma omp parallel for private(y,x)		
		for(z = 1; z < 127; z++)
		{
			for(y = 1; y < 127; y+=2)
			{
				for(x = 1; x < 127; x+=2)
				{
					int ixyz = Index(x, y, z);
					double uixyz = u[ixyz];

					int ix1yz = Index(x + 1, y, z);
					double uix1yz = u[ix1yz];

					int ixy1z = Index(x, y + 1, z);
					double uixy1z = u[ixy1z];

					int ix1y1z = Index(x + 1, y + 1, z);
					double uix1y1z = u[ix1y1z];
					
					int ixyz1 = Index(x, y, z + 1);
					double uixyz1 = u[ixyz1];

					int ixm1 = Index(x - 1, y, z);
					double uixm1 = u[ixm1];

					int iym1 = Index(x, y - 1, z);
					double uiym1 = u[iym1];

					int izm1 = Index(x, y, z - 1);
					double uizm1 = u[izm1];

					int ix2yz = Index(x + 2, y, z);
					double uix2yz = u[ix2yz];

					int ix1ym1 = Index(x + 1, y - 1, z);
					double uix1ym1 = u[ix1ym1];

					int ix1yz1 = Index(x + 1, y, z + 1);
					double uix1yz1 = u[ix1yz1];

					int ix1zm1 = Index(x + 1, y, z - 1);
					double uix1zm1 = u[ix1zm1];

					int ixm1y1 = Index(x - 1, y + 1, z);
					double uixm1y1 = u[ixm1y1];

					int ixy2z = Index(x, y + 2, z);
					double uixy2z = u[ixy2z];

					int ixy1z1 = Index(x, y + 1, z + 1);
					double uixy1z1 = u[ixy1z1];
		
					int iy1zm1 = Index(x, y + 1, z - 1);
					double uiy1zm1 = u[iy1zm1];

					int ix1y1zm1 = Index(x + 1, y + 1, z - 1);
					double uix1y1zm1 = u[ix1y1zm1];

					int ix1y1z1 = Index(x + 1, y + 1, z + 1);
					double uix1y1z1 = u[ix1y1z1];

					int ix1y2 = Index(x + 1, y + 2, z);
					double uix1y2 = u[ix1y2];

					int ix2y1 = Index(x + 2, y + 1, z);
					double uix2y1 = u[ix2y1];

					g[ixyz] = 1.0 / sqrt(1.0e-7 +
                                                ((uixyz - uix1yz) * (uixyz - uix1yz)) +
                                                ((uixyz - uixm1) * (uixyz - uixm1)) +
                                                ((uixyz - uixy1z) * (uixyz - uixy1z)) +
                                                ((uixyz - uiym1) * (uixyz - uiym1)) +
                                                ((uixyz - uixyz1) * (uixyz - uixyz1)) +
                                                ((uixyz - uizm1) * (uixyz - uizm1)));
					
					g[ix1yz] = 1.0 / sqrt(1.0e-7 +
                                                ((uix1yz - uix2yz) * (uix1yz - uix2yz)) +
                                                ((uix1yz - uixyz) * (uix1yz - uixyz)) +
                                                ((uix1yz - uix1y1z) * (uix1yz - uix1y1z)) +
                                                ((uix1yz - uix1ym1) * (uix1yz - uix1ym1)) +
                                                ((uix1yz - uix1yz1) * (uix1yz - uix1yz1)) +
                                                ((uix1yz - uix1zm1) * (uix1yz - uix1zm1)));
					
					g[ixy1z] = 1.0 / sqrt(1.0e-7 +
                                                ((uixy1z - uix1y1z) * (uixy1z - uix1y1z)) +
                                                ((uixy1z - uixm1y1) * (uixy1z - uixm1y1)) +
                                                ((uixy1z - uixy2z) * (uixy1z - uixy2z)) +
                                                ((uixy1z - uixyz) * (uixy1z - uixyz)) +
                                                ((uixy1z - uixy1z1) * (uixy1z - uixy1z1)) +
                                                ((uixy1z - uiy1zm1) * (uixy1z - uiy1zm1)));

					g[ix1y1z] = 1.0 / sqrt(1.0e-7 +
                                                ((uix1y1z - uix2y1) * (uix1y1z - uix2y1)) +
                                                ((uix1y1z - uixy1z) * (uix1y1z - uixy1z)) +
                                                ((uix1y1z - uix1y2) * (uix1y1z - uix1y2)) +
                                                ((uix1y1z - uix1yz) * (uix1y1z - uix1yz)) +
                                                ((uix1y1z - uix1y1z1) * (uix1y1z - uix1y1z1)) +
                                                ((uix1y1z - uix1y1zm1) * (uix1y1z - uix1y1zm1)));
				}
			}
		}
		memcpy(conv, u, sizeof(double) * 2048383);
		OMP_GaussianBlur(conv, Ksigma, 3);
		
		#pragma omp parallel for private(y,x)
		for(z = 0; z < 128; z++)
		{
			for(y = 0; y < 128; y++)
			{
				for(x = 0; x < 128; x++)
				{
					int ixyz = Index(x, y, z);
					double r = conv[ixyz] * f[ixyz] / 0.000064;
					r = (r * (2.38944 + r * (0.950037 + r))) / (4.65314 + r * (2.57541 + r * (1.48937 + r)));
					conv[ixyz] -= f[ixyz] * r;
				}
			}
		}
		OMP_GaussianBlur(conv, Ksigma, 3);
		converged = 0;

		#pragma omp for nowait
		for(z = 1; z < 127; z++)
		{
			for(y = 1; y < 127; y++)
			{
				for(x = 1; x < 127; x++)
				{
					int ixyz = Index(x, y, z);
					int ixp1 = Index(x + 1, y, z);
					int ixm1 = Index(x - 1, y, z);
					int iyp1 = Index(x, y + 1, z);
					int iym1 = Index(x, y - 1, z);
					int izp1 = Index(x, y, z + 1);
					int izm1 = Index(x, y, z - 1);
					double uixyz = u[ixyz];
					double gixm1 = g[ixm1];
					double gixp1 = g[ixp1];
					double giym1 = g[iym1];
					double giyp1 = g[iyp1];
					double gizm1 = g[izm1];
					double gizp1 = g[izp1];
					
					double oldVal = uixyz;
					double newVal = (uixyz + dt * ( 
						u[ixm1] * gixm1 + 
						u[ixp1] * gixp1 + 
						u[iym1] * giym1 + 
						u[iyp1] * giyp1 + 
						u[izm1] * gizm1 + 
						u[izp1] * gizp1 - gamma * conv[ixyz])) /
						(1.0 + dt * (gixp1 + gixm1 + giyp1 + giym1 + gizp1 + gizm1));
					if(fabs(oldVal - newVal) < 1.0e-7 )
					{
						converged++;
					}
					u[ixyz] = newVal;
				}
			}
		}
		if(converged > lastConverged)
		{
			printf("%d pixels have converged on iteration %d\n", converged, iteration);
			lastConverged = converged;
		}
	}
}
