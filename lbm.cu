/* ============================================================ */
/* LATTICE BOLTZMANN SIMULATOR                                  */
/* GPU accelerated with CUDA                                    */
/*                                                              */
/* Copyright (c) 2017 Tom Scherlis and Henry Friedlander        */
/* For SSA Physics 3                                            */
/* ============================================================ */

//comment out this line to hide prints:
#define DEBUG
#define trace_x 50
#define trace_y 57
#define DEBUG_DELAY 0

#ifdef DEBUG
# define DEBUG_PRINT(x) printf x
#else
# define DEBUG_PRINT(x) do {} while (0)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <malloc.h>
//#include <vector_types.h>

// OpenGL Graphics includes
#include <dependencies/common/inc/helper_gl.h>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <dependencies/common/inc/GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <dependencies/common/inc/GL/freeglut.h>
#endif

//Cuda includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <dependencies/common/inc/helper_functions.h>

#include <dependencies/common/inc/helper_cuda.h>

//-----------------------------------------------------------------------//
//                     GLOBAL VARS AND STRUCT DEFS                       //
//-----------------------------------------------------------------------//


// texture and pixel objects
GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource;

//timing variables:
unsigned long last_draw_time = 0;
unsigned long current_draw_time = 0;
float delta_t = 1;

typedef struct {
	//velocities:
	float ux;	//x velocity
	float uy;	//y velocity

	float rho;	//density. aka rho
	float f[9];
}lbm_node;


typedef struct {
	char ex; //x location
	char ey; //y location
	float wt; //weight
	unsigned char op; //opposite char
}d2q9_node;

typedef struct {
	float viscosity;
	float omega;
	unsigned int height;
	unsigned int width;
	float contrast;
	float v;
	unsigned char mode;
}parameter_set;
parameter_set params;

//GPU/CPU interop memory pointers:
unsigned char state = 0;
lbm_node* array1;
lbm_node* array2;
lbm_node* array1_gpu;
lbm_node* array2_gpu;
unsigned char* barrier;
unsigned char* barrier_gpu;
d2q9_node* d2q9_gpu;
parameter_set* params_gpu;

enum directions {
	d0 = 0,
	dE,
	dN,
	dW,
	dS,
	dNE,
	dNW,
	dSW,
	dSE
};

enum render_modes {
	mRho,
	mCurl,
	mSpeed,
	mUx,
	mUy
};

//cuda error variables:
cudaError_t ierrAsync;
cudaError_t ierrSync;


void getParams(parameter_set* params)
{
	float viscosity = 0.005;

	params->contrast = 75;
	params->v = 0.1;
	params->mode = mCurl;
	params->omega = 1 / (3 * viscosity + 0.5);
	params->height = 300;
	params->width = 400;
}

//------------------------------------------------------------------------------//
//                                HELPER FUNCTIONS                              //
//------------------------------------------------------------------------------//

//get 1d flat index from row and col
int getIndex_cpu(int x, int y) {
	return y * params.width + x;
}

//--------------------------------------------------------------------------------//
//                   CUDA HELPER AND RENDER FUNCTIONS                             //
//--------------------------------------------------------------------------------//
__device__
unsigned char clip(int n) {
	return n > 255 ? 255 : (n < 0 ? 0 : n);
}

//get 1d flat index from row and col
__device__
int getIndex(int x, int y, parameter_set* params)
{
	return y * params->width + x;
}

__device__
void printNode(lbm_node* node, lbm_node* before, lbm_node* after)
{
	DEBUG_PRINT(("\t\t\ttest: %x\n", after));
	DEBUG_PRINT(("\trho: %.6f\n\tux: %.6f\n\tuy: %.6f\n\tvN: %.6f\n\tvE: %.6f\n\tvW: %.6f\n\tvS: %.6f\n\tv0: %.6f\n\tvNW: %.6f\n\tvNE: %.6f\n\tvSW: %.6f\n\tvSE: %.6f\n",
		node->rho,
		node->ux,
		node->uy,
		(node->f)[dN],
		(node->f)[dE],
		(node->f)[dW],
		(node->f)[dS],
		(node->f)[d0],
		(node->f)[dNW],
		(node->f)[dNE],
		(node->f)[dSW],
		(node->f)[dSW]
		));

	DEBUG_PRINT(("\n\tbefore: %p \n\tafter: %p \n\t node : %p \n", before, after, node));
}

__device__
uchar4 getRGB_roh(float i, parameter_set* params)
{

	uchar4 val;
	if (i == i)
	{
		int j = (1 - i) * 255 * 10; // approximately -255 to 255;

		val.x = 0;
		val.w = 0;
		val.z = 255;

		if (j > 0)
		{
			val.y = clip(j);
			val.z = 0;
		}
		else
		{
			val.z = clip(-j);
			val.y = 0;
		}
	}
	else
	{
		val.y = 0;
		val.x = 255;
		val.w = 0;
		val.z = 255;
	}
	return val;
}

__device__
uchar4 getRGB_u(float i)
{

	uchar4 val;
	if (i == i)
	{
		val.w = 255;
		val.x = 0;
		val.y = clip(i*255.0 / 1.0);
		val.z = 0;
	}
	else
	{
		val.w = 255;
		val.x = 255;
		val.y = 0;
		val.z = 0;
	}
	return val;
}

__device__
float computeCurlMiddleCase(int x, int y, lbm_node * array1, parameter_set* params) {
	return (array1[getIndex(x, y + 1, params)].ux
		- array1[getIndex(x, y - 1, params)].ux)
		- (array1[getIndex(x + 1, y, params)].uy
			- array1[getIndex(x - 1, y, params)].uy);
}

__device__
uchar4 getRGB_curl(int x, int y, lbm_node* array, parameter_set* params)
{

	uchar4 val;
	val.x = 0;
	val.w = 255;
	if (0 < x && x < params->width - 1) {
		if (0 < y && y < params->height - 1) {
			//picture[getIndex(x,y)]
			if (computeCurlMiddleCase(x, y, array, params) > 0)
			{
				val.y = clip(20 * params->contrast * computeCurlMiddleCase(x, y, array, params));
				val.z = 0;
			}
			else
			{
				val.z = clip(20 * params->contrast * -1 * computeCurlMiddleCase(x, y, array, params));
				val.y = 0;
			}
		}
		//else {
		//	//picture[getIndex(x,y)]
		//	colorIndex = (int)(nColors * (0.5 + computeCurlEdgeCase(col, row, array) * contrast * 0.3));
		//}
	}

	if (array[getIndex(x, y, params)].rho != array[getIndex(x, y, params)].rho)
	{
		val.x = 255;
		val.y = 0;
		val.z = 0;
		val.w = 255;
	}
	return val;
}

__device__
void computeColor(lbm_node* array, int x, int y, parameter_set* params, uchar4* image, unsigned char* barrier)
{
	int i = getIndex(x, y, params);

	if (barrier[i] == 1)
	{
		//DEBUG_PRINT("drawin a barrier!\n");
		image[i].w = 255;
		image[i].x = 255;
		image[i].y = 255;
		image[i].z = 255;
	}
	else
	{
		switch (params->mode)
		{
		case(mRho):
			image[i] = getRGB_roh(array[i].rho, params);
			break;
		case(mCurl):
			image[i] = getRGB_curl(x, y, array, params);
			break;
		case(mSpeed):
			image[i] = getRGB_u(sqrt(array[i].ux * array[i].ux + array[i].uy * array[i].uy));
			break;
		case(mUx):
			image[i] = getRGB_u(array[i].ux);
			break;
		case(mUy):
			image[i] = getRGB_u(array[i].uy);
			break;
		}
	}
	if (x == trace_x && y == trace_y)
	{
		image[i].x = 255;
		image[i].y = 0;
		image[i].z = 255;
		image[i].w = 255;
	}
}

//--------------------------------------------------------------------------------//
//                   CUDA COLLIDE STEP KERNEL AND DEVICES                         //
//--------------------------------------------------------------------------------//

__device__
void macro_gen(float* f, float* ux, float* uy, float* rho, int i, parameter_set* params)
{
	const float top_row = f[6] + f[2] + f[5];
	const float mid_row = f[3] + f[0] + f[1];
	const float bot_row = f[7] + f[4] + f[8];

	if (i == getIndex(trace_x, trace_y, params))
		for (int i = 0; i < 9;i++)
		{
			DEBUG_PRINT(("\t\tmacro_gen: f[%d]=%.6f\n", i, f[i]));
		}

	*rho = top_row + mid_row + bot_row;
	if (*rho > 0)
	{
		*ux = ((f[5] + f[1] + f[8]) - (f[6] + f[3] + f[7])) / (*rho);
		*uy = (bot_row - top_row) / (*rho);
	}
	else
	{
		*ux = 0;
		*uy = 0;
	}

	return;
}

//return acceleration
__device__
float accel_gen(int node_num, float ux, float uy, float u2, float rho, d2q9_node* d2q9)
{
	float u_direct = ux * d2q9[node_num].ex + uy * (-d2q9[node_num].ey);
	float unweighted = 1 + 3 * u_direct + 4.5*u_direct*u_direct - 1.5*u2;

	return rho * d2q9[node_num].wt * unweighted;
}

__global__
void collide(d2q9_node* d2q9, lbm_node* before, lbm_node* after, parameter_set* params, unsigned char* barrier)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = getIndex(x, y, params);

	//toss out out of bounds
	if (x<0 || x >= params->width || y<0 || y >= params->height)
		return;

	if (x == trace_x && y == trace_y)
	{
		DEBUG_PRINT(("\n\nPre-Collision (before):\n"));
		printNode(&(before[i]), before, after);
		DEBUG_PRINT(("\n\nPre-Collision (after) (not used):\n"));
		printNode(&(after[i]), before, after);
	}


	/*if (barrier[i] == 1)
	{
		//after[i].rho =1;
		//after[i].ux = 0;
		//after[i].uy = 0;
		for (int j = 0;j < 9;j++)
		{
			(after[i].f)[j] = (before[i].f)[j];
		}
		if (x == trace_x && y == trace_y)
		{
			//DEBUG_PRINT(("\n\nPre-Collision (before):\n"));
			//printNode(&(before[i]), before, after);
			DEBUG_PRINT(("\n\nwall test results (after):\n"));
			printNode(&(after[i]), before, after);
		}

		return;
	}*/

	macro_gen(before[i].f, &(after[i].ux), &(after[i].uy), &(after[i].rho), i, params);

	int dir = 0;
	for (dir = 0; dir<9;dir += 1)
	{
		(after[i].f)[dir] = (before[i].f)[dir] + params->omega
			* (accel_gen(dir, after[i].ux, after[i].uy,
				after[i].ux * after[i].ux + after[i].uy
				* after[i].uy, after[i].rho, d2q9) - (before[i].f)[dir]);
	}
	return;
}


//--------------------------------------------------------------------------------//
//                   CUDA STREAM STEP KERNEL AND DEVICES                          //
//--------------------------------------------------------------------------------//
__device__
void doLeftWall(int x, int y, lbm_node* after, d2q9_node* d2q9, float v, parameter_set* params)
{
	//DEBUG_PRINT(("setting left wall to %.6f (wt: %.3f, v: %.3f)\n", d2q9[dE].wt  * (1 + 3 * v + 3 * v * v), d2q9[dE].wt,v));
	(after[getIndex(x, y, params)].f)[dE] = d2q9[dE].wt  * (1 + 3 * v + 3 * v * v);
	(after[getIndex(x, y, params)].f)[dNE] = d2q9[dNE].wt * (1 + 3 * v + 3 * v * v);
	(after[getIndex(x, y, params)].f)[dSE] = d2q9[dSE].wt * (1 + 3 * v + 3 * v * v);
}

__device__
void doRightWall(int x, int y, lbm_node* after, d2q9_node* d2q9, float v, parameter_set* params)
{
	(after[getIndex(x, y, params)].f)[dW] = d2q9[dW].wt  * (1 - 3 * v + 3 * v * v);
	(after[getIndex(x, y, params)].f)[dNW] = d2q9[dNW].wt * (1 - 3 * v + 3 * v * v);
	(after[getIndex(x, y, params)].f)[dSW] = d2q9[dSW].wt * (1 - 3 * v + 3 * v * v);
}

//(top and bottom walls)
__device__
void doFlanks(int x, int y, lbm_node* after, d2q9_node* d2q9, float v, parameter_set* params)
{
	(after[getIndex(x, y, params)].f)[d0] = d2q9[d0].wt  * (1 - 1.5 * v * v);
	(after[getIndex(x, y, params)].f)[dE] = d2q9[dE].wt  * (1 + 3 * v + 3 * v * v);
	(after[getIndex(x, y, params)].f)[dW] = d2q9[dW].wt  * (1 - 3 * v + 3 * v * v);
	(after[getIndex(x, y, params)].f)[dN] = d2q9[dN].wt  * (1 - 1.5 * v * v);
	(after[getIndex(x, y, params)].f)[dS] = d2q9[dS].wt  * (1 - 1.5 * v * v);
	(after[getIndex(x, y, params)].f)[dNE] = d2q9[dNE].wt * (1 + 3 * v + 3 * v * v);
	(after[getIndex(x, y, params)].f)[dSE] = d2q9[dSE].wt * (1 + 3 * v + 3 * v * v);
	(after[getIndex(x, y, params)].f)[dNW] = d2q9[dNW].wt * (1 - 3 * v + 3 * v * v);
	(after[getIndex(x, y, params)].f)[dSW] = d2q9[dSW].wt * (1 - 3 * v + 3 * v * v);
}

__device__
void streamEdgeCases(int x, int y, lbm_node* after, unsigned char* barrier,
	parameter_set* params, d2q9_node* d2q9)
{

	if (x == 0)
	{
		if (barrier[getIndex(x, y, params)] != 1)
		{
			//DEBUG_PRINT(("doing left wall!"));
			doLeftWall(x, y, after, d2q9, params->v, params);
		}
	}
	else if (x == params->width - 1)
	{
		if (barrier[getIndex(x, y, params)] != 1)
		{
			doRightWall(x, y, after, d2q9, params->v, params);
		}
	}
	else if (y == 0 || y == params->width - 1)
	{
		if (barrier[getIndex(x, y, params)] != 1)
		{
			doFlanks(x, y, after, d2q9, params->v, params);
		}
	}
}

//stream: handle particle propagation, ignoring edge cases.
__global__
void stream(d2q9_node* d2q9, lbm_node* before, lbm_node* after,
	unsigned char* barrier, parameter_set* params)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = getIndex(x, y, params);


	if (x == trace_x && y == trace_y)
	{
		DEBUG_PRINT(("\n\nPre-stream: (before)\n"));
		printNode(&(before[i]), before, after);
	}

	//toss out out of bounds and edge cases
	if (x < 0 || x >= params->width || y < 0 || y >= params->height)
		return;

	after[i].rho = before[i].rho;
	after[i].ux = before[i].ux;
	after[i].uy = before[i].uy;

	if (!(x > 0 && x < params->width - 1 && y > 0 && y < params->height - 1))
	{
		//return;
		streamEdgeCases(x, y, after, barrier, params, d2q9);
	}
	else
	{
		//propagate all f values around a bit
		int dir = 0;
		for (dir = 0;dir < 9;dir += 1)
		{
			(after[getIndex(d2q9[dir].ex + x, -d2q9[dir].ey + y, params)].f)[dir] =
				before[i].f[dir];
		}
	}
}

//--------------------------------------------------------------------------------//
//                   CUDA BOUNCE STEP KERNEL AND DEVICES                          //
//--------------------------------------------------------------------------------//

/*__device__
void bounceEdgeCases(int x, int y, lbm_node* after, unsigned char* barrier,
parameter_set* params, d2q9_node* d2q9)
{

}*/

__global__
void bounceAndRender(d2q9_node* d2q9, lbm_node* before, lbm_node* after,
	unsigned char* barrier, parameter_set* params, uchar4* image)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = getIndex(x, y, params);

	if (x == trace_x && y == trace_y)
	{
		DEBUG_PRINT(("\n\npre-barriers:\n"));
		printNode(&(after[i]), before, after);
	}

	//toss out out of bounds and edge cases
	if (x < 0 || x >= params->width || y < 0 || y >= params->height)
		return;

	if (x > 0 && x < params->width - 1 && y>0 && y < params->height - 1)
	{
		if (barrier[i] == 1)
		{
			int dir;
			for (dir = 1; dir < 9; dir += 1)
			{
				if (d2q9[dir].op > 0 && (after[i].f)[dir]>0)
				{
					(after[getIndex(d2q9[dir].ex + x, -d2q9[dir].ey + y, params)].f)[dir]
						= (before[i].f)[d2q9[dir].op];
					//printf("doin a barrier bounce! %d\n",dir);
					//(after[i].f)[dir] += (after[i].f)[d2q9[dir].op];
					//	+ (after[i].f)[dir];
					//(after[i].f)[d2q9[dir].op] = 0;

				}
			}
		}
	}
	else
	{
		//bounceEdgeCases(x, y, after, barrier, params, d2q9);
	}

	if (x == trace_x && y == trace_y)
	{
		DEBUG_PRINT(("\n\nFinal rendered:\n"));
		printNode(&(after[i]), before, after);
	}

	computeColor(after, x, y, params, image, barrier);
}

//--------------------------------------------------------------------------------//
//                        CUDA INITIALIZER FUNCTIONS                              //
//--------------------------------------------------------------------------------//

//provide LBM constants for d2q9 style nodes
//assumes positive is up and right, whereas our program assumes positive down and right.
void init_d2q9(d2q9_node* d2q9)
{
	d2q9[0].ex =  0;	d2q9[0].ey =  0;	d2q9[0].wt = 4.0 /  9.0;	d2q9[0].op = 0;
	d2q9[1].ex =  1;	d2q9[1].ey =  0;	d2q9[1].wt = 1.0 /  9.0;	d2q9[1].op = 3;
	d2q9[2].ex =  0;	d2q9[2].ey =  1;	d2q9[2].wt = 1.0 /  9.0;	d2q9[2].op = 4;
	d2q9[3].ex = -1;	d2q9[3].ey =  0;	d2q9[3].wt = 1.0 /  9.0;	d2q9[3].op = 1;
	d2q9[4].ex =  0;	d2q9[4].ey = -1;	d2q9[4].wt = 1.0 /  9.0;	d2q9[4].op = 2;
	d2q9[5].ex =  1;	d2q9[5].ey =  1;	d2q9[5].wt = 1.0 / 36.0;	d2q9[5].op = 7;
	d2q9[6].ex = -1;	d2q9[6].ey =  1;	d2q9[6].wt = 1.0 / 36.0;	d2q9[6].op = 8;
	d2q9[7].ex = -1;	d2q9[7].ey = -1;	d2q9[7].wt = 1.0 / 36.0;	d2q9[7].op = 5;
	d2q9[8].ex =  1;	d2q9[8].ey = -1;	d2q9[8].wt = 1.0 / 36.0;	d2q9[8].op = 6;
}

void setBarrier(unsigned char* barrier)
{
	//int H = params.height;
	//int W = params.width;
	/*
	barrier[getIndex_cpu(99, 100)] = 1;
	barrier[getIndex_cpu(100, 100)] = 1;
	barrier[getIndex_cpu(101, 100)] = 1;
	barrier[getIndex_cpu(102, 100)] = 1;
	barrier[getIndex_cpu(103, 100)] = 1;
	*/

	for (int i = 0; i < 20; i++)
	{
		
		/*for (int j = 0; j < 10; j++)
		{
		//if(i==0 || i==9 || j==0 || j == 9)
		barrier[getIndex_cpu(i+100, j+100)] = 1;
		}
		//barrier[getIndex_cpu(50, 47 + i)] = 1;
		*/
		barrier[getIndex_cpu(100 + (i / 3), 100 + i)] = 1;
	}
}

void zeroSite(lbm_node* array, int index)
{
	int dir = 0;
	for (dir = 0; dir < 9; dir += 1)
	{
		(array[index].f)[dir] = 0;
	}
	array[index].rho = 1;
	array[index].ux = 0;
	array[index].uy = 0;
}

void initFluid() {
	getParams(&params);
	int W = params.width;
	int H = params.height;
	float v = params.v;
	printf("velocity is %.6f my dude\n", v);

	barrier = (unsigned char*)calloc(W*H, sizeof(unsigned char));
	setBarrier(barrier);

	array1 = (lbm_node*)calloc(W*H, sizeof(lbm_node));
	array2 = (lbm_node*)calloc(W*H, sizeof(lbm_node));

	lbm_node* before = array1;

	d2q9_node* d2q9 = (d2q9_node*)calloc(9, sizeof(d2q9_node));
	init_d2q9(d2q9);

	DEBUG_PRINT(("\tTESTWEIGHT = %.6f", d2q9[dE].wt));

	int i;
	for (int x = 0; x < params.width; x++)
	{
		for (int y = 0; y < params.height; y++)
		{
			i = getIndex_cpu(x, y);
			/*
			if (barrier[i] == 1)
			{
				zeroSite(before, i);
				DEBUG_PRINT(("there's a barrier here!\n"));
			}
			else
			{*/
				(before[i].f)[d0] = d2q9[d0].wt  * (1 - 1.5 * v * v);
				(before[i].f)[dE] = d2q9[dE].wt  * (1 + 3 * v + 3 * v * v);
				(before[i].f)[dW] = d2q9[dW].wt  * (1 - 3 * v + 3 * v * v);
				(before[i].f)[dN] = d2q9[dN].wt  * (1 - 1.5 * v * v);
				(before[i].f)[dS] = d2q9[dS].wt  * (1 - 1.5 * v * v);
				(before[i].f)[dNE] = d2q9[dNE].wt * (1 + 3 * v + 3 * v * v);
				(before[i].f)[dSE] = d2q9[dSE].wt * (1 + 3 * v + 3 * v * v);
				(before[i].f)[dNW] = d2q9[dNW].wt * (1 - 3 * v + 3 * v * v);
				(before[i].f)[dSW] = d2q9[dSW].wt * (1 - 3 * v + 3 * v * v);
				before[i].rho = 1;
				before[i].ux = params.v;
				before[i].uy = 0;
			//}
		}
	}

	ierrSync = cudaMalloc(&d2q9_gpu, 9 * sizeof(d2q9_node));
	if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
	ierrSync = cudaMalloc(&params_gpu, sizeof(parameter_set));
	if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
	ierrSync = cudaMalloc(&barrier_gpu, sizeof(unsigned char)*W*H);
	if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
	ierrSync = cudaMalloc(&array1_gpu, sizeof(lbm_node)*W*H);
	if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
	ierrSync = cudaMalloc(&array2_gpu, sizeof(lbm_node)*W*H);
	if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }


	ierrSync = cudaMemcpy(d2q9_gpu, d2q9, sizeof(d2q9_node) * 9, cudaMemcpyHostToDevice);
	if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
	ierrSync = cudaMemcpy(params_gpu, &params, sizeof(params), cudaMemcpyHostToDevice);
	if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
	ierrSync = cudaMemcpy(barrier_gpu, barrier, sizeof(unsigned char)*W*H, cudaMemcpyHostToDevice);
	if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
	ierrSync = cudaMemcpy(array1_gpu, array1, sizeof(lbm_node)*W*H, cudaMemcpyHostToDevice);
	if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
	ierrSync = cudaMemcpy(array2_gpu, array2, sizeof(lbm_node)*W*H, cudaMemcpyHostToDevice);
	if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }

	cudaDeviceSynchronize();

	return;
}

//determine front and back lattice buffer orientation
//and launch all 3 LBM kernels
void kernelLauncher(uchar4* image)
{
	cudaMemcpy(barrier_gpu, barrier, sizeof(unsigned char)*params.width * params.height, cudaMemcpyHostToDevice);

	lbm_node* before = array1_gpu;
	lbm_node* after = array2_gpu;
	//lbm_node* temp;

	/*
	//assign lattice buffers
	if (!state) {
	before = array1_gpu;
	after = array2_gpu;
	}
	else {
	before = array2_gpu;
	after = array1_gpu;
	}
	state = !state;
	*/

	DEBUG_PRINT(("these are the addresses: \n\t\tb4=%p\taft=%p\n\t\tar1=%p\tar2=%p", before, after, array1_gpu, array2_gpu));

	//determine number of threads and blocks required
	dim3 threads_per_block = dim3(32, 32, 1);
	dim3 number_of_blocks = dim3(params.width / 32 + 1, params.height / 32 + 1, 1);

	collide << < number_of_blocks, threads_per_block >> > (d2q9_gpu, before, after, params_gpu, barrier_gpu);

	ierrSync = cudaGetLastError();
	ierrAsync = cudaDeviceSynchronize(); // Wait for the GPU to finish
	if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
	if (ierrAsync != cudaSuccess) { DEBUG_PRINT(("Async error: %s\n", cudaGetErrorString(ierrAsync))); }

	before = array2_gpu;
	after = array1_gpu;
	stream << < number_of_blocks, threads_per_block >> > (d2q9_gpu, before, after, barrier_gpu, params_gpu);

	ierrSync = cudaGetLastError();
	ierrAsync = cudaDeviceSynchronize(); // Wait for the GPU to finish
	if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
	if (ierrAsync != cudaSuccess) { DEBUG_PRINT(("Async error: %s\n", cudaGetErrorString(ierrAsync))); }


	bounceAndRender << < number_of_blocks, threads_per_block >> > (d2q9_gpu, before, after, barrier_gpu, params_gpu, image);

	ierrSync = cudaGetLastError();
	ierrAsync = cudaDeviceSynchronize(); // Wait for the GPU to finish
	if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
	if (ierrAsync != cudaSuccess) { DEBUG_PRINT(("Async error: %s\n", cudaGetErrorString(ierrAsync))); }


}

//-----------------------------------------------------------//
//              OPENGL CALLBACK FUNCTIONS                    //
//-----------------------------------------------------------//

//keyboard callback
void keyboard(unsigned char a, int b, int c)
{
	DEBUG_PRINT(("%x pressed\n", a));
}

//special keyboard callback
void handleSpecialKeypress(int a, int b, int c)
{

}

//mouse move callback
void mouseMove(int a, int b)
{

}

//mouse drag callback
void mouseDrag(int x, int y)
{
	if (x >= params.width || y >= params.height)
		return;
	barrier[getIndex_cpu(x, y)] = 1;
}

//gl exit callback
void exitfunc()
{
	//empty all cuda resources
	if (pbo)
	{
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffers(1, &pbo);
		glDeleteTextures(1, &tex);
	}

	cudaFree(array1_gpu);
	cudaFree(array2_gpu);
	cudaFree(barrier_gpu);
	cudaFree(params_gpu);
	cudaFree(d2q9_gpu);
	//add cudaFree calls here!
}

//display stats of all detected cuda capable devices,
//and return the number
int deviceQuery()
{
	cudaDeviceProp prop;
	int nDevices = 1;
	cudaError_t ierr;


	ierr = cudaGetDeviceCount(&nDevices);

	int i = 0;
	for (i = 0; i < nDevices; ++i)
	{
		ierr = cudaGetDeviceProperties(&prop, i);
		DEBUG_PRINT(("Device number: %d\n", i));
		DEBUG_PRINT(("  Device name: %s\n", prop.name));
		DEBUG_PRINT(("  Compute capability: %d.%d\n", prop.major, prop.minor));
		DEBUG_PRINT(("  Max threads per block: %d\n", prop.maxThreadsPerBlock));
		DEBUG_PRINT(("  Max threads in X-dimension of block: %d\n", prop.maxThreadsDim[0]));
		DEBUG_PRINT(("  Max threads in Y-dimension of block: %d\n", prop.maxThreadsDim[1]));
		DEBUG_PRINT(("  Max threads in Z-dimension of block: %d\n\n", prop.maxThreadsDim[2]));
	}

	return nDevices;
}



//----------------------------------------------------------------------------//
//               RENDERING AND DISPLAY FUNCTIONS                              //
//----------------------------------------------------------------------------//

//render the image (but do not display it yet)
void render(int delta_t) {
	//reset image pointer
	uchar4 *d_out = 0;
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	//set d_out as a texture memory pointer
	cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL, cuda_pbo_resource);

	//launch cuda kernels to calculate LBM step
	kernelLauncher(d_out);

	//unmap the resources for next time
	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

//update textures to reflect texture memory
void drawTexture() {
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, params.width, params.height,
		0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(0, params.height);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(params.width, params.height);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(params.width, 0);
	glEnd();
	glDisable(GL_TEXTURE_2D);
}

//update the live display
void display(int delta_t) {

	//launch cuda kernels to update Lattice-Boltzmann,
	//flip front and back LBM buffers,
	//and update texture memory
	render(delta_t);

	//redraw textures
	drawTexture();

	//swap the buffers
	glutSwapBuffers();
}

// (gl idle callback) handle frame limitting, fps calculating, and call display functions
// triggered when glutmainloop() is idle
float fps;
void update()
{
	//find time since last frame update. Will replace with timers later for precision beyond 1ms
	current_draw_time = clock();
	delta_t = current_draw_time - last_draw_time;

	//limit framerate to 5Hz
	if (delta_t < DEBUG_DELAY)
	{
		return;
	}

	last_draw_time = current_draw_time;

	//calculate fps
	fps = delta_t != 0 ? 1000.0 / delta_t : 0;
	//DEBUG_PRINT("in render: delta t = %.6f\n", delta_t);
	display(delta_t);
}

//creates and binds texture memory
void initPixelBuffer() {
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * params.width * params.height
		* sizeof(GLubyte), 0, GL_STREAM_DRAW);
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

void initGLUT(int *argc, char **argv) {
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(params.width, params.height);
	glutCreateWindow("LBM GPU");

#ifndef __APPLE__
	glewInit();
#endif

	gluOrtho2D(0, params.width, params.height, 0);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(handleSpecialKeypress);
	glutPassiveMotionFunc(mouseMove);
	glutMotionFunc(mouseDrag);
	glutDisplayFunc(update);
	glutIdleFunc(update);
	initPixelBuffer();
}

//---------------------------------------------------------------------------------------//
//                                MAIN FUNCTION!!                                        //
//---------------------------------------------------------------------------------------//

int main(int argc, char** argv) {

	//discover all Cuda-capable hardware
	int i = 1;//deviceQuery();
	//DEBUG_PRINT(("num devices is %d\n", i));

	if (i < 1)
	{
		//DEBUG_PRINT(("ERROR: no cuda capable hardware detected!\n"));
		getchar();
		return 0; //return if no cuda-capable hardware is present
	}

	//allocate memory and initialize fluid arrays
	initFluid();

	//construct output window
	initGLUT(&argc, argv);

	//run gl main loop!
	glutMainLoop();

	//declare exit callback
	atexit(exitfunc);
	getchar();
	return 0;
}