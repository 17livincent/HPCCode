
/*
 * FLAME GPU v 1.5.X for CUDA 9
 * Copyright University of Sheffield.
 * Original Author: Dr Paul Richmond (user contributions tracked on https://github.com/FLAMEGPU/FLAMEGPU)
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *
 * University of Sheffield retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * University of Sheffield is strictly prohibited.
 *
 * For terms of licence agreement please attached licence or view licence
 * on www.flamegpu.com website.
 *
 */


  //Disable internal thrust warnings about conversions
  #ifdef _MSC_VER
  #pragma warning(push)
  #pragma warning (disable : 4267)
  #pragma warning (disable : 4244)
  #endif
  #ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wunused-parameter"
  #endif

  // includes
  #include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cub/cub.cuh>

// include FLAME kernels
#include "FLAMEGPU_kernals.cu"


#ifdef _MSC_VER
#pragma warning(pop)
#endif
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

/* Error check function for safe CUDA API calling */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* Error check function for post CUDA Kernel calling */
#define gpuErrchkLaunch() { gpuLaunchAssert(__FILE__, __LINE__); }
inline void gpuLaunchAssert(const char *file, int line, bool abort=true)
{
	gpuAssert( cudaPeekAtLastError(), file, line );
#ifdef _DEBUG
	gpuAssert( cudaDeviceSynchronize(), file, line );
#endif
   
}

/* SM padding and offset variables */
int SM_START;
int PADDING;

unsigned int g_iterationNumber;

/* Agent Memory */

/* example_agent Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_example_agent_list* d_example_agents;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_example_agent_list* d_example_agents_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_example_agent_list* d_example_agents_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_example_agent_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_example_agent_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_example_agent_values;  /**< Agent sort identifiers value */

/* example_agent state variables */
xmachine_memory_example_agent_list* h_example_agents_default;      /**< Pointer to agent list (population) on host*/
xmachine_memory_example_agent_list* d_example_agents_default;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_example_agent_default_count;   /**< Agent population size counter */ 


/* Variables to track the state of host copies of state lists, for the purposes of host agent data access.
 * @future - if the host data is current it may be possible to avoid duplicating memcpy in xml output.
 */
unsigned int h_example_agents_default_variable_id_data_iteration;
unsigned int h_example_agents_default_variable_x_data_iteration;
unsigned int h_example_agents_default_variable_y_data_iteration;
unsigned int h_example_agents_default_variable_colour_data_iteration;


/* Message Memory */

/* example_message Message variables */
xmachine_message_example_message_list* h_example_messages;         /**< Pointer to message list on host*/
xmachine_message_example_message_list* d_example_messages;         /**< Pointer to message list on device*/
xmachine_message_example_message_list* d_example_messages_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_example_message_count;         /**< message list counter*/
int h_message_example_message_output_type;   /**< message output type (single or optional)*/

  
/* CUDA Streams for function layers */
cudaStream_t stream1;

/* Device memory and sizes for CUB values */

void * d_temp_scan_storage_example_agent;
size_t temp_scan_storage_bytes_example_agent;


/*Global condition counts*/

/* RNG rand48 */
RNG_rand48* h_rand48;    /**< Pointer to RNG_rand48 seed list on host*/
RNG_rand48* d_rand48;    /**< Pointer to RNG_rand48 seed list on device*/

/* Cuda Event Timers for Instrumentation */
#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEvent_t instrument_iteration_start, instrument_iteration_stop;
	float instrument_iteration_milliseconds = 0.0f;
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS)
	cudaEvent_t instrument_start, instrument_stop;
	float instrument_milliseconds = 0.0f;
#endif

/* CUDA Parallel Primatives variables */
int scan_last_sum;           /**< Indicates if the position (in message list) of last message*/
int scan_last_included;      /**< Indicates if last sum value is included in the total sum count*/

/* Agent function prototypes */

/** example_agent_output_example
 * Agent function prototype for output_example function of example_agent agent
 */
void example_agent_output_example(cudaStream_t &stream);

/** example_agent_input_example
 * Agent function prototype for input_example function of example_agent agent
 */
void example_agent_input_example(cudaStream_t &stream);

  
void setPaddingAndOffset()
{
    PROFILE_SCOPED_RANGE("setPaddingAndOffset");
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int x64_sys = 0;

	// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
	if (deviceProp.major == 9999 && deviceProp.minor == 9999){
		printf("Error: There is no device supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}
    
    //check if double is used and supported
#ifdef _DOUBLE_SUPPORT_REQUIRED_
	printf("Simulation requires full precision double values\n");
	if ((deviceProp.major < 2)&&(deviceProp.minor < 3)){
		printf("Error: Hardware does not support full precision double values!\n");
		exit(EXIT_FAILURE);
	}
    
#endif

	//check 32 or 64bit
	x64_sys = (sizeof(void*)==8);
	if (x64_sys)
	{
		printf("64Bit System Detected\n");
	}
	else
	{
		printf("32Bit System Detected\n");
	}

	SM_START = 0;
	PADDING = 0;
  
	//copy padding and offset to GPU
	gpuErrchk(cudaMemcpyToSymbol( d_SM_START, &SM_START, sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol( d_PADDING, &PADDING, sizeof(int)));     
}

int is_sqr_pow2(int x){
	int r = (int)pow(4, ceil(log(x)/log(4)));
	return (r == x);
}

int lowest_sqr_pow2(int x){
	int l;
	
	//escape early if x is square power of 2
	if (is_sqr_pow2(x))
		return x;
	
	//lower bound		
	l = (int)pow(4, floor(log(x)/log(4)));
	
	return l;
}

/* Unary function required for cudaOccupancyMaxPotentialBlockSizeVariableSMem to avoid warnings */
int no_sm(int b){
	return 0;
}

/* Unary function to return shared memory size for reorder message kernels */
int reorder_messages_sm_size(int blockSize)
{
	return sizeof(unsigned int)*(blockSize+1);
}


/** getIterationNumber
 *  Get the iteration number (host)
 *  @return a 1 indexed value for the iteration number, which is incremented at the start of each simulation step.
 *      I.e. it is 0 on up until the first call to singleIteration()
 */
extern unsigned int getIterationNumber(){
    return g_iterationNumber;
}

void initialise(char * inputfile){
    PROFILE_SCOPED_RANGE("initialise");

	//set the padding and offset values depending on architecture and OS
	setPaddingAndOffset();
  
    // Initialise some global variables
    g_iterationNumber = 0;

    // Initialise variables for tracking which iterations' data is accessible on the host.
    h_example_agents_default_variable_id_data_iteration = 0;
    h_example_agents_default_variable_x_data_iteration = 0;
    h_example_agents_default_variable_y_data_iteration = 0;
    h_example_agents_default_variable_colour_data_iteration = 0;
    



	printf("Allocating Host and Device memory\n");
    PROFILE_PUSH_RANGE("allocate host");
	/* Agent memory allocation (CPU) */
	int xmachine_example_agent_SoA_size = sizeof(xmachine_memory_example_agent_list);
	h_example_agents_default = (xmachine_memory_example_agent_list*)malloc(xmachine_example_agent_SoA_size);

	/* Message memory allocation (CPU) */
	int message_example_message_SoA_size = sizeof(xmachine_message_example_message_list);
	h_example_messages = (xmachine_message_example_message_list*)malloc(message_example_message_SoA_size);

	//Exit if agent or message buffer sizes are to small for function outputs

  /* Graph memory allocation (CPU) */
  

    PROFILE_POP_RANGE(); //"allocate host"
	

	//read initial states
	readInitialStates(inputfile, h_example_agents_default, &h_xmachine_memory_example_agent_default_count);

  // Read graphs from disk
  

  PROFILE_PUSH_RANGE("allocate device");
	
	/* example_agent Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_example_agents, xmachine_example_agent_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_example_agents_swap, xmachine_example_agent_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_example_agents_new, xmachine_example_agent_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_example_agent_keys, xmachine_memory_example_agent_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_example_agent_values, xmachine_memory_example_agent_MAX* sizeof(uint)));
	/* default memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_example_agents_default, xmachine_example_agent_SoA_size));
	gpuErrchk( cudaMemcpy( d_example_agents_default, h_example_agents_default, xmachine_example_agent_SoA_size, cudaMemcpyHostToDevice));
    
	/* example_message Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_example_messages, message_example_message_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_example_messages_swap, message_example_message_SoA_size));
	gpuErrchk( cudaMemcpy( d_example_messages, h_example_messages, message_example_message_SoA_size, cudaMemcpyHostToDevice));
		


  /* Allocate device memory for graphs */
  

    PROFILE_POP_RANGE(); // "allocate device"

    /* Calculate and allocate CUB temporary memory for exclusive scans */
    
    d_temp_scan_storage_example_agent = nullptr;
    temp_scan_storage_bytes_example_agent = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_example_agent, 
        temp_scan_storage_bytes_example_agent, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_example_agent_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_example_agent, temp_scan_storage_bytes_example_agent));
    

	/*Set global condition counts*/

	/* RNG rand48 */
    PROFILE_PUSH_RANGE("Initialse RNG_rand48");
	int h_rand48_SoA_size = sizeof(RNG_rand48);
	h_rand48 = (RNG_rand48*)malloc(h_rand48_SoA_size);
	//allocate on GPU
	gpuErrchk( cudaMalloc( (void**) &d_rand48, h_rand48_SoA_size));
	// calculate strided iteration constants
	static const unsigned long long a = 0x5DEECE66DLL, c = 0xB;
	int seed = 123;
	unsigned long long A, C;
	A = 1LL; C = 0LL;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		C += A*c;
		A *= a;
	}
	h_rand48->A.x = A & 0xFFFFFFLL;
	h_rand48->A.y = (A >> 24) & 0xFFFFFFLL;
	h_rand48->C.x = C & 0xFFFFFFLL;
	h_rand48->C.y = (C >> 24) & 0xFFFFFFLL;
	// prepare first nThreads random numbers from seed
	unsigned long long x = (((unsigned long long)seed) << 16) | 0x330E;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		x = a*x + c;
		h_rand48->seeds[i].x = x & 0xFFFFFFLL;
		h_rand48->seeds[i].y = (x >> 24) & 0xFFFFFFLL;
	}
	//copy to device
	gpuErrchk( cudaMemcpy( d_rand48, h_rand48, h_rand48_SoA_size, cudaMemcpyHostToDevice));

    PROFILE_POP_RANGE();

	/* Call all init functions */
	/* Prepare cuda event timers for instrumentation */
#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventCreate(&instrument_iteration_start);
	cudaEventCreate(&instrument_iteration_stop);
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS)
	cudaEventCreate(&instrument_start);
	cudaEventCreate(&instrument_stop);
#endif

	
  
  /* Init CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamCreate(&stream1));

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("Init agent_example_agent_default_count: %u\n",get_agent_example_agent_default_count());
	
#endif
} 


void sort_example_agents_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_example_agent_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_example_agent_default_count); 
	gridSize = (h_xmachine_memory_example_agent_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_example_agent_keys, d_xmachine_memory_example_agent_values, d_example_agents_default);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_example_agent_keys),  thrust::device_pointer_cast(d_xmachine_memory_example_agent_keys) + h_xmachine_memory_example_agent_default_count,  thrust::device_pointer_cast(d_xmachine_memory_example_agent_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_example_agent_agents, no_sm, h_xmachine_memory_example_agent_default_count); 
	gridSize = (h_xmachine_memory_example_agent_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_example_agent_agents<<<gridSize, blockSize>>>(d_xmachine_memory_example_agent_values, d_example_agents_default, d_example_agents_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_example_agent_list* d_example_agents_temp = d_example_agents_default;
	d_example_agents_default = d_example_agents_swap;
	d_example_agents_swap = d_example_agents_temp;	
}


void cleanup(){
    PROFILE_SCOPED_RANGE("cleanup");

    /* Call all exit functions */
	

	/* Agent data free*/
	
	/* example_agent Agent variables */
	gpuErrchk(cudaFree(d_example_agents));
	gpuErrchk(cudaFree(d_example_agents_swap));
	gpuErrchk(cudaFree(d_example_agents_new));
	
	free( h_example_agents_default);
	gpuErrchk(cudaFree(d_example_agents_default));
	

	/* Message data free */
	
	/* example_message Message variables */
	free( h_example_messages);
	gpuErrchk(cudaFree(d_example_messages));
	gpuErrchk(cudaFree(d_example_messages_swap));
	

    /* Free temporary CUB memory if required. */
    
    if(d_temp_scan_storage_example_agent != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_example_agent));
      d_temp_scan_storage_example_agent = nullptr;
      temp_scan_storage_bytes_example_agent = 0;
    }
    

  /* Graph data free */
  
  
  /* CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamDestroy(stream1));

  /* CUDA Event Timers for Instrumentation */
#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventDestroy(instrument_iteration_start);
	cudaEventDestroy(instrument_iteration_stop);
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS)
	cudaEventDestroy(instrument_start);
	cudaEventDestroy(instrument_stop);
#endif
}

void singleIteration(){
PROFILE_SCOPED_RANGE("singleIteration");

#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventRecord(instrument_iteration_start);
#endif

    // Increment the iteration number.
    g_iterationNumber++;

  /* set all non partitioned, spatial partitioned and On-Graph Partitioned message counts to 0*/
	h_message_example_message_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_example_message_count, &h_message_example_message_count, sizeof(int)));
	

	/* Call agent functions in order iterating through the layer functions */
	
	/* Layer 1*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("example_agent_output_example");
	example_agent_output_example(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: example_agent_output_example = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 2*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("example_agent_input_example");
	example_agent_input_example(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: example_agent_input_example = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
    
    /* Call all step functions */
	

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("agent_example_agent_default_count: %u\n",get_agent_example_agent_default_count());
	
#endif

#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventRecord(instrument_iteration_stop);
	cudaEventSynchronize(instrument_iteration_stop);
	cudaEventElapsedTime(&instrument_iteration_milliseconds, instrument_iteration_start, instrument_iteration_stop);
	printf("Instrumentation: Iteration Time = %f (ms)\n", instrument_iteration_milliseconds);
#endif
}

/* Environment functions */

//host constant declaration



/* Agent data access functions*/

    
int get_agent_example_agent_MAX_count(){
    return xmachine_memory_example_agent_MAX;
}


int get_agent_example_agent_default_count(){
	//continuous agent
	return h_xmachine_memory_example_agent_default_count;
	
}

xmachine_memory_example_agent_list* get_device_example_agent_default_agents(){
	return d_example_agents_default;
}

xmachine_memory_example_agent_list* get_host_example_agent_default_agents(){
	return h_example_agents_default;
}



/* Host based access of agent variables*/

/** int get_example_agent_default_variable_id(unsigned int index)
 * Gets the value of the id variable of an example_agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_example_agent_default_variable_id(unsigned int index){
    unsigned int count = get_agent_example_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_example_agents_default_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_example_agents_default->id,
                    d_example_agents_default->id,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_example_agents_default_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_example_agents_default->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of example_agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_example_agent_default_variable_x(unsigned int index)
 * Gets the value of the x variable of an example_agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_example_agent_default_variable_x(unsigned int index){
    unsigned int count = get_agent_example_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_example_agents_default_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_example_agents_default->x,
                    d_example_agents_default->x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_example_agents_default_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_example_agents_default->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of example_agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_example_agent_default_variable_y(unsigned int index)
 * Gets the value of the y variable of an example_agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_example_agent_default_variable_y(unsigned int index){
    unsigned int count = get_agent_example_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_example_agents_default_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_example_agents_default->y,
                    d_example_agents_default->y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_example_agents_default_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_example_agents_default->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of example_agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_example_agent_default_variable_colour(unsigned int index)
 * Gets the value of the colour variable of an example_agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable colour
 */
__host__ float get_example_agent_default_variable_colour(unsigned int index){
    unsigned int count = get_agent_example_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_example_agents_default_variable_colour_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_example_agents_default->colour,
                    d_example_agents_default->colour,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_example_agents_default_variable_colour_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_example_agents_default->colour[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access colour for the %u th member of example_agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}



/* Host based agent creation functions */
// These are only available for continuous agents.



/* copy_single_xmachine_memory_example_agent_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_example_agent_hostToDevice(xmachine_memory_example_agent_list * d_dst, xmachine_memory_example_agent * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->x, &h_agent->x, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, &h_agent->y, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->colour, &h_agent->colour, sizeof(float), cudaMemcpyHostToDevice));

}
/*
 * Private function to copy some elements from a host based struct of arrays to a device based struct of arrays for a single agent state.
 * Individual copies of `count` elements are performed for each agent variable or each component of agent array variables, to avoid wasted data transfer.
 * There will be a point at which a single cudaMemcpy will outperform many smaller memcpys, however host based agent creation should typically only populate a fraction of the maximum buffer size, so this should be more efficient.
 * @optimisation - experimentally find the proportion at which transferring the whole SoA would be better and incorporate this. The same will apply to agent variable arrays.
 * 
 * @param d_dst device destination SoA
 * @oaram h_src host source SoA
 * @param count the number of agents to transfer data for
 */
void copy_partial_xmachine_memory_example_agent_hostToDevice(xmachine_memory_example_agent_list * d_dst, xmachine_memory_example_agent_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->id, h_src->id, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->x, h_src->x, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, h_src->y, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->colour, h_src->colour, count * sizeof(float), cudaMemcpyHostToDevice));

    }
}

xmachine_memory_example_agent* h_allocate_agent_example_agent(){
	xmachine_memory_example_agent* agent = (xmachine_memory_example_agent*)malloc(sizeof(xmachine_memory_example_agent));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_example_agent));

	return agent;
}
void h_free_agent_example_agent(xmachine_memory_example_agent** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_example_agent** h_allocate_agent_example_agent_array(unsigned int count){
	xmachine_memory_example_agent ** agents = (xmachine_memory_example_agent**)malloc(count * sizeof(xmachine_memory_example_agent*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_example_agent();
	}
	return agents;
}
void h_free_agent_example_agent_array(xmachine_memory_example_agent*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_example_agent(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_example_agent_AoS_to_SoA(xmachine_memory_example_agent_list * dst, xmachine_memory_example_agent** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->id[i] = src[i]->id;
			 
			dst->x[i] = src[i]->x;
			 
			dst->y[i] = src[i]->y;
			 
			dst->colour[i] = src[i]->colour;
			
		}
	}
}


void h_add_agent_example_agent_default(xmachine_memory_example_agent* agent){
	if (h_xmachine_memory_example_agent_count + 1 > xmachine_memory_example_agent_MAX){
		printf("Error: Buffer size of example_agent agents in state default will be exceeded by h_add_agent_example_agent_default\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_example_agent_hostToDevice(d_example_agents_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_example_agent_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_example_agent_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_example_agents_default, d_example_agents_new, h_xmachine_memory_example_agent_default_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_example_agent_default_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_example_agent_default_count, &h_xmachine_memory_example_agent_default_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_example_agents_default_variable_id_data_iteration = 0;
    h_example_agents_default_variable_x_data_iteration = 0;
    h_example_agents_default_variable_y_data_iteration = 0;
    h_example_agents_default_variable_colour_data_iteration = 0;
    

}
void h_add_agents_example_agent_default(xmachine_memory_example_agent** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_example_agent_count + count > xmachine_memory_example_agent_MAX){
			printf("Error: Buffer size of example_agent agents in state default will be exceeded by h_add_agents_example_agent_default\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_example_agent_AoS_to_SoA(h_example_agents_default, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_example_agent_hostToDevice(d_example_agents_new, h_example_agents_default, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_example_agent_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_example_agent_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_example_agents_default, d_example_agents_new, h_xmachine_memory_example_agent_default_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_example_agent_default_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_example_agent_default_count, &h_xmachine_memory_example_agent_default_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_example_agents_default_variable_id_data_iteration = 0;
        h_example_agents_default_variable_x_data_iteration = 0;
        h_example_agents_default_variable_y_data_iteration = 0;
        h_example_agents_default_variable_colour_data_iteration = 0;
        

	}
}


/*  Analytics Functions */

int reduce_example_agent_default_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_example_agents_default->id),  thrust::device_pointer_cast(d_example_agents_default->id) + h_xmachine_memory_example_agent_default_count);
}

int count_example_agent_default_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_example_agents_default->id),  thrust::device_pointer_cast(d_example_agents_default->id) + h_xmachine_memory_example_agent_default_count, count_value);
}
int min_example_agent_default_id_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_example_agents_default->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_example_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_example_agent_default_id_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_example_agents_default->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_example_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_example_agent_default_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_example_agents_default->x),  thrust::device_pointer_cast(d_example_agents_default->x) + h_xmachine_memory_example_agent_default_count);
}

float min_example_agent_default_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_example_agents_default->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_example_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_example_agent_default_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_example_agents_default->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_example_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_example_agent_default_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_example_agents_default->y),  thrust::device_pointer_cast(d_example_agents_default->y) + h_xmachine_memory_example_agent_default_count);
}

float min_example_agent_default_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_example_agents_default->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_example_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_example_agent_default_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_example_agents_default->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_example_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_example_agent_default_colour_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_example_agents_default->colour),  thrust::device_pointer_cast(d_example_agents_default->colour) + h_xmachine_memory_example_agent_default_count);
}

float min_example_agent_default_colour_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_example_agents_default->colour);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_example_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_example_agent_default_colour_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_example_agents_default->colour);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_example_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}



/* Agent functions */


	
/* Shared memory size calculator for agent function */
int example_agent_output_example_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** example_agent_output_example
 * Agent function prototype for output_example function of example_agent agent
 */
void example_agent_output_example(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_example_agent_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_example_agent_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_example_agent_list* example_agents_default_temp = d_example_agents;
	d_example_agents = d_example_agents_default;
	d_example_agents_default = example_agents_default_temp;
	//set working count to current state count
	h_xmachine_memory_example_agent_count = h_xmachine_memory_example_agent_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_example_agent_count, &h_xmachine_memory_example_agent_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_example_agent_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_example_agent_default_count, &h_xmachine_memory_example_agent_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_example_message_count + h_xmachine_memory_example_agent_count > xmachine_message_example_message_MAX){
		printf("Error: Buffer size of example_message message will be exceeded in function output_example\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_example, example_agent_output_example_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = example_agent_output_example_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_example_message_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_example_message_output_type, &h_message_example_message_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (output_example)
	//Reallocate   : false
	//Input        : 
	//Output       : example_message
	//Agent Output : 
	GPUFLAME_output_example<<<g, b, sm_size, stream>>>(d_example_agents, d_example_messages);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_example_message_count += h_xmachine_memory_example_agent_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_example_message_count, &h_message_example_message_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_example_agent_default_count+h_xmachine_memory_example_agent_count > xmachine_memory_example_agent_MAX){
		printf("Error: Buffer size of output_example agents in state default will be exceeded moving working agents to next state in function output_example\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  example_agents_default_temp = d_example_agents;
  d_example_agents = d_example_agents_default;
  d_example_agents_default = example_agents_default_temp;
        
	//update new state agent size
	h_xmachine_memory_example_agent_default_count += h_xmachine_memory_example_agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_example_agent_default_count, &h_xmachine_memory_example_agent_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int example_agent_input_example_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_example_message));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** example_agent_input_example
 * Agent function prototype for input_example function of example_agent agent
 */
void example_agent_input_example(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_example_agent_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_example_agent_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_example_agent_list* example_agents_default_temp = d_example_agents;
	d_example_agents = d_example_agents_default;
	d_example_agents_default = example_agents_default_temp;
	//set working count to current state count
	h_xmachine_memory_example_agent_count = h_xmachine_memory_example_agent_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_example_agent_count, &h_xmachine_memory_example_agent_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_example_agent_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_example_agent_default_count, &h_xmachine_memory_example_agent_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_input_example, example_agent_input_example_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = example_agent_input_example_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (input_example)
	//Reallocate   : false
	//Input        : example_message
	//Output       : 
	//Agent Output : 
	GPUFLAME_input_example<<<g, b, sm_size, stream>>>(d_example_agents, d_example_messages);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_example_agent_default_count+h_xmachine_memory_example_agent_count > xmachine_memory_example_agent_MAX){
		printf("Error: Buffer size of input_example agents in state default will be exceeded moving working agents to next state in function input_example\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  example_agents_default_temp = d_example_agents;
  d_example_agents = d_example_agents_default;
  d_example_agents_default = example_agents_default_temp;
        
	//update new state agent size
	h_xmachine_memory_example_agent_default_count += h_xmachine_memory_example_agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_example_agent_default_count, &h_xmachine_memory_example_agent_default_count, sizeof(int)));	
	
	
}


 
extern void reset_example_agent_default_count()
{
    h_xmachine_memory_example_agent_default_count = 0;
}
