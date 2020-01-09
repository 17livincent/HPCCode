
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



#ifndef __HEADER
#define __HEADER

#if defined __NVCC__
   // Disable annotation on defaulted function warnings (glm 0.9.9 and CUDA 9.0 introduced this warning)
   #pragma diag_suppress esa_on_defaulted_function_ignored 
#endif

#define GLM_FORCE_NO_CTOR_INIT
#include <glm/glm.hpp>

/* General standard definitions */
//Threads per block (agents per block)
#define THREADS_PER_TILE 64
//Definition for any agent function or helper function
#define __FLAME_GPU_FUNC__ __device__
//Definition for a function used to initialise environment variables
#define __FLAME_GPU_INIT_FUNC__
#define __FLAME_GPU_STEP_FUNC__
#define __FLAME_GPU_EXIT_FUNC__
#define __FLAME_GPU_HOST_FUNC__ __host__

#define USE_CUDA_STREAMS
#define FAST_ATOMIC_SORTING

// FLAME GPU Version Macros.
#define FLAME_GPU_MAJOR_VERSION 1
#define FLAME_GPU_MINOR_VERSION 5
#define FLAME_GPU_PATCH_VERSION 0

typedef unsigned int uint;

//FLAME GPU vector types float, (i)nteger, (u)nsigned integer, (d)ouble
typedef glm::vec2 fvec2;
typedef glm::vec3 fvec3;
typedef glm::vec4 fvec4;
typedef glm::ivec2 ivec2;
typedef glm::ivec3 ivec3;
typedef glm::ivec4 ivec4;
typedef glm::uvec2 uvec2;
typedef glm::uvec3 uvec3;
typedef glm::uvec4 uvec4;
typedef glm::dvec2 dvec2;
typedef glm::dvec3 dvec3;
typedef glm::dvec4 dvec4;

	

/* Agent population size definitions must be a multiple of THREADS_PER_TILE (default 64) */
//Maximum buffer size (largest agent buffer size)
#define buffer_size_MAX 1024

//Maximum population size of xmachine_memory_example_agent
#define xmachine_memory_example_agent_MAX 1024


  
  
/* Message population size definitions */
//Maximum population size of xmachine_mmessage_example_message
#define xmachine_message_example_message_MAX 1024


/* Define preprocessor symbols for each message to specify the type, to simplify / improve portability */

#define xmachine_message_example_message_partitioningNone

/* Spatial partitioning grid size definitions */

/* Static Graph size definitions*/
  

/* Default visualisation Colour indices */
 
#define FLAME_GPU_VISUALISATION_COLOUR_BLACK 0
#define FLAME_GPU_VISUALISATION_COLOUR_RED 1
#define FLAME_GPU_VISUALISATION_COLOUR_GREEN 2
#define FLAME_GPU_VISUALISATION_COLOUR_BLUE 3
#define FLAME_GPU_VISUALISATION_COLOUR_YELLOW 4
#define FLAME_GPU_VISUALISATION_COLOUR_CYAN 5
#define FLAME_GPU_VISUALISATION_COLOUR_MAGENTA 6
#define FLAME_GPU_VISUALISATION_COLOUR_WHITE 7
#define FLAME_GPU_VISUALISATION_COLOUR_BROWN 8

/* enum types */

/**
 * MESSAGE_OUTPUT used for all continuous messaging
 */
enum MESSAGE_OUTPUT{
	single_message,
	optional_message,
};

/**
 * AGENT_TYPE used for templates device message functions
 */
enum AGENT_TYPE{
	CONTINUOUS,
	DISCRETE_2D
};


/* Agent structures */

/** struct xmachine_memory_example_agent
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_example_agent
{
    int id;    /**< X-machine memory variable id of type int.*/
    float x;    /**< X-machine memory variable x of type float.*/
    float y;    /**< X-machine memory variable y of type float.*/
    float colour;    /**< X-machine memory variable colour of type float.*/
};



/* Message structures */

/** struct xmachine_message_example_message
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_example_message
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    float x;        /**< Message variable x of type float.*/  
    float y;        /**< Message variable y of type float.*/
};



/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_memory_example_agent_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_example_agent_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_example_agent_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_example_agent_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_memory_example_agent_MAX];    /**< X-machine memory variable list id of type int.*/
    float x [xmachine_memory_example_agent_MAX];    /**< X-machine memory variable list x of type float.*/
    float y [xmachine_memory_example_agent_MAX];    /**< X-machine memory variable list y of type float.*/
    float colour [xmachine_memory_example_agent_MAX];    /**< X-machine memory variable list colour of type float.*/
};



/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_message_example_message_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_example_message_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_example_message_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_example_message_MAX];  /**< Used during parallel prefix sum */
    
    float x [xmachine_message_example_message_MAX];    /**< Message memory variable list x of type float.*/
    float y [xmachine_message_example_message_MAX];    /**< Message memory variable list y of type float.*/
    
};



/* Spatially Partitioned Message boundary Matrices */



/* Graph structures */


/* Graph Edge Partitioned message boundary structures */


/* Graph utility functions, usable in agent functions and implemented in FLAMEGPU_Kernels */


  /* Random */
  /** struct RNG_rand48
  *	structure used to hold list seeds
  */
  struct RNG_rand48
  {
  glm::uvec2 A, C;
  glm::uvec2 seeds[buffer_size_MAX];
  };


/** getOutputDir
* Gets the output directory of the simulation. This is the same as the 0.xml input directory.
* @return a const char pointer to string denoting the output directory
*/
const char* getOutputDir();

  /* Random Functions (usable in agent functions) implemented in FLAMEGPU_Kernels */

  /**
  * Templated random function using a DISCRETE_2D template calculates the agent index using a 2D block
  * which requires extra processing but will work for CONTINUOUS agents. Using a CONTINUOUS template will
  * not work for DISCRETE_2D agent.
  * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
  * @return			returns a random float value
  */
  template <int AGENT_TYPE> __FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);
/**
 * Non templated random function calls the templated version with DISCRETE_2D which will work in either case
 * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
 * @return			returns a random float value
 */
__FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);

/* Agent function prototypes */

/**
 * output_example FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_example_agent. This represents a single agent instance and can be modified directly.
 * @param example_message_messages Pointer to output message list of type xmachine_message_example_message_list. Must be passed as an argument to the add_example_message_message function ??.
 */
__FLAME_GPU_FUNC__ int output_example(xmachine_memory_example_agent* agent, xmachine_message_example_message_list* example_message_messages);

/**
 * input_example FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_example_agent. This represents a single agent instance and can be modified directly.
 * @param example_message_messages  example_message_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_example_message_message and get_next_example_message_message functions.
 */
__FLAME_GPU_FUNC__ int input_example(xmachine_memory_example_agent* agent, xmachine_message_example_message_list* example_message_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) example_message message implemented in FLAMEGPU_Kernels */

/** add_example_message_message
 * Function for all types of message partitioning
 * Adds a new example_message agent to the xmachine_memory_example_message_list list using a linear mapping
 * @param agents	xmachine_memory_example_message_list agent list
 * @param x	message variable of type float
 * @param y	message variable of type float
 */
 
 __FLAME_GPU_FUNC__ void add_example_message_message(xmachine_message_example_message_list* example_message_messages, float x, float y);
 
/** get_first_example_message_message
 * Get first message function for non partitioned (brute force) messages
 * @param example_message_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_example_message * get_first_example_message_message(xmachine_message_example_message_list* example_message_messages);

/** get_next_example_message_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param example_message_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_example_message * get_next_example_message_message(xmachine_message_example_message* current, xmachine_message_example_message_list* example_message_messages);

  
/* Agent Function Prototypes implemented in FLAMEGPU_Kernels */

/** add_example_agent_agent
 * Adds a new continuous valued example_agent agent to the xmachine_memory_example_agent_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_example_agent_list agent list
 * @param id	agent agent variable of type int
 * @param x	agent agent variable of type float
 * @param y	agent agent variable of type float
 * @param colour	agent agent variable of type float
 */
__FLAME_GPU_FUNC__ void add_example_agent_agent(xmachine_memory_example_agent_list* agents, int id, float x, float y, float colour);


/* Graph loading function prototypes implemented in io.cu */


  
/* Simulation function prototypes implemented in simulation.cu */
/** getIterationNumber
 *  Get the iteration number (host)
 */
extern unsigned int getIterationNumber();

/** initialise
 * Initialise the simulation. Allocated host and device memory. Reads the initial agent configuration from XML.
 * @param input        XML file path for agent initial configuration
 */
extern void initialise(char * input);

/** cleanup
 * Function cleans up any memory allocations on the host and device
 */
extern void cleanup();

/** singleIteration
 *	Performs a single iteration of the simulation. I.e. performs each agent function on each function layer in the correct order.
 */
extern void singleIteration();

/** saveIterationData
 * Reads the current agent data fromt he device and saves it to XML
 * @param	outputpath	file path to XML file used for output of agent data
 * @param	iteration_number
 * @param h_example_agents Pointer to agent list on the host
 * @param d_example_agents Pointer to agent list on the GPU device
 * @param h_xmachine_memory_example_agent_count Pointer to agent counter
 */
extern void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_example_agent_list* h_example_agents_default, xmachine_memory_example_agent_list* d_example_agents_default, int h_xmachine_memory_example_agent_default_count);


/** readInitialStates
 * Reads the current agent data from the device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_example_agents Pointer to agent list on the host
 * @param h_xmachine_memory_example_agent_count Pointer to agent counter
 */
extern void readInitialStates(char* inputpath, xmachine_memory_example_agent_list* h_example_agents, int* h_xmachine_memory_example_agent_count);


/* Return functions used by external code to get agent data from device */

    
/** get_agent_example_agent_MAX_count
 * Gets the max agent count for the example_agent agent type 
 * @return		the maximum example_agent agent count
 */
extern int get_agent_example_agent_MAX_count();



/** get_agent_example_agent_default_count
 * Gets the agent count for the example_agent agent type in state default
 * @return		the current example_agent agent count in state default
 */
extern int get_agent_example_agent_default_count();

/** reset_default_count
 * Resets the agent count of the example_agent in state default to 0. This is useful for interacting with some visualisations.
 */
extern void reset_example_agent_default_count();

/** get_device_example_agent_default_agents
 * Gets a pointer to xmachine_memory_example_agent_list on the GPU device
 * @return		a xmachine_memory_example_agent_list on the GPU device
 */
extern xmachine_memory_example_agent_list* get_device_example_agent_default_agents();

/** get_host_example_agent_default_agents
 * Gets a pointer to xmachine_memory_example_agent_list on the CPU host
 * @return		a xmachine_memory_example_agent_list on the CPU host
 */
extern xmachine_memory_example_agent_list* get_host_example_agent_default_agents();


/** sort_example_agents_default
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_example_agents_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_example_agent_list* agents));



/* Host based access of agent variables*/

/** int get_example_agent_default_variable_id(unsigned int index)
 * Gets the value of the id variable of an example_agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_example_agent_default_variable_id(unsigned int index);

/** float get_example_agent_default_variable_x(unsigned int index)
 * Gets the value of the x variable of an example_agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_example_agent_default_variable_x(unsigned int index);

/** float get_example_agent_default_variable_y(unsigned int index)
 * Gets the value of the y variable of an example_agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_example_agent_default_variable_y(unsigned int index);

/** float get_example_agent_default_variable_colour(unsigned int index)
 * Gets the value of the colour variable of an example_agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable colour
 */
__host__ float get_example_agent_default_variable_colour(unsigned int index);




/* Host based agent creation functions */

/** h_allocate_agent_example_agent
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated example_agent struct.
 */
xmachine_memory_example_agent* h_allocate_agent_example_agent();
/** h_free_agent_example_agent
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_example_agent(xmachine_memory_example_agent** agent);
/** h_allocate_agent_example_agent_array
 * Utility function to allocate an array of structs for  example_agent agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_example_agent** h_allocate_agent_example_agent_array(unsigned int count);
/** h_free_agent_example_agent_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_example_agent_array(xmachine_memory_example_agent*** agents, unsigned int count);


/** h_add_agent_example_agent_default
 * Host function to add a single agent of type example_agent to the default state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_example_agent_default instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_example_agent_default(xmachine_memory_example_agent* agent);

/** h_add_agents_example_agent_default(
 * Host function to add multiple agents of type example_agent to the default state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of example_agent agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_example_agent_default(xmachine_memory_example_agent** agents, unsigned int count);

  
  
/* Analytics functions for each varible in each state*/
typedef enum {
  REDUCTION_MAX,
  REDUCTION_MIN,
  REDUCTION_SUM
}reduction_operator;


/** int reduce_example_agent_default_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_example_agent_default_id_variable();



/** int count_example_agent_default_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_example_agent_default_id_variable(int count_value);

/** int min_example_agent_default_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_example_agent_default_id_variable();
/** int max_example_agent_default_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_example_agent_default_id_variable();

/** float reduce_example_agent_default_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_example_agent_default_x_variable();



/** float min_example_agent_default_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_example_agent_default_x_variable();
/** float max_example_agent_default_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_example_agent_default_x_variable();

/** float reduce_example_agent_default_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_example_agent_default_y_variable();



/** float min_example_agent_default_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_example_agent_default_y_variable();
/** float max_example_agent_default_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_example_agent_default_y_variable();

/** float reduce_example_agent_default_colour_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_example_agent_default_colour_variable();



/** float min_example_agent_default_colour_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_example_agent_default_colour_variable();
/** float max_example_agent_default_colour_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_example_agent_default_colour_variable();


  
/* global constant variables */


/** getMaximumBound
 * Returns the maximum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the maximum x, y and z positions of all agents
 */
glm::vec3 getMaximumBounds();

/** getMinimumBounds
 * Returns the minimum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the minimum x, y and z positions of all agents
 */
glm::vec3 getMinimumBounds();
    
    
#ifdef VISUALISATION
/** initVisualisation
 * Prototype for method which initialises the visualisation. Must be implemented in separate file
 * @param argc	the argument count from the main function used with GLUT
 * @param argv	the argument values from the main function used with GLUT
 */
extern void initVisualisation();

extern void runVisualisation();


#endif

#if defined(PROFILE)
#include "nvToolsExt.h"

#define PROFILE_WHITE   0x00eeeeee
#define PROFILE_GREEN   0x0000ff00
#define PROFILE_BLUE    0x000000ff
#define PROFILE_YELLOW  0x00ffff00
#define PROFILE_MAGENTA 0x00ff00ff
#define PROFILE_CYAN    0x0000ffff
#define PROFILE_RED     0x00ff0000
#define PROFILE_GREY    0x00999999
#define PROFILE_LILAC   0xC8A2C8

const uint32_t profile_colors[] = {
  PROFILE_WHITE,
  PROFILE_GREEN,
  PROFILE_BLUE,
  PROFILE_YELLOW,
  PROFILE_MAGENTA,
  PROFILE_CYAN,
  PROFILE_RED,
  PROFILE_GREY,
  PROFILE_LILAC
};
const int num_profile_colors = sizeof(profile_colors) / sizeof(uint32_t);

// Externed value containing colour information.
extern unsigned int g_profile_colour_id;

#define PROFILE_PUSH_RANGE(name) { \
    unsigned int color_id = g_profile_colour_id % num_profile_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = profile_colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
    g_profile_colour_id++; \
}
#define PROFILE_POP_RANGE() nvtxRangePop();

// Class for simple fire-and-forget profile ranges (ie. functions with multiple return conditions.)
class ProfileScopedRange {
public:
    ProfileScopedRange(const char * name){
      PROFILE_PUSH_RANGE(name);
    }
    ~ProfileScopedRange(){
      PROFILE_POP_RANGE();
    }
};
#define PROFILE_SCOPED_RANGE(name) ProfileScopedRange uniq_name_using_macros(name);
#else
#define PROFILE_PUSH_RANGE(name)
#define PROFILE_POP_RANGE()
#define PROFILE_SCOPED_RANGE(name)
#endif


#endif //__HEADER

