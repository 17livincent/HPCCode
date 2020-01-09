/*
 * Copyright 2011 University of Sheffield.
 * Author: Dr Paul Richmond 
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

#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include <header.h>

#define NUM_COLOURS 9

extern "C" float* get_numberOfTurtles();
extern "C" void set_numberOfTurtles(float* h_numberOfTurtles);
extern "C" float* get_energyFromGrass();
extern "C" float* get_birthEnergy();

__FLAME_GPU_FUNC__ int moveTurtles(xmachine_memory_turtle turtle, RNG_rand48* rand48) {
  float turn = rand<CONTINUOUS>(rand48) % (2 * 3.14159);
  turtle->vX = cosf(turn);
  turtle->vP = sinf(turn);
  // foward 1 for each turtle
  float vMag = sqrtf(vX * vX + vP * vP);
  turtle->vX /= vMag;
  turtle->vY /= vMag;
  // update position
  turtle->pX += turtle->vX;
  turtle->pY += turtle->vY;
  // update energy
  turtle->energy -= 1;

  return 0;
}
  
__FLAME_GPU_FUNC__ int eatGrass(xmachine_memory_turtle turtle, xmachine_message_location grassLocation) {
  // find grass agent that the turtle is on
  int x = (int)turtle->pX;
  int y = (int)turtle->pY;
  grassLocation = get_first_location<DISCRETE_2D>(location_messages, (int)turtle->pX, (int)turtle->pY);
  if(grassLocation->green == 1) {
    // set green to 0

    // set energy
    turtle->energy += get_energyFromGrass();
  }
  return 0;
}

__FLAME_GPU_FUNC__ int reproduce(xmachine_memory_turtle turtle) {
  if(turtle->energy > 50) {
    turtle->energy -= get_birthEnergy();
    int newID = get_numberOfTurtles() + 1;
    set_numberOfTurtles(&newID);
    add_turtle_agent(turtle_agents, get_numberOfTurtles(), get_birthEnergy(), turtle->pX, turtle->pY, 0, 0);
  }
  return 0;
}

__FLAME_GPU_FUNC__ int checkDeath(xmachine_memory_turtle turtle) {
  if(turtle->energy <= 0) {
    return 1; // agent dies
  }
  else {
    return 0;
  }
}

__FLAME_GPU_FUNC__ int regrowGrass(xmachine_memory_grass grass) {
  if(rnd<DISCRETE>(rand48) % 100 <= 3) {
    grass->green = 1;
    // add location message
    add_location_message(location_messages, grass->id, grass->green, grass->pX, grass->pY);
  }
  
  return 0;
}

#endif //_FLAMEGPU_FUNCTIONS
