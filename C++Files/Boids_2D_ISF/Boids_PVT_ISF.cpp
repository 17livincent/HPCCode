// Boids_PVT_ISF.cpp

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "Boids_PVT_ISF.h"

using namespace std;

int produceFile(int numOfAgents) {
  int xPBound = 69;
  int yPBound = 69;
  // int zBound = 0.5 * 10;	// zBound * 10
  int xVBound = 4;
  int yVBound = 4;
  srand(time(NULL));
  
  FILE* ISF;
  ISF = fopen("0.xml", "w");
  if(ISF == NULL) {
    return 0;
  }
  else {
    float xP, yP, zP, xV, yV, zV;
    // print stuff
    fprintf(ISF, "<!-- %d agents -->\n", numOfAgents);
    fprintf(ISF, "<states>\n");
    fprintf(ISF, "\t<itno>0</itno>\n");
    fprintf(ISF, "\t<environment></environment>\n");
    for(int i = 0; i < numOfAgents; i++) {
      fprintf(ISF, "\t<xagent>\n");
      fprintf(ISF, "\t\t<name>Boid</name>\n");
      fprintf(ISF, "\t\t<id>%d</id>\n", i);
      xP = (rand() % xPBound);
      yP = (rand() % yPBound);
      zP = 0;
      xV = (rand() % xVBound) + 0.1;
      yV = (rand() % yVBound) + 0.1;
      zV = 0;
      fprintf(ISF, "\t\t<position>%f, %f, %f</position>\n", xP, yP, zP);
      fprintf(ISF, "\t\t<velocity>%f, %f, %f</velocity>\n", xV, yV, zV);
      fprintf(ISF, "\t</xagent>\n");
    }
    fprintf(ISF, "</states>\n");
  }
} 

int main(int argc, char** argv) {
  printf("Enter number of agents: ");
  int numOfAgents;
  scanf("%d", &numOfAgents);
  
  int ok = produceFile(numOfAgents);
  if(ok == 0) {
    printf("File not read/created\n");
    return 0;
  }
  else {
    printf("File created\n");
    return 0;
  }
}

/*
  <states>
  <itno>0</itno>
  <environment></environment>
  <xagent>
  <name>Boid</name>
  <id>0</id>
  <position>-0.306696, 0.063585, -0.498749</position>
  <velocity>-0.020127, 0.085009, 0.308741</velocity>
  </xagent>
  </states>
  
*/
