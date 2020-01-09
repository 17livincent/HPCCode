/*
 * Used for FlameGPU example GameOfLife1
 * Creates an initial states XML file for the cellular automata model GameOfLife1
 * Enter integer values that are the number of cells on each axis 
 * Enter 25 and 25 for cells with coordinates (0,0) to (24,24)
 */

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include "GameOfLife1_ISF.h"

using namespace std;

int produceFile(int xHeight, int yHeight) {
  srand(time(NULL));
  int randomState;    // 10% chance that cell will be 1 (alive)
  // open XML initial states file
  FILE* ISF;
  ISF = fopen("0.xml", "w");
  if(ISF == NULL) return 0;
  fprintf(ISF, "<!-- %d x %d cells -->\n", xHeight, yHeight);
  fprintf(ISF, "<states>\n");
  fprintf(ISF, "\t<itno>0</itno>\n");
  fprintf(ISF, "\t<environment></environment>\n");
  // loop for 0 to xHeight - 1 by yHeight - 1
  for(int i = 0; i < xHeight; i++) {
    for(int j = 0; j < yHeight; j++) {
      randomState = rand() % 10 + 1;
      if(randomState == 1) /* do nothing */;
      else randomState = 0;
      fprintf(ISF, "\t<xagent>\n");
      fprintf(ISF, "\t\t<name>cell</name>\n");
      fprintf(ISF, "\t\t<state>%d</state>\n", randomState);
      fprintf(ISF, "\t\t<x>%d</x><y>%d</y>\n", i, j);
      fprintf(ISF, "\t</xagent>\n");
    }
  }
  fprintf(ISF, "</states>\n");
  fclose(ISF);
  return 1;
}

int main(int argc, char** argv) {
  printf("Creating an XML initial states file for GameOfLife1:\n");
  // get xHeight and yHeight
  printf("Enter length of x axis: ");
  int xHeight = 0;
  scanf("%d", &xHeight);
  printf("\nEnter length of y axis: ");
  int yHeight = 0;
  scanf("%d", &yHeight);
  // call produceFile
  int successful;
  successful = produceFile(xHeight, yHeight);
  if(successful == 1) {
    printf("\nFile created\n");
  }
  else {
    printf("\nUnsuccessful\n");
  }
  
  return 0;
}

