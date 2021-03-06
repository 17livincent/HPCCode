// Useless_ISF.cpp

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include "Useless_ISF.h"

using namespace std;

int produceFile(int numOfAgents) {
  srand(time(NULL));
  float randomVar;
  // open XML initial states file
  FILE* ISF;
  ISF = fopen("0.xml", "w");
  if(ISF == NULL) return 0;
  fprintf(ISF, "<!-- %d turtles -->\n", numOfAgents);
  fprintf(ISF, "<states>\n");
  fprintf(ISF, "\t<itno>0</itno>\n");
  fprintf(ISF, "\t<environment></environment>\n");
  // loop for numOfAgents
  for(int i = 1; i <= numOfAgents; i++) {
    randomVar = rand() % 500000;
    randomVar /= 100000;
    fprintf(ISF, "\t<xagent>\n");
    fprintf(ISF, "\t\t<name>turtle</name>\n");
    fprintf(ISF, "\t\t<id>%d</id>\n", i);
    fprintf(ISF, "\t\t<var>%f</var>\n", randomVar);
    fprintf(ISF, "\t</xagent>\n");
  }
  fprintf(ISF, "</states>\n");
  fclose(ISF);
  return 1;
}

int main(int argc, char** argv) {
  printf("Creating an XML initial states file for Useless:\n");
  // get numOfAgents
  int numOfAgents;
  printf("Enter number of agents: ");
  scanf("%d", &numOfAgents);
  // call produceFile
  int successful;
  successful = produceFile(numOfAgents);
  if(successful == 1) {
    printf("\nFile created\n");
  }
  else {
    printf("\nUnsuccessful\n");
  }
  return 0;
}
