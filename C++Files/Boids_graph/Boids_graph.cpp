// Boids_PVT_ISF.cpp

#pragma warning(disable : 4996)

#include <stdlib.h>
#include <stdio.h>
#include "Header.h"

using namespace std;

int produceFile(int numOfAgents) {
	FILE* graph;
	FILE* xml;

	if ((graph = fopen("graph.txt", "w")) == NULL) {
		printf("graph.txt is null\n");
		return 0;
	}
	else if ((xml = fopen("200.xml", "r")) == NULL) {
		printf("200.xml is null\n");
		return 0;
	}

	else {
		fprintf(graph, "turtles-own [id]\n\n");
		fprintf(graph, "to setup\n");
		fprintf(graph, "\tclear-all\n");
		fprintf(graph, "\tcreate-turtles %d [\n", numOfAgents);
		fprintf(graph, "\t\tset id zero\n");
		fprintf(graph, "\t\tset zero zero + 1\n");
		fprintf(graph, "\t]\n");
		fprintf(graph, "\task turtles [\n");
		float xP, yP, zP, xV, yV, zV;
		double pointX, pointY;
		int id;
		fscanf(xml, "<states>\n");
		for (int i = 0; i < numOfAgents; i++) {
			if (fscanf(xml, "\t<xagent>\n") != 0) {
				printf("Can't read xagent\n");
				return 0;
			}
			if (fscanf(xml, "\t<name>Boid</name>\n") != 0) {
				printf("Can't read name\n");
				return 0;
			}
			if (fscanf(xml, "\t<id>%d</id>\n", &id) != 1) {
				printf("Can't read id\n");
				return 0;
			}
			fprintf(graph, "\t\tif id = %d [\n", id);
			if (fscanf(xml, "<position>%f, %f, %f</position>\n", &xP, &yP, &zP) != 3) {
				printf("Can't read position\n");
				return 0;
			}
			if (fscanf(xml, "\t<velocity>%f, %f, %f</velocity>\n", &xV, &yV, &zV) != 3) {
				printf("Can't read velocity\n");
				return 0;
			}
			if (fscanf(xml, "\t</xagent>\n") != 0) {
				printf("Can't read /xagent\n");
			}
			fprintf(graph, "\t\t\tsetxy %f %f\n", xP, yP);
			pointX = (double)xP + (double)xV;
			pointY = (double)yP + (double)yV;
			fprintf(graph, "\t\t\tfacexy %f %f\n", pointX, pointY);
			fprintf(graph, "\t\t]\n");
		}
		fprintf(graph, "\t]\n");
		fprintf(graph, "end\n");
	}
	fclose(graph);
	fclose(xml);
	return 1;
}

int main(int argc, char** argv) {
	printf("Enter number of agents: ");
	int numOfAgents;
	if(scanf("%d", &numOfAgents) != 1) return 0;

	int ok = produceFile(numOfAgents);
	if (ok == 0) {
		printf("Error\n");
		return 0;
	}
	else {
		printf("File created\n");
		return 1;
	}
}
