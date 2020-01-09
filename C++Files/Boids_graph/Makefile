# Makefile for Boids_PartitioningVecTypes

EXEC = getcode
CC = g++
CFLAGS = -c -Wall

$(EXEC) :Boids_graph.o
	$(CC) -o $(EXEC) Boids_graph.o

GameOfLife1_ISF.o :Boids_graph.h Boids_graph.cpp
	$(CC) $(CFLAGS) Boids_graph.cpp

clean:
	rm *.o
