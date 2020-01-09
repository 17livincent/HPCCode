# Makefile for Boids_PartitioningVecTypes

EXEC = ISF
CC = g++
CFLAGS = -c -Wall

$(EXEC) :Boids_PVT_ISF.o
	$(CC) -o $(EXEC) Boids_PVT_ISF.o

GameOfLife1_ISF.o :Boids_PVT_ISF.h Boids_PVT_ISF.cpp
	$(CC) $(CFLAGS) Boids_PVT_ISF.cpp

clean:
	rm *.o
