# for GameOfLife_ISF.cpp
EXEC = ISF
CC = g++
CFLAGS = -c -Wall

$(EXEC) :GameOfLife1_ISF.o
	$(CC) -o $(EXEC) GameOfLife1_ISF.o

GameOfLife1_ISF.o :GameOfLife1_ISF.h GameOfLife1_ISF.cpp
	$(CC) $(CFLAGS) GameOfLife1_ISF.cpp

clean:
	rm *.o
