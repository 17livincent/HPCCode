# for Useless_ISF.cpp
EXEC = ISF
CC = g++
CFLAGS = -c -Wall

$(EXEC) :Useless_ISF.o
	$(CC) -o $(EXEC) Useless_ISF.o

Useless_ISF.o :Useless_ISF.h Useless_ISF.cpp
	$(CC) $(CFLAGS) Useless_ISF.cpp

clean:
	rm *.o
