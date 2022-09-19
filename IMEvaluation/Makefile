all: evaluate

evaluate: src/*.cpp src/*.h
	g++ -DDISCRETE -g src/main.cpp -Wall -std=c++17 -O3 src/sfmt/SFMT.c  -o evaluate

clean:
	rm -f *.o *~ evaluate
