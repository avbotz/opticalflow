all: test

test:
	g++ -ggdb `pkg-config --cflags --libs opencv` ./test.cpp -o test

clean:
	rm test
