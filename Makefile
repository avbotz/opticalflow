all: kek

kek:
	g++ -ggdb `pkg-config --cflags --libs opencv` ./kek.cpp -o kek

clean:
	rm -f kek
