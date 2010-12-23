CC=g++
#CFLAGS=-c -Wall -I /usr/local/include/boost-1_37/ -g
CFLAGS=-c -Wall -I /usr/include/boost/ -g
#LDFLAGS=-L /usr/lib -l boost_iostreams-gcc43-mt -l boost_filesystem-gcc43-mt
LDFLAGS=-L /usr/lib -lboost_iostreams -lboost_filesystem
SOURCES=memory_map.cpp test.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=memory_map

all: $(SOURCES) $(EXECUTABLE)
	

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -fr *.o *~ $(EXECUTABLE)
