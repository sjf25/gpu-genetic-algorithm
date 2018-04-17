#TODO: get the makefile to recompile when header changes

CXX = g++-4.9
CXXFLAGS = -std=c++14 -W -Wall -O3
LFLAGS = -lboost_program_options
TARGET = bin/project
SRC_FILES = $(wildcard src/*.cpp)
OBJ_FILES = $(patsubst src/%.cpp, obj/%.o, $(SRC_FILES))

.PHONY: all clean run

all: $(TARGET)

clean:
	rm -f bin/*
	rm -f obj/*

run: $(TARGET)
	$(TARGET) --input=$(INPUT) ; notify-send "Done Running" -t 3000

obj/%.o: src/%.cpp
	@mkdir -p obj
	$(CXX) -c $^ -o $@ $(CXXFLAGS)

$(TARGET): $(OBJ_FILES)
	@mkdir -p bin
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LFLAGS)
