CXX = g++-4.9
CXXFLAGS = -std=c++14 -W -Wall
TARGET = bin/project
SRC_FILES = $(wildcard src/*.cpp)
OBJ_FILES = $(patsubst src/%.cpp, obj/%.o, $(SRC_FILES))

.PHONY: all clean run

all: $(TARGET)

clean:
	rm -f bin/*
	rm -f obj/*

run: $(TARGET)
	$(TARGET)

obj/%.o: src/%.cpp
	@mkdir -p obj
	$(CXX) -c $^ -o $@ $(CXXFLAGS)

$(TARGET): $(OBJ_FILES)
	@mkdir -p bin
	$(CXX) -o $@ $^ $(CXXFLAGS)
