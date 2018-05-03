#TODO: get the makefile to recompile when header changes

CXX = nvcc
#CXXFLAGS = -std=c++11 --compiler-options -W --compiler-options -Wall -g -arch=sm_61 -dc
CXXFLAGS = -std=c++11 --compiler-options -W --compiler-options -Wall -O3 -Xptxas -O3 -arch=sm_61 -dc
LFLAGS = -lboost_program_options -arch=sm_61 -std=c++11
TARGET = bin/project
CPP_SRC_FILES = $(wildcard src/*.cpp)
CU_SRC_FILES = $(wildcard src/*.cu)
OBJ_FILES = $(patsubst src/%.cpp, obj/%.o, $(CPP_SRC_FILES))
OBJ_FILES += $(patsubst src/%.cu, obj/%.o, $(CU_SRC_FILES))

.PHONY: all clean run profile

all: $(TARGET)

clean:
	rm -f bin/*
	rm -f obj/*

run: $(TARGET)
#	$(TARGET) --input=$(INPUT) ; notify-send "Done Running" -t 3000
	$(TARGET) --input=$(INPUT) --pop-size=$(POP_SIZE) --func-evals=$(FUNC_EVALS) --run-mode=$(RUN_MODE) \
		--problem=$(PROBLEM) --num-runs=$(NUM_RUNS) --maxone-len=$(MAXONE_LEN)

profile: $(TARGET)
#	nvprof --print-gpu-trace $(TARGET) --input=$(INPUT) --pop-size=$(POP_SIZE) --func-evals=$(FUNC_EVALS)
	nvprof $(TARGET) --input=$(INPUT) --pop-size=$(POP_SIZE) --func-evals=$(FUNC_EVALS)

obj/%.o: src/%.cpp
	@mkdir -p obj
	$(CXX) -c $^ -o $@ $(CXXFLAGS)

obj/genetic_cuda.o: src/genetic_cuda.cu
	@mkdir -p obj
	$(CXX) -c $^ -o $@ $(CXXFLAGS)

obj/%.o: src/%.cu
	@mkdir -p obj
	$(CXX) -c $^ -o $@ $(CXXFLAGS)

$(TARGET): $(OBJ_FILES)
	echo $^
	@mkdir -p bin
	$(CXX) -o $@ $^ $(LFLAGS)
