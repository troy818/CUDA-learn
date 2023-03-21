all: program

program: erosion.o erosionCPU.o
	g++ -o program -I../common/inc -I/usr/local/cuda/include -lcuda main.cpp erosion.o erosionCPU.o -L/usr/local/cuda/lib64 -lcudart -std=c++11

# erosionFuncTemplate.o:
# 	nvcc -I../common/inc -c -arch=sm_20 erosionFuncTemplate.cu -L/usr/local/cuda/lib64 -lcudart -gencode arch=compute_20,code=sm_20

erosion.o:
	nvcc -I../common/inc -c -arch=sm_60 erosion.cu -L/usr/local/cuda/lib64 -lcudart -gencode arch=compute_60,code=sm_60

erosionCPU.o: erosionCPU.cpp
	g++ -c erosionCPU.cpp

clean: 
	rm -rf *o program