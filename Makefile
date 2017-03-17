###########################################
# 	Copyright (C) 2017
# 	Project : Matrix Convolution
# 	Author(s)  : Hemant Nigam
#       Description : Makefile 
###########################################

CC=nvcc
ARCH=sm_53
DEBUG_FLAGS=-g -lineinfo
CFLAGS= -arch $(ARCH) $(DEBUG_FLAGS)
LDFLAGS=-lopencv_core -lopencv_highgui -lopencv_imgproc
SRC=main.cu
SRC_CPU=cpukernel.cpp
TARGET=matconv
OBJ=$(SRC:.cu=.o) $(SRC_CPU:.cpp=.o) 

.SUFFIXES: .cu .cpp .o

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) $(OBJ) -o $(TARGET)
.cu.o:
	$(CC) $(CFLAGS) $< -c -o $@
.cpp.o:
	$(CC) $(CFLAGS) $< -c -o $@
clean:
	rm -rf *.o $(TARGET)

