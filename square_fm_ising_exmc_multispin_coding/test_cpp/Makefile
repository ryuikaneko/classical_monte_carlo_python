#CXX=mpic++
CXX=mpicxx
#CXXFLAGS=-O3 -std=c++17 -march=native
CXXFLAGS=-O3 -std=c++17 -mcpu=native
TARGET=ising2d_msc_pt_openmpi
SRC=ising2d_msc_pt_openmpi.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
