
CXX      := mpicxx
#CXXFLAGS := -O3 -std=c++17 -march=native -ffast-math
CXXFLAGS := -O3 -std=c++17 -mcpu=native -ffast-math
LDFLAGS  :=

TARGET   := kagome_rex_mpi
SRC      := kagome_rex_mpi.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o *.csv
