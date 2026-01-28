CXX=g++
CXXFLAGS=-g -Wall -Wextra -std=c++23 -march=native -O3 -fopenmp
INCLUDES=-I"/mnt/hdd_barracuda/opt/eigen/" -I"/mnt/hdd_barracuda/opt/HighFive/include/" -I"/mnt/hdd_barracuda/opt/hdf5-v1.14.0/include/" -I"~/opt/eigen/"
LDFLAGS=-L"/mnt/hdd_barracuda/opt/hdf5-v1.14.0/lib/" -L"/usr/lib/" -L"/usr/local/lib/" -lhdf5 -lgsl -lgslcblas -lm # -lhdf5_cpp
# INCLUDES=-I"/Users/nobuyukimatsumoto/opt/eigen/"
# LDFLAGS=

SRCS := $(wildcard *.cc)
OBJS := $(SRCS:%.cc=%.o)

all: $(OBJS)

%.d: %.cc
	$(CXX) $(INCLUDES) -M $< -o $@

include $(SRCS:.cc=.d)

%.o: %.cc
	$(CXX) $< -o $@ $(INCLUDES) $(LDFLAGS) $(CXXFLAGS)

# all:
# 	g++ hist_spline.cc ${INCLUDES} ${LDFLAGS}
# 	g++ get_potential.cc ${INCLUDES} ${LDFLAGS}


# all:
# # 	# g++ test.cc -I"/opt/eigen-3.4.0/" -std=c++17 -O3 -o a.out
# # 	# g++ excursion.cc -I"/opt/eigen-3.4.0/" -std=c++17 -O3 -march=native -o b.out
# # 	# g++ rmhmc.cc -I"/opt/eigen-3.4.0/" -std=c++17 -O3 -march=native -o a.out
# 	g++ hmc2d.cc -I"/opt/eigen-3.4.0/" -O3 -fopenmp -march=native -std=c++17 -DEIGEN_DONT_PARALLELIZE -o a.out # -DEIGEN_DONT_PARALLELIZE -pg

# # test:
# # 	g++ test_kernel.cc -I"/opt/eigen-3.4.0/" -O2 -fopenmp -std=c++17 -DEIGEN_DONT_PARALLELIZE -o a.out # -DEIGEN_DONT_PARALLELIZE -pg 
