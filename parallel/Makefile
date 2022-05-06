all: parallel parallel_stitch

CFLAGS = `pkg-config --cflags opencv4`
LIBS = `pkg-config --libs opencv4` 

parallel: parallel_stitching.cpp
	g++ -fopenmp $(CFLAGS) -std=c++11 parallel_stitching.cpp -o parallel $(LIBS) $(CFLAGS)

parallel_stitch: parallel_stitch.cpp
	g++ -fopenmp $(CFLAGS) -std=c++11 parallel_stitch.cpp -o parallel_stitch $(LIBS) $(CFLAGS)