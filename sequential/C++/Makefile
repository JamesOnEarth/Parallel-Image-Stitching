all: stitch

CFLAGS = `pkg-config --cflags opencv4`
LIBS = `pkg-config --libs opencv4` 

stitch: image_stitching.cpp
	g++ $(CFLAGS) -std=c++11 image_stitching.cpp -o stitch $(LIBS) $(CFLAGS)

native: native_stitching.cpp
	g++ $(CFLAGS) -std=c++11 native_stitching.cpp -o native $(LIBS) $(CFLAGS)

git: git.cpp
	g++ $(CFLAGS) -std=c++11 git.cpp -o git $(LIBS) $(CFLAGS)

parallel: parallel_stitching.cpp
	g++ -fopenmp $(CFLAGS) -std=c++11 parallel_stitching.cpp -o parallel $(LIBS) $(CFLAGS)
