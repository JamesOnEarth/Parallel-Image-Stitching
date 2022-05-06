all: stitch

CFLAGS = `pkg-config --cflags opencv4`
LIBS = `pkg-config --libs opencv4` 

stitch: image_stitching.cpp
	g++ $(CFLAGS) -std=c++11 image_stitching.cpp -o stitch $(LIBS) $(CFLAGS)