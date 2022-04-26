all: stitch

CFLAGS = `pkg-config --cflags --libs /usr/local/Cellar/opencv/4.5.0_5/lib/pkgconfig/opencv4.pc`

stitch: image_stitching.cpp
	g++ $(CFLAGS) -std=c++11 image_stitching.cpp -o stitch