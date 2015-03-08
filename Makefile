CXXFLAGS =	-O2 -g -Wall -fmessage-length=0

OBJS =		magic-mirror.o

LIBS =		-lopencv_core -lopencv_contrib -lopencv_highgui -lraspicam -lraspicam_cv -lopencv_objdetect -lopencv_imgproc

TARGET =	magic-mirror

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
