
CXX = g++

CXXFLAGS = -O3 -DNDEBUG -W -Wall -Wno-deprecated -Wno-sign-compare -pedantic -ansi -finline-functions -std=c++0x
LINKFLAGS = -lm -lgmpxx -lgmp

SRCS = \
	Main.cpp \
	Lcm.cpp \
	Mht.cpp \
	Database.cpp \
	OccurenceDeriver.cpp \
	tree.cpp \

OBJS = $(SRCS:%.cpp=%.o)

all: mht

mht: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OTHERFLAGS) $(OBJS) $(LINKFLAGS) -o mht
debug:
	make all CXXFLAGS="-ggdb -W -Wall -pedantic"

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(OTHERFLAGS) -c $<

clean:
	rm -f mh *.o *~
