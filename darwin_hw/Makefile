
CXX = g++
#INCLUDE_DIRS = -I../../Linux/include -I../../Framework/include
INCLUDE_DIRS = -I../../Linux/include -I../../Framework/include -I../../SDK/phasespace/include #-I/usr/include/boost -I/usr/include/eigen3
CXXFLAGS += -DLINUX -O3 -flto -Wall -std=c++11 $(INCLUDE_DIRS)
#CXXFLAGS += -DLINUX -g -Wall -fmessage-length=0 -std=c++11 $(INCLUDE_DIRS)
LIBS += -lpthread -lrt -L/usr/local/lib/ -lowlsock -lphidget21 -llabjackusb

all: darwin.a test_darwin test_contacts

SUPPORT_LIBS = u6.o ftconfig.o ftrt.o dom.o expatls.o node.o stack.o xmlparse.o xmlrole.o xmltok.o

darwin.a:
	make -C ../../Linux/build

test_darwin: ./test_darwin.o ../../Linux/lib/darwin.a
	$(CXX) $(CXXFLAGS) -o test_darwin ./test_darwin.o ../../Linux/lib/darwin.a $(LIBS)

test_contacts: ./test_contacts.o contacts.h ../../Linux/lib/darwin.a $(SUPPORT_LIBS)
	$(CXX) $(CXXFLAGS) -o test_contacts ./test_contacts.o $(SUPPORT_LIBS) ../../Linux/lib/darwin.a $(LIBS)

darwin_sim: ./darwin_sim.o ../../Linux/lib/darwin.a
	$(CXX) $(CXXFLAGS) -o darwin_sim ./darwin_sim.o ../../Linux/lib/darwin.a $(LIBS) -lmujoco

clean:
	rm -f *.a *.o $(TARGET) core *~ *.so *.lo

libclean:
	make -C ../../Linux/build clean


u6.o: labjack/u6.c labjack/u6.h
	$(CXX) $(CXXFLAGS) -c labjack/u6.c
ftconfig.o: atidaq/ftconfig.c atidaq/ftconfig.h
	$(CXX) $(CXXFLAGS) -fpermissive -c atidaq/ftconfig.c -I./atidaq
ftrt.o: atidaq/ftrt.c atidaq/ftrt.h  atidaq/ftsharedrt.h
	$(CXX) $(CXXFLAGS) -fpermissive -c atidaq/ftrt.c -I./atidaq
dom.o: atidaq/dom.c
	$(CXX) $(CXXFLAGS) -fpermissive -c atidaq/dom.c
expatls.o: atidaq/expatls.c
	$(CXX) $(CXXFLAGS) -fpermissive -c atidaq/expatls.c
node.o: atidaq/node.c
	$(CXX) $(CXXFLAGS) -fpermissive -c atidaq/node.c
stack.o: atidaq/stack.c
	$(CXX) $(CXXFLAGS) -fpermissive -c atidaq/stack.c
xmlparse.o: atidaq/xmlparse.c
	$(CXX) $(CXXFLAGS) -fpermissive -c atidaq/xmlparse.c
xmlrole.o: atidaq/xmlrole.c
	$(CXX) $(CXXFLAGS) -fpermissive -c atidaq/xmlrole.c
xmltok.o: atidaq/xmltok.c
	$(CXX) $(CXXFLAGS) -fpermissive -c atidaq/xmltok.c
distclean: clean libclean

# useful to make a backup "make tgz"
tgz: clean
	mkdir -p backups
	tar czvf ./backups/dxl_monitor_`date +"%Y_%m_%d_%H.%M.%S"`.tgz --exclude backups *
