#pragma once

//#include <thread>
#include <mutex>
#include "atidaq/ftconfig.h"
#include "labjack/u6.h"

class ContactSensors {

  public:
    ContactSensors() : resIndex(1), sendSize(56), recvSize(46){
      // Calibration of ATI-NANO 25
      cal_r=createCalibration("./FT18087.cal", 1); // right foot
      cal_l=createCalibration("./FT18086.cal", 1); // left foot
      if (cal_r==NULL) {
        printf("\nSpecified right foot calibration could not be loaded.\n");
      }
      if (cal_l==NULL) {
        printf("\nSpecified left foot calibration could not be loaded.\n");
      }
      this->m_Initialized = false;
      // Set force units.
      // This step is optional; by default, the units are inherited from the calibration file.
      short sts;
      sts=SetForceUnits(cal_r,"N");
      switch (sts) {
        case 0: break;	// successful completion
        case 1: printf("Invalid Calibration struct");
        case 2: printf("Invalid force units");
        default: printf("Unknown error");
      }
      sts=SetForceUnits(cal_l,"N");
      switch (sts) {
        case 0: break;	// successful completion
        case 1: printf("Invalid Calibration struct");
        case 2: printf("Invalid force units");
        default: printf("Unknown error");
      }

      // Set torque units.
      // This step is optional; by default, the units are inherited from the calibration file.
      sts=SetTorqueUnits(cal_r,"N-m");
      switch (sts) {
        case 0: break;	// successful completion
        case 1: printf("Invalid Calibration struct");
        case 2: printf("Invalid torque units");
        default: printf("Unknown error");
      }
      sts=SetTorqueUnits(cal_l,"N-m");
      switch (sts) {
        case 0: break;	// successful completion
        case 1: printf("Invalid Calibration struct");
        case 2: printf("Invalid torque units");
        default: printf("Unknown error");
      }
      // Set tool transform.
      // This line is only required if you want to move or rotate the sensor's coordinate system.
      // This example tool transform translates the coordinate system 20 mm along the Z-axis 
      // and rotates it 45 degrees about the X-axis.
      //sts=SetToolTransform(cal,SampleTT,"mm","degrees");
      //switch (sts) {
      //	case 0: break;	// successful completion
      //	case 1: printf("Invalid Calibration struct");
      //	case 2: printf("Invalid distance units");
      //	case 3: printf("Invalid angle units");
      //	default: printf("Unknown error");
      //}

      // may need different bias for different feets?
      // bias vector need only be of length 6
      //float SampleBias[7]={0.2651,-0.0177,-0.0384,-0.0427,-0.1891,0.1373,-3.2423};
      //Bias(cal,SampleBias);
      // Calibration of ATI-NANO 25


      // Initialize Labjack u6
      long error=0;
      int idx = 7;
      if( (hDevice = openUSBConnection(-1)) == NULL ) { // should only have one u6...
        printf("Couldn't open U6. Please connect one and try again.\n");
        goto close;
      }

      if( getCalibrationInfo(hDevice, &caliInfo) < 0 )
        goto close;

      // command
      sendBuff = new uint8[sendSize];
      sendBuff[0] = 0;
      sendBuff[1] = (uint8)(0xF8);  //Command byte
      sendBuff[2] = 25;             //Number of data words (.5 word for echo, 10.5 words for IOTypes)
      sendBuff[3] = (uint8)(0x00);  //Extended command number
      sendBuff[6] = 0;  //Echo, for order keeping. ignore for now

      // data requests, 12 channels of data from contact sensors
      for (int input=0; input<12; input++) {
        sendBuff[idx++] = 2;           //IOType is AIN24; Analog Input
        sendBuff[idx++] = input;       //Positive channel, single read
        sendBuff[idx++] = resIndex + (0<<4);  //ResolutionIndex(Bits 0-3) = 1, GainIndex(Bits 4-7) = 0 (+-10V)
        sendBuff[idx++] = 0 + 0*128;  //SettlingFactor(Bits 0-2) = 0 (5 microseconds), Differential(Bit 7) = 0
      }

      // idx should be 55
      sendBuff[idx++] = 0;    //Padding byte; 24 words + 0.5 words (echo) + 0.5 padding

      extendedChecksum(sendBuff, idx);

      // recieve buffer
      recvBuff = new uint8[recvSize];


      this->m_Initialized = true;

      return;
close:
      if( error > 0 )
        printf("Received an error code of %ld\n", error);
      closeUSBConnection(hDevice);

      destroyCalibration(cal_r);
      destroyCalibration(cal_l);
      return;
    }

    bool is_running() {
      return this->m_Initialized;
    }

    bool start_streaming() {
      return false;
    }

    bool stop_streaming() {
      return false;
    }

    bool getData(double * r, double * l) {
      if (this->m_Initialized) {
        // read values from DAQ with low level packets
        unsigned long sendChars, recChars;
        uint16 checksumTotal;

        //Sending command to U6
        if( (sendChars = LJUSB_Write(hDevice, sendBuff, sendSize)) < sendSize ) {
          if(sendChars == 0) printf("Feedback loop error : write failed\n");
          else printf("Feedback loop error : did not write all of the buffer\n");
          return false;
        }

        //Reading response from U6
        if( (recChars = LJUSB_Read(hDevice, recvBuff, recvSize)) < recvSize ) {
          if( recChars == 0 ) {
            printf("Feedback loop error : read failed\n");
            return false;
          }
          else printf("Feedback loop error : did not read all of the expected buffer\n");
        }

        if( recChars < 10 ) {
          printf("Feedback loop error : response is not large enough\n");
          return false;
        }

        checksumTotal = extendedChecksum16(recvBuff, recChars);

        if( (uint8)((checksumTotal / 256 ) & 0xff) != recvBuff[5] ) {
          printf("Feedback loop error : read buffer has bad checksum16(MSB)\n");
          return false;
        }

        if( (uint8)(checksumTotal & 0xff) != recvBuff[4] ) {
          printf("Feedback loop error : read buffer has bad checksum16(LBS)\n");
          return false;
        }

        if( extendedChecksum8(recvBuff) != recvBuff[0] ) {
          printf("Feedback loop error : read buffer has bad checksum8\n");
          return false;
        }

        if( recvBuff[1] != (uint8)(0xF8) ||  recvBuff[3] != (uint8)(0x00) ) {
          printf("Feedback loop error : read buffer has wrong command bytes \n");
          return false;
        }

        if( recvBuff[6] != 0 ) {
          printf("Feedback loop error : received errorcode %d for frame %d ", recvBuff[6], recvBuff[7]);
          switch( recvBuff[7] ) {
            case 1: printf("(AIN0(SE))\n"); break;
            case 2: printf("(AIN1(SE))\n"); break;
            case 3: printf("(AIN2(SE))\n"); break;
            case 4: printf("(AIN3(SE))\n"); break;
            case 5: printf("(AIN4(SE))\n"); break;
            case 6: printf("(AIN5(SE))\n"); break;
            case 7: printf("(AIN6(SE))\n"); break;
            case 8: printf("(AIN7(SE))\n"); break;
            case 9: printf("(AIN8(SE))\n"); break;
            case 10: printf("(AIN9(SE))\n"); break;
            case 11: printf("(AIN10(SE))\n"); break;
            case 12: printf("(AIN11(SE))\n"); break;
            default: printf("(Unknown)\n"); break;
          }
          return false;
        }

        double voltage;
        int idx=9;
        float raw[6];
        float FT[6];            // This array will hold the resultant force/torque vector.
        for (int input=0; input<6; input++) {
          int a=idx++; int b=idx++; int c=idx++;
          getAinVoltCalibrated(&caliInfo, resIndex, 0, 1,
              recvBuff[a]+(recvBuff[b]*256)+(recvBuff[c]*65536),
              &voltage);
          raw[input] = (float)voltage;
        }

        // convert raw values to usable units / calibration
        ConvertToFT(cal_r,raw,FT);
        for (int i = 0; i<6; i++) {
          r[i] = (double)FT[i];
        }

        for (int input=0; input<6; input++) {
          int a=idx++; int b=idx++; int c=idx++;
          getAinVoltCalibrated(&caliInfo, resIndex, 0, 1,
              recvBuff[a]+(recvBuff[b]*256)+(recvBuff[c]*65536),
              &voltage);
          raw[input] = (float)voltage;
        }
        ConvertToFT(cal_l,raw,FT);
        for (int i = 0; i<6; i++) {
          l[i] = (double)FT[i];
        }
        return true;

        //this->mutex.lock(); // not a try; wait for newest
        //for (int i = 0; i<POSE_SIZE; i++) {
        //  p[i] = (double)pose[i];
        //}
        //for (int i = 0; i<(MARKER_COUNT * 4); i++) {
        //  m[i] = (double)marker_d[i];
        //}
        //this->mutex.unlock();
        //return true;
      }
      else {
        printf("ContactSensors not running.\n");
        return false;
      }
    }

    ~ContactSensors() {
      destroyCalibration(cal_r);
      destroyCalibration(cal_l);

      closeUSBConnection(hDevice);

      if (sendBuff) delete[] sendBuff;
      if (recvBuff) delete[] recvBuff;

      if (this->m_TrackerRunning) {

        printf("Stopping ContactSensors Module\n");
        this->m_FinishTracking = true;
        // wait for the thread to end
        //m_Thread.join();
        this->m_Initialized = false;
        this->m_FinishTracking = false;
        this->m_TrackerRunning = false;
      }
    }

  private:
    bool m_Initialized;
    bool m_TrackerRunning;
    bool m_FinishTracking;

    // ATI NANO 25
    Calibration *cal_r;		// struct containing calibration information
    Calibration *cal_l;		// struct containing calibration information

    // Labjack U6
    HANDLE hDevice;
    u6CalibrationInfo caliInfo;

    int resIndex;
    unsigned long sendSize;
    unsigned long recvSize;
    uint8 *sendBuff;
    uint8 *recvBuff;
};


