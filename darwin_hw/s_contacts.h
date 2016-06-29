#pragma once

//#include <thread>
//#include <mutex>

#include "atidaq/ftconfig.h"
#include "labjack/u6.h"

class ContactSensors {

  public:
    ContactSensors() : resIndex(1), sendSize(56), recvSize(46),
    NumChannels(12), SamplesPerPacket(25) {
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
      if (ConfigIO_example(hDevice) != 0)
        goto close;

      //Stopping any previous streams
      StreamStop(hDevice);

      if (StreamConfig_example(hDevice) != 0)
        goto close;

      if (StreamStart(hDevice) != 0)
        goto close;

      //StreamData_example(hDevice, &caliInfo);
      //StreamStop(hDevice);


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
      static int packetCounter = 0;
      static int recBuffSize = 14 + SamplesPerPacket * 2;
      static int totalPackets;             //The total number of StreamData responses read

      static int numReadsPerDisplay = 1;
      static int readSizeMultiplier = 12;
      static int responseSize = 14 + SamplesPerPacket * 2;
      if (this->m_Initialized) {
        int recChars, backLog;
        int i, j, k, m, currChannel, scanNumber;
        uint16 voltageBytes, checksumTotal;
        int autoRecoveryOn;


        /* Each StreamData response contains (SamplesPerPacket / NumChannels) * readSizeMultiplier
         * samples for each channel.
         * Total number of scans = (SamplesPerPacket / NumChannels) * readSizeMultiplier * numReadsPerDisplay * numDisplay
         */
        //int div = SamplesPerPacket/NumChannels;
        //double voltages[(div) * readSizeMultiplier * numReadsPerDisplay * numDisplay][NumChannels];
        double volts[NumChannels];
        int count[NumChannels];
        //double voltages[(div) * readSizeMultiplier * numReadsPerDisplay * numDisplay][NumChannels];
        uint8 recBuff[responseSize * readSizeMultiplier];
        //packetCounter = 0;
        currChannel = 0;
        scanNumber = 0;
        totalPackets = 0;
        recChars = 0;
        autoRecoveryOn = 0;

        //printf("Reading Samples...\n");

        for (k = 0; k < NumChannels; k++) {
          volts[k] = 0.0;
          count[k] = 0;
        }
        double t1 = getTickCount();
        for (j = 0; j < numReadsPerDisplay; j++) {
          /* For USB StreamData, use Endpoint 3 for reads.  You can read the multiple
           * StreamData responses of 64 bytes only if SamplesPerPacket is 25 to help
           * improve streaming performance.  In this example this multiple is adjusted
           * by the readSizeMultiplier variable.
           */

          //Reading stream response from U6
          recChars = LJUSB_Stream(hDevice, recBuff, responseSize * readSizeMultiplier);
          if (recChars < responseSize * readSizeMultiplier) {
            if (recChars == 0)
              printf("Error : read failed (StreamData).\n");
            else
              printf("Error : did not read all of the buffer, expected %d bytes but received %d(StreamData).\n",
                  responseSize * readSizeMultiplier, recChars);

            return -1;
          }
          //Checking for errors and getting data out of each StreamData response
          for (m = 0; m < readSizeMultiplier; m++) {
            totalPackets++;

            checksumTotal = extendedChecksum16(recBuff + m * recBuffSize, recBuffSize);
            if ((uint8) ((checksumTotal >> 8) & 0xff) != recBuff[m * recBuffSize + 5]) {
              printf("Error : read buffer has bad checksum16(MSB) (StreamData).\n");
              return -1;
            }
            if ((uint8) (checksumTotal & 0xff) != recBuff[m * recBuffSize + 4]) {
              printf("Error : read buffer has bad checksum16(LSB) (StreamData).\n");
              return -1;
            }
            checksumTotal = extendedChecksum8(recBuff + m * recBuffSize);
            if (checksumTotal != recBuff[m * recBuffSize]) {
              printf("Error : read buffer has bad checksum8 (StreamData).\n");
              return -1;
            }
            if (recBuff[m * recBuffSize + 1] != (uint8) (0xF9)
                || recBuff[m * recBuffSize + 2] != 4 + SamplesPerPacket
                || recBuff[m * recBuffSize + 3] != (uint8) (0xC0)) {
              printf("Error : read buffer has wrong command bytes (StreamData).\n");
              return -1;
            }
            if (recBuff[m * recBuffSize + 11] == 59) {
              if (!autoRecoveryOn) {
                printf("\nU6 data buffer overflow detected in packet %d.\nNow using auto-recovery and reading buffered samples.\n",
                    totalPackets);
                autoRecoveryOn = 1;
              }
            } else if (recBuff[m * recBuffSize + 11] == 60) {
              printf("Auto-recovery report in packet %d: %d scans were dropped.\nAuto-recovery is now off.\n",
                  totalPackets,
                  recBuff[m * recBuffSize + 6] + recBuff[m * recBuffSize + 7] * 256);
              autoRecoveryOn = 0;
            } else if (recBuff[m * recBuffSize + 11] != 0) {
              printf("Errorcode # %d from StreamData read.\n", (unsigned int) recBuff[11]);
              return -1;
            }
            if (packetCounter != (int) recBuff[m * recBuffSize + 10]) {
              printf("PacketCounter (%d) does not match with with current packet count (%d)(StreamData).\n", recBuff[m * recBuffSize + 10], packetCounter);
              return -1;
            }

            backLog = (int) recBuff[m * 48 + 12 + SamplesPerPacket * 2];

            for (k = 12; k < (12 + SamplesPerPacket * 2); k += 2) {
              voltageBytes =
                (uint16) recBuff[m * recBuffSize + k] +
                (uint16) recBuff[m * recBuffSize + k + 1] * 256;

              //getAinVoltCalibrated(caliInfo, 1, 0, 0, voltageBytes, &(voltages[scanNumber][currChannel]));
              double raw;
              getAinVoltCalibrated(&caliInfo, 1, 0, 0, voltageBytes, &(raw));

              volts[currChannel]+=raw;
              count[currChannel]+=1;
              //printf("   %.4f V %d r\n", volts[currChannel], count[currChannel]);

              currChannel++;
              if (currChannel >= NumChannels) {
                currChannel = 0;
                scanNumber++;
              }
            }

            if (packetCounter >= 255) packetCounter = 0;
            else packetCounter++;
          }
        }
        double t2 = getTickCount();

        printf("\nNumber of scans: %d\n", scanNumber);
        printf("Total packets read: %d\n", totalPackets);
        printf("Current PacketCounter: %d\n", ((packetCounter == 0) ? 255 : packetCounter - 1));
        printf("Current BackLog: %d\n", backLog);
        printf("Milliseconds per read: %f\n", (t2-t1)/(double)numReadsPerDisplay);

        for (k = 0; k < NumChannels; k++) {
          //printf("  AI%d: %.4f V %d c\n", k, volts[k]/count[k], count[k]);
          volts[k] = volts[k] / count[k];
        }

        float raw[6];
        float FT[6];            // This array will hold the resultant force/torque vector.
        for (int k=0; k<6; k++) {
          raw[k] = volts[k];// / count[k];
        }
        // convert raw values to usable units / calibration
        ConvertToFT(cal_r,raw,FT);
        for (int i = 0; i<6; i++) {
          r[i] = (double)FT[i];
          printf("  AI%d: %.4f V %d c %f f/t\n", i, volts[i], count[i], r[i]);
        }

        for (k = 0; k < NumChannels; k++) {
        }
        for (int k=0; k<6; k++) {
          raw[k] = volts[k+6];// / count[k+6];
        }
        ConvertToFT(cal_l,raw,FT);
        for (int i = 0; i<6; i++) {
          l[i] = (double)FT[i];
          printf("  AI%d: %.4f V %d c %f f/t\n", i+6, volts[i], count[i], r[i]);
        }
        return true;
        
      }
      else {
        printf("ContactSensors not running.\n");
        return false;
      }
    }

    ~ContactSensors() {
      destroyCalibration(cal_r);
      destroyCalibration(cal_l);

      StreamStop(hDevice);
      closeUSBConnection(hDevice);

      //if (sendBuff) delete[] sendBuff;
      //if (recvBuff) delete[] recvBuff;

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

    int ConfigIO_example(HANDLE hDevice)
    {
      uint8 sendBuff[16], recBuff[16];
      uint16 checksumTotal;
      int sendChars, recChars, i;

      sendBuff[1] = (uint8) (0xF8); //Command byte
      sendBuff[2] = (uint8) (0x03); //Number of data words
      sendBuff[3] = (uint8) (0x0B); //Extended command number

      sendBuff[6] = 1;              //Writemask : Setting writemask for TimerCounterConfig (bit 0)

      sendBuff[7] = 0;              //NumberTimersEnabled : Setting to zero to disable all timers.
      sendBuff[8] = 0;              //CounterEnable: Setting bit 0 and bit 1 to zero to disable both counters
      sendBuff[9] = 0;              //TimerCounterPinOffset

      for (i = 10; i < 16; i++)
        sendBuff[i] = 0;            //Reserved
      extendedChecksum(sendBuff, 16);

      //Sending command to U6
      if ((sendChars = LJUSB_Write(hDevice, sendBuff, 16)) < 16) {
        if (sendChars == 0)
          printf("ConfigIO error : write failed\n");
        else
          printf("ConfigIO error : did not write all of the buffer\n");
        return -1;
      }
      //Reading response from U6
      if ((recChars = LJUSB_Read(hDevice, recBuff, 16)) < 16) {
        if (recChars == 0)
          printf("ConfigIO error : read failed\n");
        else
          printf("ConfigIO error : did not read all of the buffer\n");
        return -1;
      }

      checksumTotal = extendedChecksum16(recBuff, 15);
      if ((uint8) ((checksumTotal / 256) & 0xff) != recBuff[5]) {
        printf("ConfigIO error : read buffer has bad checksum16(MSB)\n");
        return -1;
      }

      if ((uint8) (checksumTotal & 0xff) != recBuff[4]) {
        printf("ConfigIO error : read buffer has bad checksum16(LSB)\n");
        return -1;
      }

      if (extendedChecksum8(recBuff) != recBuff[0]) {
        printf("ConfigIO error : read buffer has bad checksum8\n");
        return -1;
      }

      if (recBuff[1] != (uint8) (0xF8) || recBuff[2] != (uint8) (0x05)
          || recBuff[3] != (uint8) (0x0B)) {
        printf("ConfigIO error : read buffer has wrong command bytes\n");
        return -1;
      }

      if (recBuff[6] != 0) {
        printf("ConfigIO error : read buffer received errorcode %d\n",
            recBuff[6]);
        return -1;
      }

      if (recBuff[8] != 0) {
        printf("ConfigIO error : NumberTimersEnabled was not set to 0\n");
        return -1;
      }

      if (recBuff[9] != 0) {
        printf("ConfigIO error : CounterEnable was not set to 0\n");
        return -1;
      }

      return 0;
    }

    int StreamConfig_example(HANDLE hDevice)
    {
      int sendBuffSize;
      sendBuffSize = 14 + NumChannels * 2;
      uint8 sendBuff[sendBuffSize], recBuff[8];
      int sendChars, recChars;
      uint16 checksumTotal;
      uint16 scanInterval;
      int i;

      sendBuff[1] = (uint8) (0xF8); //Command byte
      sendBuff[2] = 4 + NumChannels;        //Number of data words = NumChannels + 4
      sendBuff[3] = (uint8) (0x11); //Extended command number
      sendBuff[6] = NumChannels;    //NumChannels
      sendBuff[7] = 1;              //ResolutionIndex
      sendBuff[8] = SamplesPerPacket;       //SamplesPerPacket
      sendBuff[9] = 0;              //Reserved
      sendBuff[10] = 0;             //SettlingFactor: 0
      sendBuff[11] = 0;             //ScanConfig:
      //  Bit 3: Internal stream clock frequency = b0: 4 MHz
      //  Bit 1: Divide Clock by 256 = b0

      scanInterval = 1000;
      sendBuff[12] = (uint8) (scanInterval & (0x00FF));     //scan interval (low byte)
      sendBuff[13] = (uint8) (scanInterval / 256);  //scan interval (high byte)

      for (i = 0; i < NumChannels; i++) {
        sendBuff[14 + i * 2] = i;   //ChannelNumber (Positive) = i
        sendBuff[15 + i * 2] = 0;   //ChannelOptions: 
        //  Bit 7: Differential = 0
        //  Bit 5-4: GainIndex = 0 (+-10V)
      }

      extendedChecksum(sendBuff, sendBuffSize);

      //Sending command to U6
      sendChars = LJUSB_Write(hDevice, sendBuff, sendBuffSize);
      if (sendChars < sendBuffSize) {
        if (sendChars == 0)
          printf("Error : write failed (StreamConfig).\n");
        else
          printf("Error : did not write all of the buffer (StreamConfig).\n");
        return -1;
      }

      for (i = 0; i < 8; i++)
        recBuff[i] = 0;

      //Reading response from U6
      recChars = LJUSB_Read(hDevice, recBuff, 8);
      if (recChars < 8) {
        if (recChars == 0)
          printf("Error : read failed (StreamConfig).\n");
        else
          printf
            ("Error : did not read all of the buffer, %d (StreamConfig).\n",
             recChars);

        for (i = 0; i < 8; i++)
          printf("%d ", recBuff[i]);

        return -1;
      }

      checksumTotal = extendedChecksum16(recBuff, 8);
      if ((uint8) ((checksumTotal / 256) & 0xff) != recBuff[5]) {
        printf
          ("Error : read buffer has bad checksum16(MSB) (StreamConfig).\n");
        return -1;
      }

      if ((uint8) (checksumTotal & 0xff) != recBuff[4]) {
        printf
          ("Error : read buffer has bad checksum16(LSB) (StreamConfig).\n");
        return -1;
      }

      if (extendedChecksum8(recBuff) != recBuff[0]) {
        printf("Error : read buffer has bad checksum8 (StreamConfig).\n");
        return -1;
      }

      if (recBuff[1] != (uint8) (0xF8) || recBuff[2] != (uint8) (0x01)
          || recBuff[3] != (uint8) (0x11) || recBuff[7] != (uint8) (0x00)) {
        printf
          ("Error : read buffer has wrong command bytes (StreamConfig).\n");
        return -1;
      }

      if (recBuff[6] != 0) {
        printf("Errorcode # %d from StreamConfig read.\n",
            (unsigned int) recBuff[6]);
        return -1;
      }

      return 0;
    }

    int StreamStart(HANDLE hDevice)
    {
      uint8 sendBuff[2], recBuff[4];
      int sendChars, recChars;

      sendBuff[0] = (uint8) (0xA8); //Checksum8
      sendBuff[1] = (uint8) (0xA8); //Command byte

      //Sending command to U6
      sendChars = LJUSB_Write(hDevice, sendBuff, 2);
      if (sendChars < 2) {
        if (sendChars == 0)
          printf("Error : write failed.\n");
        else
          printf("Error : did not write all of the buffer.\n");
        return -1;
      }
      //Reading response from U6
      recChars = LJUSB_Read(hDevice, recBuff, 4);
      if (recChars < 4) {
        if (recChars == 0)
          printf("Error : read failed.\n");
        else
          printf("Error : did not read all of the buffer.\n");
        return -1;
      }

      if (normalChecksum8(recBuff, 4) != recBuff[0]) {
        printf("Error : read buffer has bad checksum8 (StreamStart).\n");
        return -1;
      }

      if (recBuff[1] != (uint8) (0xA9) || recBuff[3] != (uint8) (0x00)) {
        printf("Error : read buffer has wrong command bytes \n");
        return -1;
      }

      if (recBuff[2] != 0) {
        printf("Errorcode # %d from StreamStart read.\n",
            (unsigned int) recBuff[2]);
        return -1;
      }
      return 0;
    }

    int StreamStop(HANDLE hDevice)
    {
      uint8 sendBuff[2], recBuff[4];
      int sendChars, recChars;

      sendBuff[0] = (uint8) (0xB0); //Checksum8
      sendBuff[1] = (uint8) (0xB0); //Command byte

      //Sending command to U6
      sendChars = LJUSB_Write(hDevice, sendBuff, 2);
      if (sendChars < 2) {
        if (sendChars == 0)
          printf("Error : write failed (StreamStop).\n");
        else
          printf("Error : did not write all of the buffer (StreamStop).\n");
        return -1;
      }
      //Reading response from U6
      recChars = LJUSB_Read(hDevice, recBuff, 4);
      if (recChars < 4) {
        if (recChars == 0)
          printf("Error : read failed (StreamStop).\n");
        else
          printf("Error : did not read all of the buffer (StreamStop).\n");
        return -1;
      }

      if (normalChecksum8(recBuff, 4) != recBuff[0]) {
        printf("Error : read buffer has bad checksum8 (StreamStop).\n");
        return -1;
      }

      if (recBuff[1] != (uint8) (0xB1) || recBuff[3] != (uint8) (0x00)) {
        printf("Error : read buffer has wrong command bytes (StreamStop).\n");
        return -1;
      }

      if (recBuff[2] != 0) {
        printf("Errorcode # %d from StreamStop read.\n",
            (unsigned int) recBuff[2]);
        return -1;
      }

      return 0;
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
    //uint8 *sendBuff;
    //uint8 *recvBuff;
    uint8 NumChannels;
    uint8 SamplesPerPacket;
};

/* command / req config
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
*/


/* old command response code
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
*/


