/*
 *   WindowsCM730.cpp
 *
 *   Author: ROBOTIS
 *
 */
#include <stdio.h>
#include <errno.h>
//#include <unistd.h>
#include <string.h>
#include <fcntl.h>
//#include <termios.h>
#include <time.h>
//#include <sys/time.h>
//#include <linux/serial.h>
//#include <sys/ioctl.h>
#include "WindowsCM730.h"
#include <chrono>
#include <thread>

using namespace Robot;


WindowsCM730::WindowsCM730(const char* name)
{
	DEBUG_PRINT = false;
	m_Socket_fd = INVALID_HANDLE_VALUE;
	m_PacketStartTime = 0;
	m_PacketWaitTime = 0;
	m_UpdateStartTime = 0;
	m_UpdateWaitTime = 0;
	m_ByteTransferTime = 0;

	m_LowSemID =CreateSemaphore(NULL, 1, 10000, NULL);
	m_MidSemID =CreateSemaphore(NULL, 1, 10000, NULL);
	m_HighSemID=CreateSemaphore(NULL, 1, 10000, NULL);

	SetPortName(name);
}

WindowsCM730::~WindowsCM730()
{
	ClosePort();
}

void WindowsCM730::SetPortName(const char* name)
{
	strcpy_s(m_PortName, name);
}

bool WindowsCM730::OpenPort()
{
    bool DEBUG_PRINT = true;
    //struct termios newtio;
	//struct serial_struct serinfo;
	double baudrate = 1000000.0; //bps (1Mbps)
    DCB dcb = {0};
    COMMTIMEOUTS timeouts;

	ClosePort();

	if(DEBUG_PRINT == true)
		printf("\n%s open ", m_PortName);

    //m_Socket_fd=CreateFile(m_PortName, GENERIC_READ|GENERIC_WRITE, 0, 0, OPEN_EXISTING,FILE_FLAG_OVERLAPPED, 0);
    m_Socket_fd=CreateFile(m_PortName, GENERIC_READ|GENERIC_WRITE, 0, NULL, OPEN_EXISTING,0, NULL);
    if (m_Socket_fd == INVALID_HANDLE_VALUE)
		goto UART_OPEN_ERROR;

	if(DEBUG_PRINT == true)
		printf("success!\n");

	// You must set 38400bps!
    memset(&dcb, sizeof(dcb), 0);
    dcb.DCBlength = sizeof(dcb);
    //if (!BuildCommDCB("38400,n,8,1", &dcb)) {
    if (!GetCommState(m_Socket_fd, &dcb)) {
      goto UART_OPEN_ERROR;
    }
    else {
      dcb.BaudRate = (DWORD)baudrate;
      dcb.ByteSize = 8;
      dcb.Parity   = NOPARITY;
      dcb.StopBits = ONESTOPBIT;
      dcb.fParity  = NOPARITY;
      dcb.fBinary  = 1;
      dcb.fNull    = 0;
      dcb.fAbortOnError = 1;
      dcb.fErrorChar    = 0;
      dcb.fOutX         = 0;
      dcb.fInX          = 0;
      dcb.fDtrControl   = DTR_CONTROL_ENABLE;
      dcb.fRtsControl   = RTS_CONTROL_ENABLE;
      dcb.fDsrSensitivity = 0;
      dcb.fOutxDsrFlow    = 0;
      dcb.fOutxCtsFlow    = 0;
    }

    if (!SetCommState(m_Socket_fd, &dcb)) {
        printf("Unable to set communication state on '%s'\n", m_PortName);
      goto UART_OPEN_ERROR;
    }
    if (!SetCommMask(m_Socket_fd, 0)) { // Not using Comm event
        printf("Unable to set communication mask on '%s'\n", m_PortName);
      goto UART_OPEN_ERROR;
    }
    
    if (!SetupComm(m_Socket_fd, 2048, 2048)) { // Buffer size (Rx,Tx)
        printf("Unable to setup communication on '%s'\n", m_PortName);
      goto UART_OPEN_ERROR;
    }
    
    if (!PurgeComm(m_Socket_fd, PURGE_TXABORT|PURGE_TXCLEAR|PURGE_RXABORT|PURGE_RXCLEAR)) { // Clear buffer
        printf("Unable to purge communication on '%s'\n", m_PortName);
      goto UART_OPEN_ERROR;
    }
    /*
    if (!ClearCommError(m_Socket_fd, &dwError, NULL)) {
      goto UART_OPEN_ERROR;
    }
    if (!GetCommTimeouts(m_Socket_fd, &timeouts)) {
      goto UART_OPEN_ERROR;
    }
    */

    // Timeout (Not using timeout)
    // Immediatly return
    timeouts.ReadIntervalTimeout         = 0;
    timeouts.ReadTotalTimeoutMultiplier  = 0;
    timeouts.ReadTotalTimeoutConstant    = 1; // Must not be zero
    timeouts.WriteTotalTimeoutMultiplier = 0;
    timeouts.WriteTotalTimeoutConstant   = 0;

    if (!SetCommTimeouts(m_Socket_fd, &timeouts))
      goto UART_OPEN_ERROR;

    // Why is there a * 8 in the other function?
    if (DEBUG_PRINT == true)
        m_ByteTransferTime = (1000.0 / baudrate) * 12.0 * 800.0;
    else
        m_ByteTransferTime = (1000.0 / baudrate) * 12.0 * 8.0;

    return true;

UART_OPEN_ERROR:
    if(DEBUG_PRINT == true)
      printf("failed!\n");
    ClosePort();
    return false;
}

bool WindowsCM730::SetBaud(int baud)
{
  /* meaningless function; do nothing */

  m_ByteTransferTime = (float)((1000.0f / 1000000.0f) * 12.0f * 8);

  return true;
}

bool WindowsCM730::isOpen()
{
    bool status = false;

    if (m_Socket_fd != INVALID_HANDLE_VALUE) {
        DCB dcb;
        dcb.DCBlength = sizeof(DCB);
        if (GetCommState(m_Socket_fd, &dcb)) {
            status = true;
        }
    }
    return status;
}

void WindowsCM730::ClosePort()
{
  if (isOpen()) {
    this->ClearPort();
    CloseHandle(m_Socket_fd);
    m_Socket_fd = INVALID_HANDLE_VALUE;
  }
}

void WindowsCM730::ClearPort()
{
  if (isOpen()) {
    //PURGE_RXABORT: Terminates all outstanding overlapped read operations and returns immediately, even if the read operations have not been completed.
    //PURGE_RXCLEAR: Clears the input buffer (if the device driver has one).
    //PURGE_TXABORT: Terminates all outstanding overlapped write operations and returns immediately, even if the write operations have not been completed.
    //PURGE_TXCLEAR: Clears the output buffer (if the device driver has one).

    PurgeComm(m_Socket_fd, PURGE_RXABORT | PURGE_RXCLEAR);
  }
}

int WindowsCM730::WritePort(unsigned char* packet, int numPacket)
{
  int status = -1;
  DWORD written=0;
  if (WriteFile(m_Socket_fd, packet, (DWORD)numPacket, &written, 0)) {
    status = static_cast<int>(written);
  }
  return status;
}

int WindowsCM730::ReadPort(unsigned char* packet, int numPacket)
{
  int status = -1;
  DWORD read=0;
  // last null in readfile is for when serial port is not overlapped (aka
  // non-blocking)
  if (ReadFile(m_Socket_fd, packet, (DWORD)numPacket, &read, NULL)) {
    status = static_cast<int>(read);
  }
  return status;
}

void sem_wait_nointr(HANDLE *sem)
{
  int sem_result=0;
  //do {
    //sem_result = sem_wait(sem);
    sem_result=WaitForSingleObject(sem, INFINITE); //==WAIT_OBJECT_0)
    //TODO potentially very bad, but not sure how to block the interrupt
  //} while((sem_result == WAIT_FAILED) && (errno == EINTR));
}

void WindowsCM730::LowPriorityWait()
{
  sem_wait_nointr(&m_LowSemID);
}

void WindowsCM730::MidPriorityWait()
{
  sem_wait_nointr(&m_MidSemID);
}

void WindowsCM730::HighPriorityWait()
{
  sem_wait_nointr(&m_HighSemID);
}

void WindowsCM730::LowPriorityRelease()
{
  ReleaseSemaphore(m_LowSemID, 1, NULL);
}

void WindowsCM730::MidPriorityRelease()
{
  ReleaseSemaphore(m_MidSemID, 1, NULL);
}

void WindowsCM730::HighPriorityRelease()
{
  ReleaseSemaphore(m_HighSemID, 1, NULL);
}

double WindowsCM730::GetCurrentTime()
{
  std::chrono::time_point<std::chrono::high_resolution_clock> t
    = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> d=t.time_since_epoch();
  return d.count(); // returns milliseconds
}

void WindowsCM730::SetPacketTimeout(int lenPacket)
{
  m_PacketStartTime = GetCurrentTime();
  m_PacketWaitTime = m_ByteTransferTime * (double)lenPacket + 5.0;
}

bool WindowsCM730::IsPacketTimeout()
{
  if(GetPacketTime() > m_PacketWaitTime)
    return true;

  return false;
}

double WindowsCM730::GetPacketTime()
{
  double time;

  double curTime = GetCurrentTime();

  time = curTime - m_PacketStartTime;
  if(time < 0.0)
    m_PacketStartTime = curTime;

  return time;
}

void WindowsCM730::SetUpdateTimeout(int msec)
{
  m_UpdateStartTime = GetCurrentTime();
  m_UpdateWaitTime = msec;
}

bool WindowsCM730::IsUpdateTimeout()
{
  if(GetUpdateTime() > m_UpdateWaitTime)
    return true;

  return false;
}

double WindowsCM730::GetUpdateTime()
{
  double time;

  time = GetCurrentTime() - m_UpdateStartTime;
  if(time < 0.0)
    m_UpdateStartTime = GetCurrentTime();

  return time;
}

void WindowsCM730::Sleep(double msec)
{
  double start_time = GetCurrentTime();
  double curr_time = start_time;

  do {
    std::this_thread::sleep_for(std::chrono::milliseconds((int)((start_time + msec) - curr_time)));
    curr_time = GetCurrentTime();
  } while(curr_time - start_time < msec);
}
