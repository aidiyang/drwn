/*
 *   WindowsCM730.h
 *
 *   Author: ROBOTIS
 *
 */

#ifndef _WINDOWS_CM730_H_
#define _WINDOWS_CM730_H_

#include "CM730.h"
#include "windows.h"
#define _CRT_SECURE_NO_WARNINGS
#ifdef GetCurrentTime
#undef GetCurrentTime
#endif


namespace Robot
{
	class WindowsCM730 : public PlatformCM730
	{
		private:
			HANDLE m_Socket_fd;
			double m_PacketStartTime;
			double m_PacketWaitTime;
			double m_UpdateStartTime;
			double m_UpdateWaitTime;
			double m_ByteTransferTime;
			char m_PortName[20];

			HANDLE m_LowSemID;
			HANDLE m_MidSemID;
			HANDLE m_HighSemID;


		public:
			bool DEBUG_PRINT;

			WindowsCM730(const char* name);
			~WindowsCM730();

			void SetPortName(const char* name);
			const char* GetPortName()		{ return (const char*)m_PortName; }

			///////////////// Platform Porting //////////////////////
			bool OpenPort();
			bool SetBaud(int baud);
            bool isOpen();
			void ClosePort();
			void ClearPort();
			int WritePort(unsigned char* packet, int numPacket);
			int ReadPort(unsigned char* packet, int numPacket);

			void LowPriorityWait();
			void MidPriorityWait();
			void HighPriorityWait();
			void LowPriorityRelease();
            void MidPriorityRelease();
			void HighPriorityRelease();

			void SetPacketTimeout(int lenPacket);
			bool IsPacketTimeout();
			double GetPacketTime();
			void SetUpdateTimeout(int msec);
			bool IsUpdateTimeout();
			double GetUpdateTime();

			void Sleep(double msec);

            double GetCurrentTime();
			////////////////////////////////////////////////////////
	};
}

#endif
