#pragma once

#include "stdio.h"
#include "owl.h"

#include <cstring>
#include <chrono>
#include <thread>
#include <mutex>

#define MARKER_COUNT 8
#define PS_SERVER_NAME "128.208.4.127"
#define INIT_FLAGS 0

#define POSE_SIZE 7
#define COND_SIZE 1

// use rb_2_c.sh darwin.rb -- to get the formatted outputs
float RIGID_BODY[MARKER_COUNT][3] = {
  { 0.00, 0.00, 0.00 },
  { 61.43, -3.23, -36.64 },
  { 46.60, -73.71, -76.01 },
  { 60.11, -26.91, 29.44 },
  { -54.11, -6.29, -43.60 },
  { -56.23, -32.45, 23.62 },
  { -26.04, -70.43, -79.45 },
  { 12.45, -111.22, -60.58 }
};

class Phasespace {
  private:

    void Phasespace_t(Phasespace *track) {
      OWLRigid rigid;
      OWLMarker markers[32];
      int tracker;
      if (owlInit((track->server_name).c_str(), INIT_FLAGS) < 0) {
        printf("Couldn't connect to Phase Space\n");
        track->m_TrackerRunning = false;
        return;
      }
      // create tracker 0
      tracker = 0;
      owlTrackeri(tracker, OWL_CREATE, OWL_RIGID_TRACKER);
      for (int i = 0; i < MARKER_COUNT; i++) {
        owlMarkeri(MARKER(tracker, i), OWL_SET_LED, i);
        owlMarkerfv(MARKER(tracker, i), OWL_SET_POSITION, RIGID_BODY[i]);
      }
      owlTracker(tracker, OWL_ENABLE);
      if (!owlGetStatus()) {
        track->owl_print_error("error in point tracker setup", owlGetError());
        track->m_TrackerRunning = false;
        return;
      }
      owlSetFloat(OWL_FREQUENCY, OWL_MAX_FREQUENCY);
      owlSetInteger(OWL_STREAMING, OWL_ENABLE);

      int count = 0;
      std::chrono::milliseconds interval(1);
      while (!track->m_FinishTracking) {
        int nmarker = 0;
        int nrigid = 0;
        if (track->use_markers) {
          nmarker = owlGetMarkers(markers, 32);
        }
        if (track->use_rigid) {
          nrigid = owlGetRigids(&rigid, 1);
        }

        // check for error
        int err;
        if ((err = owlGetError()) != OWL_NO_ERROR) {
          track->owl_print_error("error", err);
          track->m_TrackerRunning = false;
          return;
        }

        // make sure we got a new frame
        if ((track->use_rigid && nrigid<1)
           || (track->use_markers && nmarker<1)) {
          //printf("No new frame...\n");
          std::this_thread::sleep_for(interval);
          continue;
        }

        //if (count%100 == 0) {
        //	std::cout<<"Raw:: "<<rigid.pose[0]<<","<<rigid.pose[1]<<","<<rigid.pose[2]<<","
        //		<<rigid.pose[3]<<","<<rigid.pose[4]<<","<<rigid.pose[5]<<","
        //		<<rigid.pose[6]<<std::endl;
        //}

        //	there's potentially stuff we can do with the cond and frame
        //cond[cnt] = rigid.cond;
        //frame[cnt] = rigid.frame;

        // mutex; if locked, forget this and grab next data point
        if (track->mutex.try_lock()) {
          if (track->use_markers) {
            for (int i = 0; i < MARKER_COUNT; i++) {
              marker_d[4 * i + 0] = markers[i].x / 1000.0;
              marker_d[4 * i + 1] = markers[i].x / 1000.0;
              marker_d[4 * i + 2] = markers[i].x / 1000.0;
              marker_d[4 * i + 3] = markers[i].cond;
            }
          }
          if (track->use_rigid) {
            std::memcpy(track->pose, rigid.pose, sizeof(float)*POSE_SIZE);
            track->pose[7] = rigid.cond;
          }

          track->mutex.unlock();
        }

        count++;
        std::this_thread::sleep_for(interval);
      }

      owlDone();
      printf("Phasespace loop ran %d times.\n", count);

    }

    void owl_print_error(const char *s, int n) {
      if (n < 0) printf("%s: %d\n", s, n);
      else if (n == OWL_NO_ERROR) printf("%s: No Error\n", s);
      else if (n == OWL_INVALID_VALUE) printf("%s: Invalid Value\n", s);
      else if (n == OWL_INVALID_ENUM) printf("%s: Invalid Enum\n", s);
      else if (n == OWL_INVALID_OPERATION) printf("%s: Invalid Operation\n", s);
      else printf("%s: 0x%x\n", s, n);
    }

    std::thread m_Thread;
    std::mutex mutex;
    std::string server_name;
    bool m_Initialized;
    bool m_TrackerRunning;
    bool m_FinishTracking;
    bool use_rigid;
    bool use_markers;
    float pose[POSE_SIZE + COND_SIZE];
    float marker_d[MARKER_COUNT * 4];

  public:

    Phasespace(bool rigid, bool markers, std::string server)  {

      std::memset(this->pose, 0, sizeof(float)*POSE_SIZE);
      this->m_Initialized = false;
      this->m_TrackerRunning = true;
      this->m_FinishTracking = false;
      this->server_name = server;
      this->use_rigid = rigid;
      this->use_markers = markers;
      //MotionStatus::PHASESPACE_ON = true;

      if (this->use_rigid || this->use_markers) {
        m_Thread = std::thread(&Phasespace::Phasespace_t, this, this);
      }
      else {
        printf("Not using phasespace\n");
      }
    }

    bool isRunning() {
      return this->m_TrackerRunning;
    }

    bool getData(double * p, double * m) {
      if (this->m_TrackerRunning && this->m_Initialized) {
        //std::cout<<"Copying: "<<pose[0]<<","<<pose[1]<<","<<pose[2]<<","
        //	<<pose[3]<<","<<pose[4]<<","<<pose[5]<<","
        //  	<<pose[6]<<std::endl;

        this->mutex.lock(); // not a try; wait for newest
        for (int i = 0; i<POSE_SIZE; i++) {
          p[i] = (double)pose[i];
        }
        for (int i = 0; i<(MARKER_COUNT * 4); i++) {
          m[i] = (double)marker_d[i];
        }
        this->mutex.unlock();
        return true;
      }
      else {
        printf("Phasespace not running.\n");
        return false;
      }
    }

    bool getMarkers(double *m) {
      if (this->m_TrackerRunning && this->m_Initialized) {
        //std::cout<<"Copying: "<<pose[0]<<","<<pose[1]<<","<<pose[2]<<","
        //	<<pose[3]<<","<<pose[4]<<","<<pose[5]<<","
        //  	<<pose[6]<<std::endl;

        this->mutex.lock(); // not a try; wait for newest
        for (int i = 0; i<(MARKER_COUNT*4); i++) {
          m[i] = (double)marker_d[i];
        }
        this->mutex.unlock();
        return true;
      }
      else {
        printf("Phasespace not running.\n");
        return false;
      }
    }

    bool getRigid(double *p) {
      if (this->m_TrackerRunning && this->m_Initialized) {
        //std::cout<<"Copying: "<<pose[0]<<","<<pose[1]<<","<<pose[2]<<","
        //	<<pose[3]<<","<<pose[4]<<","<<pose[5]<<","
        //  	<<pose[6]<<std::endl;

        this->mutex.lock(); // not a try; wait for newest
        for (int i = 0; i<POSE_SIZE; i++) {
          p[i] = (double)pose[i];
        }
        this->mutex.unlock();
        return true;
      }
      else {
        printf("Phasespace not running.\n");
        return false;
      }
    }

    ~Phasespace() {
      if (this->m_TrackerRunning) {
        printf("Stopping Phasespace Module\n");
        this->m_FinishTracking = true;
        // wait for the thread to end
        m_Thread.join();
        this->m_Initialized = false;
        this->m_FinishTracking = false;
        this->m_TrackerRunning = false;
      }
    }
};
