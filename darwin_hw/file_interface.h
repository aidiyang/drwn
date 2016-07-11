#pragma once


//#include "Utilities.h"
#include "robot.h"
#include <iostream>
#include <fstream>
#include <boost/tokenizer.hpp>

#include <string.h>
#include <Eigen/Dense>
#include <vector>

class FileDarwin : public MyRobot {
    private:

        std::ifstream infile;

        double s_dt;

        double s_time_noise;

        std::mt19937 gen;
        std::normal_distribution<> t_noise;
        std::normal_distribution<> sen_noise;
        std::normal_distribution<> ctrl_noise;

        // model helpers
        int nu;
        int ns;
        int rows;
        int cols;
        int t;
        Eigen::MatrixXd data;
        std::vector<double> values;

    public:
        FileDarwin(
                int _nu, int _ns,
                std::string input_file) {

            t=0;

            printf("Loading File %s\n", input_file.c_str());
            infile.open(input_file, std::ifstream::in);

            // read in all data to memory
            typedef boost::tokenizer< boost::char_separator<char> > Tokenizer;
            boost::char_separator<char> sep(",");
            std::string line;
            getline(infile, line); // get rid of first line
            rows=0;
            cols=0;
            while (getline(infile, line)) {
                cols=0;
                Tokenizer info(line, sep);
                for (Tokenizer::iterator it=info.begin(); it!=info.end(); ++it) {
                    values.push_back(strtod(it->c_str(),0));
                    cols++;
                }
                printf(".");
                fflush(stdout);
                rows++;
            }

            printf("File Loaded with %d rows, %d cols.\n", rows, cols);
            data = Eigen::MatrixXd::Zero(cols, rows);
            data = Eigen::Map<Eigen::MatrixXd> (values.data(), cols, rows);

            printf("Data lasts for %f seconds.\n", data(0, rows-1));

            //std::cout<<data.col(0).transpose()<<std::endl;
            //std::cout<<std::endl;
            //std::cout<<std::endl;
            //std::cout<<data.row(0)<<std::endl; // column

            darwin_ok = true;

            nu = _nu;
            ns = _ns;

        }

        ~FileDarwin() {
            //delete[] this->i_pose;
        }

        bool get_sensors(double * time, double* sensor, double* conf) {

            *time = data.col(t)(0);

            if (sensor) {
                double * snsr = &((data.col(t))(nu+1));
                for (int id=0; id<(20+20+6+12); id++) {
                    //sensor[id] = row[id+nu+1]; 
                    sensor[id] = snsr[id]; 
                }
                int o=(20+20+6+12);
                // remap the data
                for (int id=0; id<16; id++) {
                    sensor[o+id*3+0] = snsr[o+id*4+0]; 
                    sensor[o+id*3+1] = snsr[o+id*4+1]; 
                    sensor[o+id*3+2] = snsr[o+id*4+2]; 
                    //conf[id] = snsr[o+id*4+3];
                }
                o += 16*3;
                for (int id=0; id<16; id++) {
                    conf[id] = snsr[o+id];
                }
            }
            else {
                printf("Initialize sensor buffer\n");
                return false;
            }

            t = t+1; // advance time
            if (t>=rows)
                return false;

            printf("Row: %d, time: %f\n", t, *time);
            return true;
        }

        // get controls from file data and put into u variable
        bool set_controls(double * u, int *pgain, int *dgain) {
            // converts controls to darwin positions
            double *ctrl = &((data.col(t))(1));

            for(int id = 0; id < nu; id++) {
                //u[id] = row[id+1];
                u[id] = ctrl[id];
            }

            return true;
        }

};


