#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include "sine.h"
#include "cosine.h"
#include "inv_matrix.h"

// reference: [1]Liu JinKun. Robot Control System Design and MATLAB Simulation[M]. Tsinghua University Press, 2008.
// [2]Lee T H, Harris C J. Adaptive neural network control of robotic manipulators[M]. World Scientific, 1998.

// global variables declaration
#define PI 3.14159
#define H 5              // input layer neurons number
#define IN 4             // hidden layer neurons number
#define OUT 2            // output layer neurons number
#define INPUTMODE 1      // input mode selection, 1 or 2
#define ARRAY_SIZE 10000 // sampling times

static double Ts = 0.001;                                                                                                                        // sampling period
static double t0 = 0.0;                                                                                                                          // start time
static double t1 = 10.0;                                                                                                                         // end time
static double c_D[OUT][H] = {{-1.0, -0.5, 0.0, 0.5, 1.0}, {-1.0, -0.5, 0.0, 0.5, 1.0}};                                                          // inertia matrix RBF function center
static double c_G[OUT][H] = {{-1.0, -0.5, 0.0, 0.5, 1.0}, {-1.0, -0.5, 0.0, 0.5, 1.0}};                                                          // gravity matrix RBF function center
static double c_C[IN][H] = {{-1.0, -0.5, 0.0, 0.5, 1.0}, {-2.0, -1.0, 0.0, 1.0, 2.0}, 
                            {-1.0, -0.5, 0.0, 0.5, 1.0}, {-2.0, -1.0, 0.0, 1.0, 2.0}}; // Coriolis matrix RBF function center
static double b = 10; // RBF function width
static double p[] = {2.9, 0.76, 0.87, 3.04, 0.87};
static double g = 9.8; // gravitational acceleration

double phi_D11[H], phi_D12[H], phi_D21[H], phi_D22[H];
double phi_G1[H], phi_G2[H];
double phi_C11[H], phi_C12[H], phi_C21[H], phi_C22[H];
double weight_D11[H], weight_D12[H], weight_D21[H], weight_D22[H];
double weight_G1[H], weight_G2[H];
double weight_C11[H], weight_C12[H], weight_C21[H], weight_C22[H];
double derivative_weight_D11[H], derivative_weight_D12[H], derivative_weight_D21[H], derivative_weight_D22[H];
double derivative_weight_G1[H], derivative_weight_G2[H];
double derivative_weight_C11[H], derivative_weight_C12[H], derivative_weight_C21[H], derivative_weight_C22[H];

struct _archive{
    double q1_archive[ARRAY_SIZE];
    double dq1_archive[ARRAY_SIZE];
    double q2_archive[ARRAY_SIZE];
    double dq2_archive[ARRAY_SIZE];
    double error1_archive[ARRAY_SIZE];
    double error2_archive[ARRAY_SIZE];
    double error1_velocity_archive[ARRAY_SIZE];
    double error2_velocity_archive[ARRAY_SIZE];
    double tol1_archive[ARRAY_SIZE];
    double tol2_archive[ARRAY_SIZE];
    double C_estimate_norm_archive[ARRAY_SIZE];
    double C_norm_archive[ARRAY_SIZE];
} archive;

Data qd1, dqd1, ddqd1, qd2, dqd2, ddqd2;
Data qd1_Tri, dqd1_Tri, ddqd1_Tri, qd2_Tri, dqd2_Tri, ddqd2_Tri;  // trigonometric function input

struct Amp{
    double qd1;
    double dqd1;
    double ddqd1;
    double qd2;
    double dqd2;
    double ddqd2;
};

struct M0{
    double qd1;
    double dqd1;
    double ddqd1;
    double qd2;
    double dqd2;
    double ddqd2;
};

struct B0{
    double qd1;
    double dqd1;
    double ddqd1;
    double qd2;
    double dqd2;
    double ddqd2;
};

void TrigonometricInput(Data *qd1_Tri, Data *dqd1_Tri, Data *ddqd1_Tri, Data *qd2_Tri, Data *dqd2_Tri, Data *ddqd2_Tri, double Ts, double t0, double t1){

    struct Amp amp; // amplitude
    amp.qd1 = 0.5;
    amp.dqd1 = 0.5 * PI;
    amp.ddqd1 = - 0.5 * pow(PI, 2);
    amp.qd2 = 1;
    amp.dqd2 = PI;
    amp.ddqd2 = - pow(PI, 2);

    struct M0 m0; // angular frequency
    m0.qd1 = PI;
    m0.dqd1 = PI;
    m0.ddqd1 = PI;
    m0.qd2 = PI;
    m0.dqd2 = PI;
    m0.ddqd2 = PI;

    struct B0 b0; // vertical shift
    b0.qd1 = 0.0;
    b0.dqd1 = 0.0;
    b0.ddqd1 = 0.0;
    b0.qd2 = 0.0;
    b0.dqd2 = 0.0;
    b0.ddqd2 = 0.0;

    sine(qd1_Tri, Ts, t0, t1, amp.qd1, m0.qd1, b0.qd1);         // desired angular displacement of link 1
    cosine(dqd1_Tri, Ts, t0, t1, amp.dqd1, m0.dqd1, b0.dqd1);   // desired angular velocity of link 1
    sine(ddqd1_Tri, Ts, t0, t1, amp.ddqd1, m0.ddqd1, b0.ddqd1); // desired angular acceleration of link 1
    sine(qd2_Tri, Ts, t0, t1, amp.qd2, m0.qd2, b0.qd2);         // desired angular displacement of link 2
    cosine(dqd2_Tri, Ts, t0, t1, amp.dqd2, m0.dqd2, b0.dqd2);   // desired angular velocity of link 2
    sine(ddqd2_Tri, Ts, t0, t1, amp.ddqd2, m0.ddqd2, b0.ddqd2); // desired angular acceleration of link 2
}

void Initialization(Data *output, double Ts, double t0, double t1) {
    output->size = ARRAY_SIZE;
    output->y = malloc(sizeof(double) * ARRAY_SIZE);
    double t = t0;
    int j = 0;
    while (t0 <= t1 && j < ARRAY_SIZE) {
        output->y[j] = 0.0;
        t += Ts;
        j++;
    }
}

Data qd1_Sat, dqd1_Sat, ddqd1_Sat, qd2_Sat, dqd2_Sat, ddqd2_Sat;

void SaturationInput(Data *qd1_Sat, Data *dqd1_Sat, Data *ddqd1_Sat, Data *qd2_Sat, Data *dqd2_Sat, Data *ddqd2_Sat, double Ts, double t0, double t1){

    Initialization(qd1_Sat, Ts, t0, t1);
    Initialization(dqd1_Sat, Ts, t0, t1);
    Initialization(ddqd1_Sat, Ts, t0, t1);
    Initialization(qd2_Sat, Ts, t0, t1);
    Initialization(dqd2_Sat, Ts, t0, t1);
    Initialization(ddqd2_Sat, Ts, t0, t1);

    double qd0[2] = {0, 0};
    double qdtf[2] = {1, 2};
    double td = 1.0;

    for (int j = 0; j < ARRAY_SIZE; j++){
        double time = j * Ts + t0;
        if (time >= t0 && time < td){
            qd1_Sat->y[j] = qd0[0] + (-2 * pow(time, 3) / pow(td, 3) + 3 * pow(time, 2) / pow(td, 2)) * (qdtf[0] - qd0[0]);
            qd2_Sat->y[j] = qd0[1] + (-2 * pow(time, 3) / pow(td, 3) + 3 * pow(time, 2) / pow(td, 2)) * (qdtf[1] - qd0[1]);
            
            dqd1_Sat->y[j] = (-6 * pow(time, 2) / pow(td, 3) + 6 * time / pow(td, 2)) * (qdtf[0] - qd0[0]);
            dqd2_Sat->y[j] = (-6 * pow(time, 2) / pow(td, 3) + 6 * time / pow(td, 2)) * (qdtf[1] - qd0[1]);
            
            ddqd1_Sat->y[j] = (-12 * time / pow(td, 3) + 6 / pow(td, 2)) * (qdtf[0] - qd0[0]);
            ddqd2_Sat->y[j] = (-12 * time / pow(td, 3) + 6 / pow(td, 2)) * (qdtf[1] - qd0[1]);
        } else if(time >= td && time < t1){
            qd1_Sat->y[j] = qdtf[0];
            qd2_Sat->y[j] = qdtf[1];
            
            dqd1_Sat->y[j] = 0.0;
            dqd2_Sat->y[j] = 0.0;
            
            ddqd1_Sat->y[j] = 0.0;
            ddqd2_Sat->y[j] = 0.0;
        }
    }
}

void InputModeSelection(){
    if (INPUTMODE == 1){
        TrigonometricInput(&qd1_Tri, &dqd1_Tri, &ddqd1_Tri, &qd2_Tri, &dqd2_Tri, &ddqd2_Tri, Ts, t0, t1); // system input
        Initialization(&qd1, Ts, t0, t1);
        Initialization(&dqd1, Ts, t0, t1);
        Initialization(&ddqd1, Ts, t0, t1);
        Initialization(&qd2, Ts, t0, t1);
        Initialization(&dqd2, Ts, t0, t1);
        Initialization(&ddqd2, Ts, t0, t1);
        for (int j = 0; j < ARRAY_SIZE; j++){
            qd1.y[j] = qd1_Tri.y[j];
            dqd1.y[j] = dqd1_Tri.y[j];
            ddqd1.y[j] = ddqd1_Tri.y[j];
            qd2.y[j] = qd2_Tri.y[j];
            dqd2.y[j] = dqd2_Tri.y[j];
            ddqd2.y[j] = ddqd2_Tri.y[j];
            // printf("ddqd2.y[%d] = %f\n", j, ddqd2.y[j]);
        }
    } else if(INPUTMODE == 2){
        SaturationInput(&qd1_Sat, &dqd1_Sat, &ddqd1_Sat, &qd2_Sat, &dqd2_Sat, &ddqd2_Sat, Ts, t0, t1);
        Initialization(&qd1, Ts, t0, t1);
        Initialization(&dqd1, Ts, t0, t1);
        Initialization(&ddqd1, Ts, t0, t1);
        Initialization(&qd2, Ts, t0, t1);
        Initialization(&dqd2, Ts, t0, t1);
        Initialization(&ddqd2, Ts, t0, t1);
        for (int j = 0; j < ARRAY_SIZE; j++){
            qd1.y[j] = qd1_Sat.y[j];
            dqd1.y[j] = dqd1_Sat.y[j];
            ddqd1.y[j] = ddqd1_Sat.y[j];
            qd2.y[j] = qd2_Sat.y[j];
            dqd2.y[j] = dqd2_Sat.y[j];
            ddqd2.y[j] = ddqd2_Sat.y[j];
            // printf("ddqd2.y[%d] = %f\n", j, ddqd2.y[j]);
        }
    } else{
        printf("Invalid value for INPUTMODE: %d\n", INPUTMODE);
        exit(1);
    }
}

struct _system_state{
    double q1;   // actual angular displacement of link 1
    double dq1;  // actual angular velocity of link 1
    double ddq1; // actual angular acceleration of link 1
    double q2;   // actual angular displacement of link 2
    double dq2;  // actual angular velocity of link 2
    double ddq2; // actual angular acceleration of link 2
} system_state;

struct _torque{
    double tol1;   // control torque of link 1
    double tol1_m; // Eq. 3.56 define, control law of link 1 based on model estimation
    double tol1_r; // Eq. 3.57 define, robust term of link 1 for network modeling error
    double tol2;   // control torque of link 2
    double tol2_m; // Eq. 3.56 define, control law of link 2 based on model estimation
    double tol2_r; // Eq. 3.57 define, robust term of link 2 for network modeling error
} torque;

struct _dynamics{
    double D0[OUT][OUT];            // inertia matrix for nominal model
    double G0[OUT];                 // gravity matrix for nominal model
    double C0[OUT][OUT];            // Coriolis matrix for nominal model
    double D_norm;                  // two-paradigm number of D0
    double G_norm;                  // two-paradigm number of G0
    double C_norm;                  // two-paradigm number of C0
    double DSNN_estimate[OUT][OUT]; // estimate of RBF network modeling term DSNN
    double GSNN_estimate[OUT];      // estimate of RBF network modeling term GSNN
    double CDNN_estimate[OUT][OUT]; // estimate of RBF network modeling term CDNN
    double C_estimate_norm;         // second-paradigm number of estimate of RBF network modeling term CDNN
    double inv_D0[OUT][OUT];        // inverse of inertia matrix for nominal model
} dynamics;

struct _controller{
    double controller_u1;
    double controller_u2;
    double controller_u3;
    double controller_u4;
    double controller_u5;
    double controller_u6;
    double controller_u7;
    double controller_u8;
    double controller_u9;
    double controller_u10;
    double controller_out1;
    double controller_out2;
    double controller_out3;
    double err1;                  // angular displacement error of link 1
    double err1_velocity;         // angular velocity error of link 1
    double err2;                  // angular displacement error of link 2
    double err2_velocity;         // angular velocity error of link 2
    double Lambda_D11[H][H], Lambda_D12[H][H], Lambda_D21[H][H], Lambda_D22[H][H]; // Eq. 3.60 define
    double Lambda_G1[H][H], Lambda_G2[H][H]; // Eq. 3.62 define
    double Lambda_C11[H][H], Lambda_C12[H][H], Lambda_C21[H][H], Lambda_C22[H][H]; // Eq. 3.61 define
    double Lambda[OUT][OUT];      // error's weight factor
    double r1;                    // Eq. 3.48 define
    double r2;                    // Eq. 3.48 define
    double dqr1;                  // derivative of qr1
    double dqr2;                  // derivative of qr2
    double ddqr1;                 // second-order derivative of qr1
    double ddqr2;                 // second-order derivative of qr2
    double integral1;             // integral term
    double integral2;             // integral term
    double Kr;                    // r term factor
    double Kp1;                   // proportionality factor of link 1
    double Kp2;                   // proportionality factor of link 2
    double Ki1;                   // integral factor of link 1
    double Ki2;                   // integral factor of link 2
} controller;

void CONTROL_init(){
    system_state.q1 = 0.09;
    system_state.dq1 = 0.0;
    system_state.q2 = - 0.09;
    system_state.dq2 = 0.0;
    controller.controller_u1 = qd1.y[0];
    controller.controller_u2 = dqd1.y[0];
    controller.controller_u3 = ddqd1.y[0];
    controller.controller_u4 = qd2.y[0];
    controller.controller_u5 = dqd2.y[0];
    controller.controller_u6 = ddqd2.y[0];
    controller.controller_u7 = system_state.q1;
    controller.controller_u8 = system_state.dq1;
    controller.controller_u9 = system_state.q2;
    controller.controller_u10 = system_state.dq2;
    controller.err1 = qd1.y[0] - system_state.q1;
    controller.err1_velocity = dqd1.y[0] - system_state.dq1;
    controller.err2 = qd1.y[0] - system_state.q2;
    controller.err2_velocity = dqd2.y[0] - system_state.dq2;

    for (int j = 0; j < H; j++) {
        for (int k = 0; k < H; k++) {
            if (j == k) {
                controller.Lambda_D11[j][k] = 5.0;
                controller.Lambda_D12[j][k] = 5.0;
                controller.Lambda_D21[j][k] = 5.0;
                controller.Lambda_D22[j][k] = 5.0;
            } else {
                controller.Lambda_D11[j][k] = 0.0;
                controller.Lambda_D12[j][k] = 0.0;
                controller.Lambda_D21[j][k] = 0.0;
                controller.Lambda_D22[j][k] = 0.0;
            }
        }
    }

    for (int j = 0; j < H; j++) {
        for (int k = 0; k < H; k++) {
            if (j == k) {
                controller.Lambda_G1[j][k] = 10.0;
                controller.Lambda_G2[j][k] = 10.0;
            } else {
                controller.Lambda_G1[j][k] = 0.0;
                controller.Lambda_G2[j][k] = 0.0;
            }
        }
    }

    for (int j = 0; j < H; j++) {
        for (int k = 0; k < H; k++) {
            if (j == k) {
                controller.Lambda_C11[j][k] = 10.0;
                controller.Lambda_C12[j][k] = 10.0;
                controller.Lambda_C21[j][k] = 10.0;
                controller.Lambda_C22[j][k] = 10.0;
            } else {
                controller.Lambda_C11[j][k] = 0.0;
                controller.Lambda_C12[j][k] = 0.0;
                controller.Lambda_C21[j][k] = 0.0;
                controller.Lambda_C22[j][k] = 0.0;
            }
        }
    }

    for (int j = 0; j < OUT; j++) {
        for (int k = 0; k < OUT; k++) {
            if (j == k) {
                controller.Lambda[j][k] = 5.0;
            } else {
                controller.Lambda[j][k] = 0.0;
            }
        }
    }
    
    controller.r1 = controller.err1_velocity + controller.Lambda[0][0] * controller.err1; // Eq. 3.48
    controller.r2 = controller.err2_velocity + controller.Lambda[1][1] * controller.err2; // Eq. 3.48
    controller.integral1 = 0.0;
    controller.integral2 = 0.0;
    controller.Kr = 0.10;
    controller.Kp1 = 100;
    controller.Kp2 = 100;
    controller.Ki1 = 100;
    controller.Ki2 = 100;
}

struct _plant{
    double plant_u1;
    double plant_u2;
    double plant_out1;
    double plant_out2;
    double plant_out3;
    double plant_out4;
    double plant_out5;
} plant;

void PLANT_init(){
    system_state.q1 = 0.09;
    system_state.dq1 = 0.0;
    system_state.q2 = - 0.09;
    system_state.dq2 = 0.0;
    plant.plant_u1 = 0.0;
    plant.plant_u2 = 0.0;
    plant.plant_out1 = system_state.q1;
    plant.plant_out2 = system_state.dq1;
    plant.plant_out3 = system_state.q2;
    plant.plant_out4 = system_state.dq2;
}

double PLANT_realize(int i){
    plant.plant_u1 = torque.tol1;
    plant.plant_u2 = torque.tol2;
    dynamics.D0[0][0] = p[0] + p[1] + 2 * p[2] * cos(system_state.q2);
    dynamics.D0[0][1] = p[1] + p[2] * cos(system_state.q2);
    dynamics.D0[1][0] = p[1] + p[2] * cos(system_state.q2);
    dynamics.D0[1][1] = p[1];

    dynamics.C0[0][0] = -p[2] * cos(system_state.dq2) * sin(system_state.q2);
    dynamics.C0[0][1] = -p[2] * (system_state.dq1 + system_state.dq2) * system_state.q2;
    dynamics.C0[1][0] = p[2] * system_state.dq1 * sin(system_state.q2);
    dynamics.C0[1][1] = 0;

    dynamics.G0[0] = p[3] * g * cos(system_state.q1) + p[4] * g * cos(system_state.q1 + system_state.q2);
    dynamics.G0[1] = p[4] * g * cos(system_state.q1 + system_state.q2);
    // printf("dynamics.G0 = %f\n", dynamics.G0[1]);

    inv_matrix(dynamics.inv_D0, dynamics.D0, 2);
    double to1_C0dq_G01, to1_C0dq_G02;
    to1_C0dq_G01 = torque.tol1 - (dynamics.C0[0][0] * system_state.dq1 + dynamics.C0[0][1] * system_state.dq2) - dynamics.G0[0];
    to1_C0dq_G02 = torque.tol2 - (dynamics.C0[1][0] * system_state.dq1 + dynamics.C0[1][1] * system_state.dq2) - dynamics.G0[1];

    system_state.ddq1 = dynamics.inv_D0[0][0] * to1_C0dq_G01 + dynamics.inv_D0[0][1] * to1_C0dq_G02;
    system_state.ddq2 = dynamics.inv_D0[1][0] * to1_C0dq_G01 + dynamics.inv_D0[1][1] * to1_C0dq_G02;
    system_state.dq1 = system_state.dq1 + system_state.ddq1 * Ts;
    system_state.dq2 = system_state.dq2 + system_state.ddq2 * Ts;
    system_state.q1 = system_state.q1 + system_state.dq1 * Ts;
    system_state.q2 = system_state.q2 + system_state.dq2 * Ts;

    double sum1 = 0.0;
    for (int j = 0; j < OUT; j++){
        for (int k = 0; k < OUT; k++){
            sum1 += pow(dynamics.D0[j][k], 2);
        }
    }
    dynamics.D_norm = sqrt(sum1);

    double sum2 = 0.0;
    for (int j = 0; j < OUT; j++){
        for (int k = 0; k < OUT; k++){
            sum2 += pow(dynamics.C0[j][k], 2);
        }
    }
    dynamics.C_norm = sqrt(sum2);

    double sum3 = 0.0;
    for (int j = 0; j < OUT; j++){
        sum3 += pow(dynamics.G0[j], 2);
    }
    dynamics.G_norm = sqrt(sum3);

    plant.plant_out1 = system_state.q1;
    plant.plant_out2 = system_state.dq1;
    plant.plant_out3 = system_state.q2;
    plant.plant_out4 = system_state.dq2;
    plant.plant_out5 = dynamics.C_norm;
    archive.C_norm_archive[i] = plant.plant_out5;
}

double CONTROL_realize(int i){
    controller.controller_u1 = qd1.y[i];
    controller.controller_u2 = dqd1.y[i];
    controller.controller_u3 = ddqd1.y[i];
    controller.controller_u4 = qd2.y[i];
    controller.controller_u5 = dqd2.y[i];
    controller.controller_u6 = ddqd2.y[i];
    controller.controller_u7 = system_state.q1;
    controller.controller_u8 = system_state.dq1;
    controller.controller_u9 = system_state.q2;
    controller.controller_u10 = system_state.dq2;
    // printf("system_state.q1 = %f\n", system_state.q1);
    archive.q1_archive[i] = controller.controller_u7;
    archive.dq1_archive[i] = controller.controller_u8;
    archive.q2_archive[i] = controller.controller_u9;
    archive.dq2_archive[i] = controller.controller_u10;

    for (int j = 0; j < H; j++){
        phi_D11[j] = exp( (-pow(controller.controller_u7 - c_D[0][j], 2) - pow(controller.controller_u9 - c_D[1][j], 2) ) / (b * b)); // output of the inertia matrix RBF function
        phi_D12[j] = exp( (-pow(controller.controller_u7 - c_D[0][j], 2) - pow(controller.controller_u9 - c_D[1][j], 2) ) / (b * b));
        phi_D21[j] = exp( (-pow(controller.controller_u7 - c_D[0][j], 2) - pow(controller.controller_u9 - c_D[1][j], 2) ) / (b * b));
        phi_D22[j] = exp( (-pow(controller.controller_u7 - c_D[0][j], 2) - pow(controller.controller_u9 - c_D[1][j], 2) ) / (b * b));
    }
    for (int j = 0; j < H; j++){
        phi_G1[j] = exp( (-pow(controller.controller_u7 - c_G[0][j], 2) - pow(controller.controller_u9 - c_G[1][j], 2) ) / (b * b)); // output of the gravity matrix RBF function
        phi_G2[j] = exp( (-pow(controller.controller_u7 - c_G[0][j], 2) - pow(controller.controller_u9 - c_G[1][j], 2) ) / (b * b));
    }
    for (int j = 0; j < H; j++){
        phi_C11[j] = exp( (-pow(controller.controller_u7 - c_C[0][j], 2) - pow(controller.controller_u9 - c_C[1][j], 2)
                           -pow(controller.controller_u8 - c_C[2][j], 2) - pow(controller.controller_u10 - c_C[3][j], 2)) / (b * b)); // output of the Coriolis matrix RBF function
        phi_C12[j] = exp( (-pow(controller.controller_u7 - c_C[0][j], 2) - pow(controller.controller_u9 - c_C[1][j], 2)
                           -pow(controller.controller_u8 - c_C[2][j], 2) - pow(controller.controller_u10 - c_C[3][j], 2)) / (b * b));
        phi_C21[j] = exp( (-pow(controller.controller_u7 - c_C[0][j], 2) - pow(controller.controller_u9 - c_C[1][j], 2)
                           -pow(controller.controller_u8 - c_C[2][j], 2) - pow(controller.controller_u10 - c_C[3][j], 2)) / (b * b));
        phi_C22[j] = exp( (-pow(controller.controller_u7 - c_C[0][j], 2) - pow(controller.controller_u9 - c_C[1][j], 2)
                           -pow(controller.controller_u8 - c_C[2][j], 2) - pow(controller.controller_u10 - c_C[3][j], 2)) / (b * b));    
        // printf("phi_C11[%d] = %f\n", j, phi_C11[j]);
    }

    for (int j = 0; j < H; j++){
        weight_D11[j] = 0.0; // inertia matrix RBF network weight
        weight_D12[j] = 0.0;
        weight_D21[j] = 0.0;
        weight_D22[j] = 0.0;
        weight_G1[j] = 0.0;  // gravity matrix RBF network weight
        weight_G2[j] = 0.0;
        weight_C11[j] = 0.0; // Coriolis matrix RBF network weight
        weight_C12[j] = 0.0;
        weight_C21[j] = 0.0;
        weight_C22[j] = 0.0;
    }

    controller.err1 = qd1.y[i] - system_state.q1;
    controller.err1_velocity = dqd1.y[i] - system_state.dq1;
    controller.err2 = qd2.y[i] - system_state.q2;
    controller.err2_velocity = dqd2.y[i] - system_state.dq2;
    archive.error1_archive[i] = controller.err1;
    archive.error1_velocity_archive[i] = controller.err1_velocity;
    archive.error2_archive[i] = controller.err2;
    archive.error2_velocity_archive[i] = controller.err2_velocity;
    // printf("controller.err1 = %f\n", controller.err1);
    controller.r1 = controller.err1_velocity + controller.Lambda[0][0] * controller.err1;
    controller.dqr1 = dqd1.y[i] + controller.Lambda[0][0] * controller.err1;
    controller.ddqr1 = ddqd1.y[i] + controller.Lambda[0][0] * controller.err1_velocity;
    controller.r2 = controller.err2_velocity + controller.Lambda[1][1] * controller.err2;
    controller.dqr2 = dqd2.y[i] + controller.Lambda[1][1] * controller.err2;
    controller.ddqr2 = ddqd2.y[i] + controller.Lambda[1][1] * controller.err2_velocity;
    // printf("controller.qr1 = %f\n", controller.qr1);

    // adaptive law
    for (int j = 0; j < H; j++){
        derivative_weight_D11[j] = controller.Lambda_D11[j][j] * phi_D11[j] * controller.ddqr1 * controller.r1; // Eq. 3.60
        derivative_weight_D12[j] = controller.Lambda_D12[j][j] * phi_D12[j] * controller.ddqr2 * controller.r1;
        derivative_weight_D21[j] = controller.Lambda_D21[j][j] * phi_D21[j] * controller.ddqr1 * controller.r2;
        derivative_weight_D22[j] = controller.Lambda_D22[j][j] * phi_D22[j] * controller.ddqr2 * controller.r2;
    }
    for (int j = 0; j < H; j++){
        derivative_weight_G1[j] = controller.Lambda_G1[j][j] * phi_G1[j] * controller.r1; // Eq. 3.62
        derivative_weight_G2[j] = controller.Lambda_G2[j][j] * phi_G2[j] * controller.r2;
    }
    for (int j = 0; j < H; j++){
        derivative_weight_C11[j] = controller.Lambda_C11[j][j] * phi_C11[j] * controller.dqr1 * controller.r1; // Eq. 3.61
        derivative_weight_C12[j] = controller.Lambda_C12[j][j] * phi_C12[j] * controller.ddqr2 * controller.r1;
        derivative_weight_C21[j] = controller.Lambda_C21[j][j] * phi_C21[j] * controller.dqr1 * controller.r2;
        derivative_weight_C22[j] = controller.Lambda_C22[j][j] * phi_C22[j] * controller.ddqr2 * controller.r2;
        // printf("derivative_weight_C11[%d] = %f\n", j, derivative_weight_C11[j]);
    }

    for (int j = 0; j < H; j++){
        weight_D11[j] = weight_D11[j] + derivative_weight_D11[j] * Ts;
        weight_D12[j] = weight_D12[j] + derivative_weight_D12[j] * Ts;
        weight_D21[j] = weight_D21[j] + derivative_weight_D21[j] * Ts;
        weight_D11[j] = weight_D11[j] + derivative_weight_D11[j] * Ts;
    }
    for (int j = 0; j < H; j++){
        weight_G1[j] = weight_G1[j] + derivative_weight_G1[j] * Ts;
        weight_G2[j] = weight_G2[j] + derivative_weight_G2[j] * Ts;
    }
    for (int j = 0; j < H; j++){
        weight_C11[j] = weight_C11[j] + derivative_weight_C11[j] * Ts;
        weight_C12[j] = weight_C12[j] + derivative_weight_C12[j] * Ts;
        weight_C21[j] = weight_C21[j] + derivative_weight_C21[j] * Ts;
        weight_C22[j] = weight_C22[j] + derivative_weight_C22[j] * Ts;
    }

    controller.integral1 += controller.r1;
    controller.integral2 += controller.r2;

    for (int j = 0; j < H; j++){
        dynamics.DSNN_estimate[0][0] = weight_D11[j] * phi_D11[j];  // Eq. 3.51
        dynamics.DSNN_estimate[0][1] = weight_D12[j] * phi_D12[j];
        dynamics.DSNN_estimate[1][0] = weight_D21[j] * phi_D21[j];
        dynamics.DSNN_estimate[1][1] = weight_D22[j] * phi_D22[j];
    }
    for (int j = 0; j < H; j++){
        dynamics.GSNN_estimate[0] = weight_G1[j] * phi_G1[j];  // Eq. 3.53
        dynamics.GSNN_estimate[1] = weight_G2[j] * phi_G2[j];
    }
    for (int j = 0; j < H; j++){
        dynamics.CDNN_estimate[0][0] = weight_C11[j] * phi_C11[j];  // Eq. 3.52
        dynamics.CDNN_estimate[0][1] = weight_C12[j] * phi_C12[j];
        dynamics.CDNN_estimate[1][0] = weight_C21[j] * phi_C21[j];
        dynamics.CDNN_estimate[1][1] = weight_C22[j] * phi_C22[j];
    }

    double sum = 0.0;
    for (int j = 0; j < OUT; j++){
        for (int k = 0; k < OUT; k++){
            sum += pow(dynamics.CDNN_estimate[j][k], 2);
        }
    }
    dynamics.C_estimate_norm = sqrt(sum);
    torque.tol1_m = dynamics.DSNN_estimate[0][0] * controller.ddqr1 + dynamics.DSNN_estimate[0][1] * controller.ddqr2
     + dynamics.CDNN_estimate[0][0] * controller.dqr1 + dynamics.CDNN_estimate[0][1] * controller.dqr2
     + dynamics.GSNN_estimate[0]; // Eq. 3.56, control law based on model estimation
    torque.tol2_m = dynamics.DSNN_estimate[1][0] * controller.ddqr1 + dynamics.DSNN_estimate[1][1] * controller.ddqr2
     + dynamics.CDNN_estimate[1][0] * controller.dqr1 + dynamics.CDNN_estimate[1][1] * controller.dqr2
     + dynamics.GSNN_estimate[1];

    if (controller.r1 >= 0)
        torque.tol1_r = controller.Kr;
    else
        torque.tol1_r = -controller.Kr;
    if (controller.r2 >= 0)
        torque.tol2_r = controller.Kr;
    else
        torque.tol2_r = -controller.Kr;

    torque.tol1 = torque.tol1_m + controller.Kp1 * controller.r1 + controller.Ki1 * controller.integral1 + torque.tol1_r; // Eq. 3.55, control law
    torque.tol2 = torque.tol2_m + controller.Kp2 * controller.r2 + controller.Ki2 * controller.integral2 + torque.tol2_r;
    archive.tol1_archive[i] = torque.tol1;
    archive.tol2_archive[i] = torque.tol2;
    controller.controller_out1 = torque.tol1;
    controller.controller_out2 = torque.tol2;
    controller.controller_out3 = dynamics.C_estimate_norm;
    archive.C_estimate_norm_archive[i] = controller.controller_out3;
}

void saveArchiveToTxt(double *archive, int size, const char *filename) {

    FILE *file = fopen(filename, "w+");

    if (file == NULL) {
        perror("Failed to open file");
        exit(1);
    } else {
        for (int i = 0; i < size; i++) {
            fprintf(file, "%lf\n", archive[i]);
        }
        fclose(file);
        printf("Saved to file %s\n", filename);
    }
}

void saveArchive(){
    if (INPUTMODE == 1){
        saveArchiveToTxt(qd1.y, ARRAY_SIZE, "../TrigonometricInput/qd1.txt");
        saveArchiveToTxt(archive.q1_archive, ARRAY_SIZE, "../TrigonometricInput/q1.txt");
        saveArchiveToTxt(archive.dq1_archive, ARRAY_SIZE, "../TrigonometricInput/dq1.txt");
        saveArchiveToTxt(qd2.y, ARRAY_SIZE, "../TrigonometricInput/qd2.txt");
        saveArchiveToTxt(archive.q2_archive, ARRAY_SIZE, "../TrigonometricInput/q2.txt");
        saveArchiveToTxt(archive.dq2_archive, ARRAY_SIZE, "../TrigonometricInput/dq2.txt");
        saveArchiveToTxt(archive.error1_archive, ARRAY_SIZE, "../TrigonometricInput/error1.txt");
        saveArchiveToTxt(archive.error1_velocity_archive, ARRAY_SIZE, "../TrigonometricInput/error1_velocity.txt");
        saveArchiveToTxt(archive.error2_archive, ARRAY_SIZE, "../TrigonometricInput/error2.txt");
        saveArchiveToTxt(archive.error2_velocity_archive, ARRAY_SIZE, "../TrigonometricInput/error2_velocity.txt");
        saveArchiveToTxt(archive.tol1_archive, ARRAY_SIZE, "../TrigonometricInput/tol1.txt");
        saveArchiveToTxt(archive.tol2_archive, ARRAY_SIZE, "../TrigonometricInput/tol2.txt");
        saveArchiveToTxt(archive.C_estimate_norm_archive, ARRAY_SIZE, "../TrigonometricInput/C_estimate_norm_archive.txt");
        saveArchiveToTxt(archive.C_norm_archive, ARRAY_SIZE, "../TrigonometricInput/C_norm_archive.txt");
    } else if(INPUTMODE == 2){
        saveArchiveToTxt(qd1.y, ARRAY_SIZE, "../SaturationInput/qd1.txt");
        saveArchiveToTxt(archive.q1_archive, ARRAY_SIZE, "../SaturationInput/q1.txt");
        saveArchiveToTxt(archive.dq1_archive, ARRAY_SIZE, "../SaturationInput/dq1.txt");
        saveArchiveToTxt(qd2.y, ARRAY_SIZE, "../SaturationInput/qd2.txt");
        saveArchiveToTxt(archive.q2_archive, ARRAY_SIZE, "../SaturationInput/q2.txt");
        saveArchiveToTxt(archive.dq2_archive, ARRAY_SIZE, "../SaturationInput/dq2.txt");
        saveArchiveToTxt(archive.error1_archive, ARRAY_SIZE, "../SaturationInput/error1.txt");
        saveArchiveToTxt(archive.error1_velocity_archive, ARRAY_SIZE, "../SaturationInput/error1_velocity.txt");
        saveArchiveToTxt(archive.error2_archive, ARRAY_SIZE, "../SaturationInput/error2.txt");
        saveArchiveToTxt(archive.error2_velocity_archive, ARRAY_SIZE, "../SaturationInput/error2_velocity.txt");
        saveArchiveToTxt(archive.tol1_archive, ARRAY_SIZE, "../SaturationInput/tol1.txt");
        saveArchiveToTxt(archive.tol2_archive, ARRAY_SIZE, "../SaturationInput/tol2.txt");
        saveArchiveToTxt(archive.C_estimate_norm_archive, ARRAY_SIZE, "../SaturationInput/C_estimate_norm_archive.txt");
        saveArchiveToTxt(archive.C_norm_archive, ARRAY_SIZE, "../SaturationInput/C_norm_archive.txt");
    } else{
        printf("Invalid value for INPUTMODE: %d\n", INPUTMODE);
        exit(1);
    }
}

int main(){

    InputModeSelection();
    CONTROL_init(); // initialize controller parameter
    PLANT_init();   // initialize plant parameter

    for (int i = 0; i < ARRAY_SIZE; i++){
        double time = i * Ts + t0;
        printf("time at step %d: %f\n", i, time);
        CONTROL_realize(i);
        PLANT_realize(i);
    }

    saveArchive();

    return 0;
}
