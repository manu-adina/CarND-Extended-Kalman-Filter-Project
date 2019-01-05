#include "kalman_filter.h"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
	x_ =  F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;	
}

void KalmanFilter::Update(const VectorXd &z) {
	// From Udacity's material on Kalman Filters.
	VectorXd y = z - H_ * x_;
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd K = P_ * Ht * Si;
	
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
 	
	// To perform the measurement update, need to convert X, Y to Cartesian measurement space.
	double px = x_(0);
	double py = x_(1);
	double vx = x_(2);
	double vy = x_(3);

	// From Udacity's material on Radar measurements.
	double rho = sqrt(px*px + py*py);
	double phi = atan2(py , px);
	double rho_dot = (px*vx + py*vy) / rho;

	VectorXd H_conv = VectorXd(3);
	H_conv << rho, phi, rho_dot;

	// Difference between the predicted and measured.
	VectorXd y = z - H_conv;

	// Convert to 0-2pi domain.
	if(y(1) > M_PI) y(1) -= 2*M_PI;
	if(y(1) < -M_PI) y(1) += 2*M_PI;

	// From Udacity's material on Kalman Filters.
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd K = P_ * Ht * Si;	

	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}