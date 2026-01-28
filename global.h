#pragma once

#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <iomanip>


using Complex = std::complex<double>;
constexpr Complex I = Complex(0.0, 1.0);

using MC = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VC = Eigen::VectorXcd;
using VR = Eigen::VectorXd;


std::ostream& operator<<(std::ostream& os, const MR& W) {
  os << std::scientific << std::setprecision(15);
  for(int i=0; i<W.rows(); i++){
    for(int j=0; j<W.rows(); j++){
      os << std::setw(22) << W(i,j) << " ";
    }
    os << std::endl;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const MC& W) {
  os << std::scientific << std::setprecision(15);
  for(int i=0; i<W.rows(); i++){
    for(int j=0; j<W.rows(); j++){
      os << std::setw(22) << W(i,j).real() << " "
	 << std::setw(22) << W(i,j).imag() << " ";
    }
    os << std::endl;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const VR& W) {
  os << std::scientific << std::setprecision(15);
  for(int i=0; i<W.size(); i++){
    os << std::setw(22) << W(i) << " ";
  }
  os << std::endl;
  return os;
}

std::ostream& operator<<(std::ostream& os, const VC& W) {
  os << std::scientific << std::setprecision(15);
  for(int i=0; i<W.size(); i++){
    os << std::setw(22) << W(i).real() << " "
       << std::setw(22) << W(i).imag() << " ";
  }
  os << std::endl;
  return os;
}
