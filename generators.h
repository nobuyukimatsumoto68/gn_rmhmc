#pragma once

#include <Eigen/Dense>

struct Generators {
  using Complex = std::complex<double>;
  static constexpr Complex I = Complex(0.0, 1.0);

  using MC = Eigen::Matrix<Complex, Nc, Nc, Eigen::RowMajor>;

  std::vector<MC> t; // generators; tr(TaTb) = delta_{ab}

  Generators()
  {
    for(int i=0; i<Nc; i++){
      for(int j=i+1; j<Nc; j++){
	{
	  MC tmp = MC::Zero();
	  tmp(i,j) = 1.0;
	  tmp(j,i) = 1.0;
	  t.push_back(tmp/std::sqrt(2.0));
	}
	{
	  MC tmp = MC::Zero();
	  tmp(i,j) = -I;
	  tmp(j,i) =  I;
	  t.push_back(tmp/std::sqrt(2.0));
	}
      }}

    for(int m=1; m<Nc; m++){
      MC tmp = MC::Zero();
      for(int i=0; i<Nc; i++){
	if(i<m) tmp(i,i) = 1.0;
	else if(i==m) tmp(i,i) = -m;
      }
      t.push_back( tmp/std::sqrt(m*(m+1.0)) );
    }
    // for( auto& elem : t ) elem *= 1.0/std::sqrt(2.0);
  }

  inline MC operator[](const int a) const { return t[a]; }
};

