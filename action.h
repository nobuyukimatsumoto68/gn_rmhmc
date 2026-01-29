#pragma once

/*
  Action objects should have:

  double operator()( const Gauge& W ) const;
  Force d( const Gauge& W ) const;
*/



struct WilsonGaussianAndDet {
  using Force = ForceField<ForceSingleLink>;
  using Gauge = GaugeField;
  // using M = LinkConfig;
  using V = ForceSingleLink;

  // static constexpr int dim = 2;
  // using Idx = std::size_t;

  // using Complex = std::complex<double>;
  using MC = Eigen::Matrix<Complex, Nc, Nc, Eigen::RowMajor>;

  const Lattice& lattice;
  const double beta;
  const double lambda;
  const double kappa;
  const double c;

  const Generators t;

  WilsonGaussianAndDet(const Lattice& lattice,
			 const double beta_,
			 const double lambda_,
			 const double kappa_ ,
			 const double c_=1.0)
    : lattice(lattice)
    , beta(beta_)
    , lambda(lambda_)
    , kappa(kappa_)
    , c(c_)
    , t()
  {}

  MC plaq( const Gauge& W, const Idx ix, const int mu, const int nu ) const {
    const Coord x = lattice.get_coord(ix);
    const Coord xpmu = lattice.cshift(x, mu);
    const Coord xpnu = lattice.cshift(x, nu);
    return W(x,mu).U * W(xpmu,nu).U * W(xpnu,mu).U.adjoint() * W(x,nu).U.adjoint();
  }

  MC staples( const Gauge& W, const Idx ix, const int mu, const int nu ) const {
    // assert( 0<=mu && mu<2);
    // const int nu = 1-mu;
    const Coord x = lattice.get_coord(ix);
    const Coord x_pmu = lattice.cshift(x, mu);
    const Coord x_pnu = lattice.cshift(x, nu);
    const Coord x_pmu_mnu = lattice.cshift(x_pmu, -nu-1);
    const Coord x_mnu = lattice.cshift(x, -nu-1);

    MC res = W(x_pmu, nu).U * W(x_pnu, mu).U.adjoint() * W(x, nu).U.adjoint();
    res += W(x_pmu_mnu, nu).U.adjoint() * W(x_mnu, mu).U.adjoint() * W(x_mnu, nu).U;
    return res;
  }

  double operator()( const Gauge& W ) const {
    double res = 0.0;
    std::vector<double> tmp( W.lattice.vol, 0.0 );
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(Idx ix=0; ix<lattice.vol; ix++) {
      for(int mu=0; mu<DIM; mu++){
        for(int nu=mu+1; nu<DIM; nu++){
          tmp[ix] -= beta/Nc * plaq(W, ix, mu, nu).trace().real();
        }
      tmp[ix] += 0.5*lambda/Nc * ( W(ix,mu).Phi - c*W(ix,mu).id() ).squaredNorm();
      tmp[ix] -= kappa/Nc * std::log( W(ix,mu).Phi.determinant().real() );
      }
    }
    for(auto elem : tmp ) res += elem;
    return res;
  }

  double D( const Gauge& W, const Idx ix, const int mu, const int a ) const {
    double res = 0.0;
    for(int nu=0; nu<DIM; nu++){
      if(mu==nu) continue;
      res += ( t[a] * W(ix,mu).U * staples( W, ix, mu, nu ) ).trace().imag();
    }
    res *= beta/Nc;
    return res;
  }

  double dphi( const Gauge& W, const Idx ix, const int mu, const int a ) const {
    double res = ( W(ix,mu).Phi * t[a] ).trace().real();
    res *= lambda/Nc;
    res -= kappa/Nc * (W(ix,mu).Phi.inverse()*t[a]).trace().real();
    return res;
  }

  double dphi0( const Gauge& W, const Idx ix, const int mu ) const {
    double res = ( W(ix,mu).Phi - c*W(ix,mu).id() ).trace().real();
    res *= lambda/Nc;
    res -= kappa/Nc * W(ix,mu).Phi.inverse().trace().real();
    return res;
  }

  V d( const Gauge& W, const Idx ix, const int mu ) const {
    V dSb;
    for(int a=0; a<NA; a++) dSb[a] = D(W,ix,mu,a);
    for(int a=0; a<NA; a++) dSb[Nc*Nc+a] = dphi(W,ix,mu,a);
    dSb[2*Nc*Nc-1] = dphi0(W,ix,mu);

    // dSb.Lmult( W(ix,mu).J().inverse() );
    dSb.invert( W(ix,mu).J() );
    return dSb;
  }

  Force d( const Gauge& W ) const {
    Force res(lattice);
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(Idx ix=0; ix<lattice.n_sites(); ix++) {
      for(int mu=0; mu<DIM; mu++){
	res(ix,mu) = d(W,ix,mu);
      }}
    return res;
  }

};






// struct WilsonGaussianAndDet2 {
//   using Force = LinkForce;
//   using Gauge = LinkConf;

//   using Complex = std::complex<double>;
//   // using MC = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
//   using MR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
//   // using VC = Eigen::VectorXcd;
//   using VR = Eigen::VectorXd;

//   const double beta;
//   const double lambda;
//   const double kappa;

//   WilsonGaussianAndDet2(const double beta_,
//                        const double lambda_,
//                        const double kappa_
//                        )
//     : beta(beta_)
//     , lambda(lambda_)
//     , kappa(kappa_)
//   {}

//   double operator()( const Gauge& W ) const {
//     double res = 0.0;
//     res -= beta/Nc * ( W.U ).trace().real();
//     res += 0.5*lambda/Nc * ( W.Phi - W.id() ).squaredNorm();
//     res -= kappa/Nc * std::log( W.Phi.determinant().real() );
//     return res;
//   }

//   double Da( const Gauge& W, const int a ) const {
//     double res = beta/Nc * ( W.t[a]*W.U ).trace().imag();
//     return res;
//   }

//   double dphia( const Gauge& W, const int a ) const {
//     double res = ( W.Phi * W.t[a] ).trace().real();
//     res *= lambda/Nc;
//     res -= kappa/Nc * (W.Phi.inverse()*W.t[a]).trace().real();
//     return res;
//   }

//   double dphi0( const Gauge& W ) const {
//     double res = ( W.Phi - W.id() ).trace().real();
//     res *= lambda/Nc;
//     res -= kappa/Nc * W.Phi.inverse().trace().real();
//     return res;
//   }

//   Force d( const Gauge& W ) const {
//     Force res;
//     for(int a=0; a<NA; a++) res.pi(a) = Da(W,a);
//     for(int a=0; a<NA; a++) res.rho(a) = dphia(W,a);
//     res.rho0 = dphi0(W);
//     return res;
//   }

// };

































// struct GaussianAction {
//   using Force = ForceSingleLink;
//   using Gauge = LinkConfig;

//   using Complex = std::complex<double>;
//   // using MC = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
//   // using MR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
//   // using VC = Eigen::VectorXcd;
//   // using VR = Eigen::VectorXd;
//   using MR = Eigen::Matrix<double, Nc, Nc, Eigen::RowMajor>;
//   // using MG = Eigen::Matrix<double, NG, NG, Eigen::RowMajor>;
//   using VG = Eigen::Matrix<double, NG, 1>;
//   using VH = Eigen::Matrix<double, NH, 1>;


//   const double beta;

//   GaussianAction(const double beta_)
//     : beta(beta_)
//   {}

//   double operator()( const Gauge& W ) const {
//     double res = 0.0;
//     res += W().squaredNorm();
//     res *= 0.5*beta/Nc;
//     return res;
//   }

//   inline MR dx( const Gauge& W ) const { return beta/Nc * W().real(); }
//   inline MR dy( const Gauge& W ) const { return beta/Nc * W().imag(); }

//   Force d( const Gauge& W ) const {
//     const MR m_dx = dx(W);
//     const MR m_dy = dy(W);

//     VG res = VG::Zero();
//     res.segment(0, Nc*Nc) = Eigen::Map<const VH>( m_dx.data() );
//     res.segment(Nc*Nc, Nc*Nc) = Eigen::Map<const VH>( m_dy.data() );
//     return Force(res);
//   }

// };


// // struct GaussianPhiAction {
// //   using Force = ForceSingleLink;
// //   using Gauge = LinkConfig;

// //   using Complex = std::complex<double>;
// //   // using MC = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
// //   using MR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
// //   // using VC = Eigen::VectorXcd;
// //   using VR = Eigen::VectorXd;


// //   const double lambda;

// //   GaussianPhiAction(const double lambda_)
// //     : lambda(lambda_)
// //   {}

// //   double operator()( const Gauge& W ) const {
// //     double res = 0.0;
// //     res += ( W.Phi - W.id() ).squaredNorm();
// //     res *= 0.5*lambda/Nc;
// //     return res;
// //   }

// //   double dphia( const Gauge& W, const int a ) const {
// //     double res = ( W.Phi * W.t[a] ).trace().real();
// //     res *= lambda/Nc;
// //     return res;
// //   }

// //   double dphi0( const Gauge& W ) const {
// //     double res = ( W.Phi - W.id() ).trace().real();
// //     res *= lambda/Nc;
// //     return res;
// //   }

// //   Force d( const Gauge& W ) const {
// //     // const int Nc = W.Nc;

// //     VR dSb = VR::Zero(2*Nc*Nc);
// //     for(int a=0; a<NA; a++) dSb(Nc*Nc+a) = dphia(W,a);
// //     dSb(2*Nc*Nc-1) = dphi0(W);

// //     VR res = W.J().inverse() * dSb;
// //     return Force(res);
// //   }

// // };



// // struct WilsonGaussianAction {
// //   using Force = ForceSingleLink;
// //   using Gauge = LinkConfig;

// //   using Complex = std::complex<double>;
// //   // using MR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
// //   // using VR = Eigen::VectorXd;
// //   using MG = Eigen::Matrix<double, NG, NG, Eigen::RowMajor>;
// //   using MH = Eigen::Matrix<double, NH, NH, Eigen::RowMajor>;
// //   using VG = Eigen::Matrix<double, NG, 1>;
// //   using VH = Eigen::Matrix<double, NH, 1>;

// //   const double beta;
// //   const double lambda;

// //   WilsonGaussianAction(const double beta_,
// //                        const double lambda_)
// //     : beta(beta_)
// //     , lambda(lambda_)
// //   {}

// //   double operator()( const Gauge& W ) const {
// //     double res = 0.0;
// //     res -= beta/Nc * ( W.U ).trace().real();
// //     res += 0.5*lambda/Nc * ( W.Phi - W.id() ).squaredNorm();
// //     return res;
// //   }

// //   double Da( const Gauge& W, const int a ) const {
// //     double res = beta/Nc * ( W.t[a]*W.U ).trace().imag();
// //     return res;
// //   }

// //   double dphia( const Gauge& W, const int a ) const {
// //     double res = ( W.Phi * W.t[a] ).trace().real();
// //     res *= lambda/Nc;
// //     return res;
// //   }

// //   double dphi0( const Gauge& W ) const {
// //     double res = ( W.Phi - W.id() ).trace().real();
// //     res *= lambda/Nc;
// //     return res;
// //   }

// //   Force d( const Gauge& W ) const {
// //     VG dSb = VG::Zero();
// //     for(int a=0; a<NA; a++) dSb(a) = Da(W,a);
// //     for(int a=0; a<NA; a++) dSb(Nc*Nc+a) = dphia(W,a);
// //     dSb(2*Nc*Nc-1) = dphi0(W);

// //     const VG res = W.J().inverse() * dSb;
// //     return Force(res);
// //   }

// // };


// // struct WilsonGaussianAndDet {
// //   using Force = ForceSingleLink;
// //   using Gauge = LinkConfig;

// //   using Complex = std::complex<double>;
// //   // using MC = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
// //   using MR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
// //   // using VC = Eigen::VectorXcd;
// //   using VR = Eigen::VectorXd;


// //   const double beta;
// //   const double lambda;
// //   const double kappa;

// //   WilsonGaussianAndDet(const double beta_,
// //                        const double lambda_,
// //                        const double kappa_
// //                        )
// //     : beta(beta_)
// //     , lambda(lambda_)
// //     , kappa(kappa_)
// //   {}

// //   double operator()( const Gauge& W ) const {
// //     double res = 0.0;
// //     res -= beta/Nc * ( W.U ).trace().real();
// //     res += 0.5*lambda/Nc * ( W.Phi - W.id() ).squaredNorm();
// //     res -= kappa * std::log( W.Phi.determinant().real() );
// //     return res;
// //   }

// //   double Da( const Gauge& W, const int a ) const {
// //     double res = beta/Nc * ( W.t[a]*W.U ).trace().imag();
// //     return res;
// //   }

// //   double dphia( const Gauge& W, const int a ) const {
// //     double res = ( W.Phi * W.t[a] ).trace().real();
// //     res *= lambda/Nc;
// //     res -= kappa * (W.Phi.inverse()*W.t[a]).trace().real();
// //     return res;
// //   }

// //   double dphi0( const Gauge& W ) const {
// //     double res = ( W.Phi - W.id() ).trace().real();
// //     res *= lambda/Nc;
// //     res -= kappa * W.Phi.inverse().trace().real();
// //     return res;
// //   }

// //   Force d( const Gauge& W ) const {
// //     // const int Nc = W.Nc;

// //     VR dSb = VR::Zero(2*Nc*Nc);
// //     for(int a=0; a<NA; a++) dSb(a) = Da(W,a);
// //     for(int a=0; a<NA; a++) dSb(Nc*Nc+a) = dphia(W,a);
// //     dSb(2*Nc*Nc-1) = dphi0(W);

// //     VR res = W.J().inverse() * dSb;
// //     return Force(res);
// //   }

// // };
