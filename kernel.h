#pragma once

/*
  Kernel objects should have:

  double operator()( const Force& pi, const Gauge& W ) const; // G(U) * pi
  Force d( const Force& pi, const Gauge& W ) const;
  double det( const Gauge& W ) const; // det G(U)
  Force det_log_d( const Gauge& W ) const; // d [log det G(U)]

  Force gen( Rng& rng ) const ;
*/


struct IdpWHW {
  using Force = ForceSingleLink;
  using Gauge = LinkConfig;
  using Rng = SingleRng;

  using Complex = std::complex<double>;
  using MR = Eigen::Matrix<double, Nc, Nc, Eigen::RowMajor>;
  using VG = Eigen::Matrix<double, NG, 1>;
  using VH = Eigen::Matrix<double, NH, 1>;
  using MG = Eigen::Matrix<double, NG, NG, Eigen::RowMajor>;

  const double alpha;

  IdpWHW( const double alpha )
    : alpha(alpha)
  {};

  MG wiwj( const Gauge& W ) const {
    const MR Re = W().real(); // to avoid bugs of Eigen; no const, no .real().data()
    const MR Im = W().imag(); // to avoid bugs of Eigen; no const, no .real().data()

    VG w;
    w.segment(0, Nc*Nc) = Eigen::Map<const VH> ( Re.data() );
    w.segment(Nc*Nc, Nc*Nc) = Eigen::Map<const VH> ( Im.data() );

    MG res = MG::Zero();
    for(int i=0; i<NG; i++) for(int j=0; j<NG; j++) res(i,j) = w(i)*w(j);
    return res;
  }

  MG wiwj_d( const Gauge& W, const int i ) const {
    const MR Re = W().real(); // to avoid bugs of Eigen; no const, no .real().data()
    const MR Im = W().imag(); // to avoid bugs of Eigen; no const, no .real().data()

    VG w;
    w.segment(0, Nc*Nc) = Eigen::Map<const VH> ( Re.data() );
    w.segment(Nc*Nc, Nc*Nc) = Eigen::Map<const VH> ( Im.data() );

    MG res = MG::Zero();
    for(int j=0; j<NG; j++) res(i,j) += w(j);
    for(int j=0; j<NG; j++) res(j,i) += w(j);
    return res;
  }

  MG matrix( const Gauge& W ) const {
    const MG A = MG::Identity() + alpha*wiwj( W );
    return A.transpose() * A;
  }

  MG matrix_d( const Gauge& W, const int i ) const {
    const MG A = MG::Identity() + alpha*wiwj( W );
    const MG dA = alpha*wiwj_d( W, i );
    return dA.transpose() * A + A.transpose() * dA;
  }

  inline double operator()( const Force& f, const Gauge& W ) const { return f.pi.dot( matrix(W)*f.pi ); }

  Force d( const Force& f, const Gauge& W ) const {
    Force res;
    for(int i=0; i<NG; i++) res[i] = f.pi.dot( matrix_d(W, i)*f.pi );
    return res;
  }

  inline Force act( const Gauge& W, const Force& f ) const { return matrix(W)*f; }

  inline double logdet( const Gauge& W ) const { return std::log( matrix(W).determinant() ); }

  Force logdet_d( const Gauge& W ) const {
    Force res;
    const MG inv = matrix(W).inverse();
    for(int i=0; i<NG; i++) res[i] = ( inv * matrix_d(W, i) ).trace();
    return res;
  }

  Force gen( const Gauge& W, Rng& rng ) const {
    Force z;
    for(int i=0; i<NG; i++) z[i] = rng.gaussian();
    const Force res = matrix( W ).inverse() * z;
    return res;
  }

};



template <class Kernel, typename... Arguments>
struct ProductKernel {
  using Force = ForceField<ForceSingleLink>;
  using Gauge = GaugeField;
  using Rng = ParallelRngLink;

  const Lattice& lattice;

  std::vector<Kernel> field;

  ProductKernel( const Lattice& lattice, const Arguments... args)
    : lattice(lattice)
    , field(lattice.n_links(), Kernel(args...) )
  {};

  double operator()( const Force& f, const Gauge& W ) const {
    double res = 0.0;

    std::vector<double> tmp( lattice.n_links() );
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(Idx i=0; i<lattice.n_links(); i++) tmp[i] = field[i]( f[i], W[i] );

    for(auto elem : tmp ) res += elem;

    return res;
  }

  Force d( const Force& f, const Gauge& W ) const {
    Force res(lattice);
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(int i=0; i<lattice.n_links(); i++) res[i] = field[i].d( f[i], W[i] );
    return res;
  }

  Force act( const Gauge& W, const Force& f ) const {
    Force res(lattice);
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(int i=0; i<lattice.n_links(); i++) res[i] = field[i].act( W[i], f[i] );
    return res;
  }

  double logdet( const Gauge& W ) const {
    double res = 0.0;
    // for(int i=0; i<lattice.n_links(); i++) res += field[i].logdet( W[i] );

    std::vector<double> tmp( lattice.n_links() );
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(Idx i=0; i<lattice.n_links(); i++) tmp[i] = field[i].logdet( W[i] );

    for(auto elem : tmp ) res += elem;

    return res;
  }

  Force logdet_d( const Gauge& W ) const {
    Force res(lattice);
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(int i=0; i<lattice.n_links(); i++) res[i] = field[i].logdet_d( W[i] );
    return res;
  }

  Force gen( const Gauge& W, Rng& rng ) const {
    Force res(lattice);
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(Idx il=0; il<lattice.n_links(); il++){
      res[il] = field[il].gen( W[il], rng[il] );
    }
    return res;
  }

};



struct TrivialKernel2 {
  using Force = LinkForce;
  using Gauge = LinkConf;
  using Rng = SingleRng;

  TrivialKernel2(){};

  inline double operator()( const Force& f, const Gauge& W ) const {
    return f.wbasis( W.J() ).squaredNorm();
    // return f.square();
  }
  inline Force d( const Force& f, const Gauge& W ) const { return Force(); } // zero
  inline Force act( const Gauge& W, const Force& f ) const { return f; }
  inline double logdet( const Gauge& W ) const { return 0.0; }
  inline Force logdet_d( const Gauge& W ) const { return Force(); } // zero

  Force gen( const Gauge& W, Rng& rng ) const {
    Force::VG p;
    for(auto& elem : p) elem = rng.gaussian();
    return Force( p, W.J() );
    // Force p;
    // for(int a=0; a<NA; a++) p.pi(a) = rng.gaussian();
    // p.pi0 = rng.gaussian();
    // for(int a=0; a<NA; a++) p.rho(a) = rng.gaussian();
    // p.rho0 = rng.gaussian();
    // return p;
    // Force::VG wbasis; //  = f.wbasis( W.J() );
    // for(auto& elem : wbasis ) elem = rng.gaussian();
    // return Force( wbasis, W.J() );
  }
};





// struct Laplacian {
//   // using Force = Force2D;
//   using Force = Force2D<MatrixForceLink>;
//   using Gauge = Dim2Gauge;
//   using Rng = ParallelRngLink;

//   using Complex = std::complex<double>;
//   using MN = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
//   using Idx = std::size_t;

//   const Lattice& lattice;

//   Laplacian( const Lattice& lattice)
//     : lattice(lattice)
//   {};

//   // MN matrix_free() const {
//   //   MN res( NA*lattice.n_links(), NA*lattice.n_links() );
//   //   res.setZero();

//   //   for(Idx ix=0; ix<lattice.n_sites(); ix++) {
//   //     for(int nu=0; nu<DIM; nu++) {
//   //       const Idx il = DIM*ix + nu;

//   //       for(int mu=0; mu<DIM; mu++) {
//   //         const Idx ixp = lattice.cshift(ix, mu);
//   //         const Idx ixm = lattice.cshift(ix, -mu-1);

//   //         const Idx ilp = DIM*ixp + nu;
//   //         const Idx ilm = DIM*ixm + nu;

//   //         res.block( NA*ilp, NA*il, NA, NA ) = MN::Identity(NA, NA);
//   //         res.block( NA*ilm, NA*il, NA, NA ) = MN::Identity(NA, NA);
//   //         res.block( NA*il,  NA*il, NA, NA ) = -2.0*MN::Identity(NA, NA);
//   //       }
//   //     }
//   //   }
//   //   res /= -16.0;

//   //   return res;
//   // }

//   Force act( const Gauge& W, const Force& f ) const {
//     Force res(lattice);

//     for(Idx ix=0; ix<lattice.n_sites(); ix++) {
//       for(int nu=0; nu<DIM; nu++) {
//         const Idx il = DIM*ix + nu;

//         for(int mu=0; mu<DIM; mu++) {
//           const Idx ixp = lattice.cshift(ix, mu);
//           const Idx ixm = lattice.cshift(ix, -mu-1);

//           const Idx ilp = DIM*ixp + nu;
//           const Idx ilm = DIM*ixm + nu;

//           res[il] = W(ix,mu).U * f[ilp] * W(ix,mu).U.inverse();
//           res[il] = W(ixm,mu).U.inverse() * f[ilm] * W(ixm,mu).U;
//           res[il] = -2.0 * f[il];
//         }
//       }
//     }
//     res /= -16.0;

//     return res;
//   }


//   Force gen( const Gauge& W, Rng& rng ) const {
//     Force res(lattice);
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(nparallel)
// #endif
//     for(Idx il=0; il<lattice.n_links(); il++){
//       for(int a=0; a<NA; a++) res[il] += rng[il].gaussian() * W[il].t[a];
//     }

//     // act()

//     return res;
//   }



// };



// struct Trivial {
//   using Force = Force2D<MatrixForceLink>;
//   using Gauge = Dim2Gauge;
//   using Rng = ParallelRngLink;

//   using Complex = std::complex<double>;
//   // using MN = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
//   using Idx = std::size_t;

//   const Lattice& Trivial;

//   Trivial( const Lattice& lattice)
//     : lattice(lattice)
//   {};

//   inline double operator()( const Force& f, const Gauge& W ) const {
//     return f.pi.squaredNorm();
//   }

//   Force act( const Gauge& W, const Force& f ) const {
//     Force res(lattice);

//     for(Idx ix=0; ix<lattice.n_sites(); ix++) {
//       for(int nu=0; nu<DIM; nu++) {
//         const Idx il = DIM*ix + nu;

//         for(int mu=0; mu<DIM; mu++) {
//           const Idx ixp = lattice.cshift(ix, mu);
//           const Idx ixm = lattice.cshift(ix, -mu-1);

//           const Idx ilp = DIM*ixp + nu;
//           const Idx ilm = DIM*ixm + nu;

//           res[il] = W(ix,mu).U * f[ilp] * W(ix,mu).U.inverse();
//           res[il] = W(ixm,mu).U.inverse() * f[ilm] * W(ixm,mu).U;
//           res[il] = -2.0 * f[il];
//         }
//       }
//     }
//     res /= -16.0;

//     return res;
//   }


//   Force gen( const Gauge& W, Rng& rng ) const {
//     Force res(lattice);
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(nparallel)
// #endif
//     for(Idx il=0; il<lattice.n_links(); il++){
//       for(int a=0; a<NA; a++) res[il] += rng[il].gaussian() * W[il].t[a];
//     }

//     // act()

//     return res;
//   }



// };



struct TrivialKernel {
  using Force = ForceSingleLink;
  using Gauge = LinkConfig;
  using Rng = SingleRng;

  TrivialKernel(){};

  inline double operator()( const Force& f, const Gauge& W ) const { return f.pi.squaredNorm(); }
  inline Force d( const Force& f, const Gauge& W ) const { return Force(); } // zero
  inline Force act( const Gauge& W, const Force& f ) const { return f; }
  inline double logdet( const Gauge& W ) const { return 0.0; }
  inline Force logdet_d( const Gauge& W ) const { return Force(); } // zero

  Force gen( const Gauge& W, Rng& rng ) const {
    Force p;
    for(int i=0; i<p.size(); i++) p[i] = rng.gaussian();
    return p;
  }
};
