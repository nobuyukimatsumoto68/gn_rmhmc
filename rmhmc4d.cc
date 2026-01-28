#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <filesystem>

#include <omp.h>
constexpr int nparallel=3;

using Idx = std::size_t;
using Coord=std::array<int, DIM>;
using Complex = std::complex<double>;
static constexpr Complex I = Complex(0.0, 1.0);

#define Nc 2
#define NA Nc*Nc-1
#define NG 2*Nc*Nc
#define NH Nc*Nc
#define DIM 4

#include "lattice.h"
#include "rng.h"
#include "force.h"
#include "gauge.h"
#include "kernel.h"
#include "action.h"
#include "integrator.h"
#include "hmc.h"
#include "obs.h"

#include <unsupported/Eigen/MatrixFunctions>


int main( int argc, char *argv[] ){
  std::cout << std::scientific << std::setprecision(15);
  omp_set_dynamic(0);
  omp_set_num_threads(nparallel);
  Eigen::setNbThreads(1);
  std::cout << "openmp threads = " << omp_get_num_threads() << std::endl;
  std::cout << "eigen threads = " << Eigen::nbThreads() << std::endl;

  int seed = 0;
  if (argc>1){ seed = atoi(argv[1]); }

  // ------------------

  using Force = Force2D;
  using Gauge = Dim2Gauge;
  using Action = WilsonGaussianAndDet2D;

  // using Kernel = ProductKernel<TrivialKernel>;
  using Kernel = ProductKernel<IdpWHW, double>;
  using Integrator = ImplicitLeapfrog<Force,Gauge,Action,Kernel>;
  using Rng = ParallelRngLink;
  using HMC = HMC<Force,Gauge,Integrator,Rng>;

  // ---------------

  const Lattice lat( Lattice::Coord{{ 24, 24 }} );

  // ---------------

  Gauge W(lat);

  std::vector<Obs<double, Gauge>*> obslist;
  std::string data_path="./obs/";
  std::filesystem::create_directory(data_path);

  // ------------------

  double beta = 2.0; // 4.0
  if (argc>2){ beta = atof(argv[2]); }
  const double lambda = 1.0;
  const double kappa = 5.0;
  Action S(lat, beta, lambda, kappa);

  // ------------------

  // Kernel K(lat, Nc);
  const double alpha = 0.001;
  Kernel K(lat, alpha);

  // ------------------

  Rng rng(lat, 1);
  W.randomize( rng, 1.0 );

  // ------------------

  const double lambda_0 = 2.0 * std::cyl_bessel_i( 1, beta ) / beta;
  const double lambda_F = 2.0 * std::cyl_bessel_i( 2, beta ) / beta;
  double exact = 0.0;
  if( std::abs(lambda_0)>1.0e-14 ) exact = 2.0*lambda_F/lambda_0;
  std::cout << "exact = " << exact << std::endl;

  Obs<double, Gauge> retrU( "retrU",
                            beta,
                            [&](const Gauge& W ){
                              std::vector<double> tmp( lat.vol, 0.0 );
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
                              for(Action::Idx ix=0; ix<lat.vol; ix++) {
                                tmp[ix] += S.plaq(W, ix).trace().real();
                              }
                              double res = 0.0;
                              for(auto elem : tmp ) res += elem;
                              res /= lat.vol;
                              return res;
                            },
                            exact );
  obslist.push_back(&retrU);
  // Obs<double, Gauge> phi_norm( "trPhisq", beta, [](const Gauge& W ){
  //   return ( W.Phi - W.id() ).squaredNorm();
  // }, 0.0 );
  // obslist.push_back(&phi_norm);
  // Obs<double, Gauge> phi_det_abs( "absdetPhi", beta, [](const Gauge& W ){
  //   return std::abs(W.Phi.determinant());
  // }, 0.0 );
  // obslist.push_back(&phi_det_abs);
  // Obs<double, Gauge> detK( "absdetK", beta, [&](const Gauge& W ){
  //   return std::abs(K.det(W));
  // }, 0.0 );
  // obslist.push_back(&detK);

  // ------------------

  rng.seed( seed );
  W.randomize( rng );

  // ------------------

  const double stot = 1.0;
  int nsteps = 6;

  Integrator md(S, K, stot, nsteps);
  HMC hmc(md, rng, stot, nsteps);

  {
    int ntherm=200;
    int niter=1000;
    // int ntherm=10;
    // int niter=0;
    if (argc>3){ ntherm = atoi(argv[3]); }
    if (argc>4){ niter = atoi(argv[4]); }
    const int interval = 10;
    const int binsize = 20;

    double dH, r;
    bool is_accept;

    for(int n=0; n<ntherm; n++) {
      hmc.run(W, r, dH, is_accept);
      std::clog << "n = " << n
        	<< ", r = " << r
        	<< ", dH = " << dH
        	<< ", is_accept = " << is_accept
        	<< std::endl;
    }

    double mean = 0.0;
    for(int n=0; n<niter; n++){
      hmc.run(W, r, dH, is_accept);
      std::clog << "n = " << n
        	<< ", r = " << r
        	<< ", dH = " << dH
        	<< ", is_accept = " << is_accept
        	<< std::endl;
      mean += r;

      if(n%interval==0){
        for(auto pt : obslist) {
          pt->meas( W );
          std::cout << pt->description << "\t"
                    << *(pt->v.end()-1) << std::endl;
        }
      }
    }
    mean /= niter;
    std::clog << "acc = " << mean << std::endl;

    std::cout << "# beta \t mean \t err \t exact" << std::endl;
    for(auto pt : obslist){
      double mn, er;
      pt->jackknife( mn, er, binsize );
      std::cout << pt->description << "\t\t"
        	<< pt->param << "\t"
        	<< mn << "\t" << er << "\t" << pt->exact <<  std::endl;

      // ------------------------
      
      {
        std::ofstream of;
        of.open( data_path+pt->description+".dat", std::ios::out | std::ios::app);
        if(!of) assert(false);
        of << std::scientific << std::setprecision(15);
        of << pt->param << "\t"
           << mn << "\t"
           << er << "\t"
           << pt->exact << std::endl;
      }

      // ------------------------

      {
        std::ofstream of;
        of.open( data_path+pt->description+std::to_string(pt->param)+"_.dat", std::ios::out | std::ios::app);
        if(!of) assert(false);
        of << std::scientific << std::setprecision(15);
        for(auto iv=pt->v.begin(); iv!=pt->v.end(); ++iv) of << pt->param << "\t" << *iv << std::endl;
      }
    }
  }


  return 0;
}
