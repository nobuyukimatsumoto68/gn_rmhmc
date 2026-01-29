#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <complex>
#include <cassert>

#include <omp.h>
constexpr int nparallel=4;

#define Nc 2
#define NA Nc*Nc-1
#define NG 2*Nc*Nc
#define NH Nc*Nc
#define DIM 4

using Idx = std::size_t;
using Coord=std::array<int, DIM>;
using Complex = std::complex<double>;
static constexpr Complex I = Complex(0.0, 1.0);

#include "lattice.h"
#include "rng.h"
#include "generators.h"
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
  std::cout << "openmp threads = " << omp_get_max_threads() << std::endl;
  std::cout << "eigen threads = " << Eigen::nbThreads() << std::endl;

  int seed = 0;
  if (argc>1){ seed = atoi(argv[1]); }
  std::cout << "seed = " << seed << std::endl;

  // ------------------

  using Force = ForceField<ForceSingleLink>;
  using Gauge = GaugeField;
  using Action = WilsonGaussianAndDet; // @@@

  using Kernel = ProductKernel<TrivialKernel>;
  // using Kernel = ProductKernel<IdpWHW, double>;
  using Integrator = ExplicitLeapfrog<Force,Gauge,Action,Kernel>;
  using Rng = ParallelRngLink;
  using HMC = HMC<Force,Gauge,Integrator,Rng>;

  // ---------------

  // const Lattice lat( Coord{{ 32, 32, 32, 32 }} );
  const Lattice lat( Coord{{ 4, 4, 4, 4 }} );

  // ---------------

  Gauge W(lat);

  // ------------------

  double beta = 0.05; // 4.0
  if (argc>2){ beta = atof(argv[2]); }
  double lambda = 1.0;
  if (argc>3){ lambda = atof(argv[3]); }
  // const double kappa = 0.0;
  const double kappa = 0.0;
  const double c = 8.0;
  Action S(lat, beta, lambda, kappa, c);

  // ------------------

  std::vector<Obs<double, Gauge>*> obslist;
  std::string data_path="./obs"+std::to_string(beta)+"_"+std::to_string(kappa)+"_"+std::to_string(lambda)+"_"+"/";
  std::filesystem::create_directory(data_path);

  // ------------------


  Kernel K(lat);
  // const double alpha = 0.001;
  // Kernel K(lat, alpha);

  // ------------------

  Rng rng(lat, 1);
  // W.randomize( rng, 1.0 );

  // ------------------

  // const double lambda_0 = 2.0 * std::cyl_bessel_i( 1, beta ) / beta;
  // const double lambda_F = ;
  double exact = beta / Nc;
  // if( std::abs(lambda_0)>1.0e-14 ) exact = 2.0*lambda_F/lambda_0;
  std::cout << "exact = " << exact << std::endl;

  Obs<double, Gauge> retrU( "retrU",
                            beta,
                            [&](const Gauge& W ){
                              std::vector<double> tmp( lat.vol, 0.0 );
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
                              for(Idx ix=0; ix<lat.vol; ix++) {
                                for(int mu=0; mu<DIM; mu++){
                                  for(int nu=mu+1; nu<DIM; nu++){
                                    tmp[ix] += S.plaq(W, ix, mu, nu).trace().real();
                                  }}
                              }
                              double res = 0.0;
                              for(auto elem : tmp ) res += elem;
                              res /= lat.vol;
                              res /= 0.5*DIM*(DIM-1);
                              return res;
                            },
                            exact );
  obslist.push_back(&retrU);

//   Obs<double, Gauge> retrUsq( "retrUsq", beta,
//                               [&](const Gauge& W ){
//                                 std::vector<double> tmp( lat.vol, 0.0 );
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(nparallel)
// #endif
//                                 for(Idx ix=0; ix<lat.vol; ix++) {
//                                   for(int mu=0; mu<DIM; mu++){
//                                     for(int nu=mu+1; nu<DIM; nu++){
//                                       const Complex tmp2 = S.plaq(W, ix, mu, nu).trace();
//                                       tmp[ix] += (tmp2*tmp2).real();
//                                     }}
//                                 }
//                                 double res = 0.0;
//                                 for(auto elem : tmp ) res += elem;
//                                 res /= lat.vol;
//                                 return res;
//                               },
//                               0.0 );
//   obslist.push_back(&retrUsq);

//   Obs<double, Gauge> phi_norm( "trPhisq", beta, [&](const Gauge& W ){
//     std::vector<double> tmp( lat.vol, 0.0 );
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(nparallel)
// #endif
//     for(Idx ix=0; ix<lat.vol; ix++) {
//       tmp[ix] += ( W[ix].Phi - c*W[ix].id() ).squaredNorm();
//     }
//     double res = 0.0;
//     for(auto elem : tmp ) res += elem;
//     res *= 0.5/Nc;
//     res /= lat.vol;
//     return res;
//   }, 0.0 );
//   obslist.push_back(&phi_norm);
  // Obs<double, double> acceptance( "acceptance", beta, [](const double r ){
  //   return r;
  // }, 0.0 );
  // obslist.push_back(&);
  // Obs<double, Gauge> detK( "absdetK", beta, [&](const Gauge& W ){
  //   return std::abs(K.det(W));
  // }, 0.0 );
  // obslist.push_back(&detK);

  // ------------------

  rng.seed( seed );
  // W.randomize( rng, 0.1 );
  W.randomize( rng, 1.0, c );

  // ------------------

  const double stot = 2.0;
  int nsteps = 10;
  // int nsteps = 11;

  Integrator md(S, K, stot, nsteps);
  HMC hmc(md, rng, stot, nsteps);

  {
    int ntherm=1000;
    int niter=1000;
    int until=20;
    // int ntherm=10;
    // int niter=0;
    if (argc>4){ ntherm = atoi(argv[4]); }
    if (argc>5){ niter = atoi(argv[5]); }
    const int interval = 10;
    const int binsize = 10;

    double dH, r;
    bool is_accept;

    for(int n=0; n<ntherm; n++) {
      bool no_reject = false;
      if(n<until) no_reject = true;
      hmc.run(W, r, dH, is_accept, no_reject);
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
