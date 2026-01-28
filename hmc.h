#pragma once


template <class Force, class Gauge, class Integrator, class Rng>
struct HMC {
  const Integrator& md;
  Rng& rng;
  const double stot;
  const int nsteps;
  const double tau;

  HMC(const Integrator& md_,
      Rng& rng_,
      const double stot_=1.0, const int nsteps_=10, const int seed_=1)
    : md(md_)
    , rng(rng_)
    , stot(stot_)
    , nsteps(nsteps_)
    , tau(stot/nsteps)
  {}

  void run( Gauge& W0,
	    double& r,
	    double& dH,
	    bool& is_accept,
	    const bool no_reject = false ) {
    Force p = md.K.gen( W0, rng );
    Gauge W( W0 );
    const double h0 = md.H(p, W);
    for(int i=0; i<nsteps; i++) md.onestep( p, W );
    const double h1 = md.H(p, W);

    dH = h1-h0;
    r = std::min( 1.0, std::exp(-dH) );
    const double a = rng.uniform();
    if( a < r || no_reject ){
      W0 = W;
      is_accept=true;
    }
    else is_accept=false;
  }

};
