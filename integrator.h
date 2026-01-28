#pragma once


/*
  Integrator objects need to inherit MDBase and to have:
  onestep( Force& p, Gauge& W ) const;
*/


template <class Force, class Gauge, class Action, class Kernel>
struct MDBase {
  const Action& S;
  const Kernel& K;

  MDBase (const Action& S_, const Kernel& K_)
    : S(S_)
    , K(K_)
  {}

  double H( const Force& pi, const Gauge& W ) const {
    double res = 0.0;
    res += 0.5 * K(pi, W);
    res += S(W);
    res -= 0.5 * K.logdet(W);
    return res;
  }

  inline Force dHdp( const Force& p, const Gauge& W ) const { return K.act( W, p ); }

  Force dHdW( const Force& p, const Gauge& W ) const {
    Force res = S.d(W);
    // res += 0.5 * K.d(p, W);
    // res -= 0.5 * K.logdet_d(W); // @@@
    return res;
  }
};



template <class Force, class Gauge, class Action, class Kernel>
class ExplicitLeapfrog : public MDBase<Force,Gauge,Action,Kernel> {
public:
  const double stot;
  const int nsteps;
  const double tau;

  ExplicitLeapfrog(const Action& S_, const Kernel& K_,
		   const double stot_=1.0, const int nsteps_=10)
    : MDBase<Force,Gauge,Action,Kernel>(S_, K_)
    , stot(stot_)
    , nsteps(nsteps_)
    , tau(stot/nsteps)
  {}
  
  void onestep( Force& p, Gauge& W ) const {
    p += -0.5*tau * this->dHdW(p, W);
    W += tau * this->dHdp(p, W);
    p += -0.5*tau * this->dHdW(p, W);
  }

};


template <class Force, class Gauge, class Action, class Kernel>
class ImplicitLeapfrog : public MDBase<Force,Gauge,Action,Kernel> {
public:
  const double stot;
  const int nsteps;
  const double tau;

  ImplicitLeapfrog(const Action& S_, const Kernel& K_,
		   const double stot_=1.0, const int nsteps_=10)
    : MDBase<Force,Gauge,Action,Kernel>(S_, K_)
    , stot(stot_)
    , nsteps(nsteps_)
    , tau(stot/nsteps)
  {}

  Force get_phalf( const Force& p, const Gauge& W, const double TOL=1.0e-10 ) const {
    Force ph( p - 0.5*tau * this->dHdW(p, W) );
    Force pold( p );
 
    double norm = (ph-pold).norm();
    while( norm>TOL ){
      pold = ph;
      ph = p - 0.5*tau * this->dHdW(ph, W);
      norm = (ph-pold).norm();
    }
    return ph;
  }

  Gauge get_Wp( const Force& ph, const Gauge& Wh, const double TOL=1.0e-10 ) const {
    Gauge Wtmp( Wh + 0.5 * tau * this->dHdp( ph, Wh ) );
    // Gauge Wp( Wh + 0.25 * tau * ( this->dHdp( ph, Wtmp ) + this->dHdp( ph, Wh ) ) );
    Gauge Wp( Wh + 0.5 * tau * this->dHdp( ph, Wtmp ) );
    Gauge Wold( Wh );

    double norm = (Wp-Wold).norm();
    while( norm>TOL ){
      Wold = Wp;
      Wp = Wh + 0.5 * tau * this->dHdp( ph, Wp );
      norm = (Wp-Wold).norm();
    }
    return Wp;
  }

  void onestep( Force& p, Gauge& W ) const {
    p = get_phalf( p, W );
    W += 0.5 * tau * this->dHdp( p, W );
    W = get_Wp( p, W );
    p += -0.5*tau * this->dHdW(p, W);
  }

};

