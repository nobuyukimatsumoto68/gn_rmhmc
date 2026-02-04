#pragma once

#include <cassert>
#include <cmath>
#include <vector>
#include <Eigen/Dense>

#include <unsupported/Eigen/MatrixFunctions>

#include "lattice.h"


/*
  Gauge objects should have:
  Gauge operator+=(const Force& rhs);
  friend Gauge operator-(Gauge v, const Gauge& w);
*/


template<typename M>
M AH(const M& m, const int kmax=20){
  M res = M::Zero();
  M old = M::Zero();
  double factorial=1.0;
  M pow = M::Identity();

  int k;
  for(k=0; k<kmax; k++){
    factorial *= k+1;
    res += pow/factorial;
    pow = pow*m;
    if( (res-old).norm()<1.0e-14 ) break;
    old = res;
  }
  if(k>=kmax-1) assert( false );
  return res;
}



struct LinkConfig { // Force=ForceSingleLink
  using Gauge=LinkConfig;
  using Force=ForceSingleLink;

  using Rng = SingleRng;

  using MC = Eigen::Matrix<Complex, Nc, Nc, Eigen::RowMajor>;
  using MR = Eigen::Matrix<double, Nc, Nc, Eigen::RowMajor>;
  using VH = Eigen::Matrix<double, NH, 1>;
  using MG = Eigen::Matrix<double, NG, NG, Eigen::RowMajor>;

  inline double& re(std::complex<double>& c){ return reinterpret_cast<double (&)[2]>(c)[0]; }
  inline double& im(std::complex<double>& c){ return reinterpret_cast<double (&)[2]>(c)[1]; }
  inline double re(const std::complex<double>& c) const { return reinterpret_cast<const double (&)[2]>(c)[0]; }
  inline double im(const std::complex<double>& c) const { return reinterpret_cast<const double (&)[2]>(c)[1]; }

  const Generators2 t;
  MC W;
  double theta;
  MC U;
  MC Phi;

  LinkConfig()
    : t()
    , W( id() )
    , theta( 0.0 )
    , U( id() )
    , Phi( id() )
  {
    check_consistency();
  }

  LinkConfig( const LinkConfig& other )
    : t( other.t )
    , W( other.W )
    , theta( other.theta )
    , U( other.U )
    , Phi( other.Phi )
  {
    check_consistency();
  }

  LinkConfig& operator=(const LinkConfig& other) {
    if (this == &other) return *this;

    W = other.W;
    theta = other.theta;
    U = other.U;
    Phi = other.Phi;
    check_consistency();

    return *this;
  }

  Gauge& operator+=(const Force& f) {
    VH dwr = f.pi.segment(0, Nc*Nc);
    VH dwi = f.pi.segment(Nc*Nc, Nc*Nc);
    W += Eigen::Map<MR>( dwr.data() );
    W += I*Eigen::Map<MR>( dwi.data() );
    update_others();
    return *this;
  }
  friend Gauge operator+(Gauge v, const Force& w) { v += w; return v; }

  Gauge& operator-=(const Force& f) {
    VH dwr = f.pi.segment(0, Nc*Nc);
    VH dwi = f.pi.segment(Nc*Nc, Nc*Nc);
    W -= Eigen::Map<MR>( dwr.data() );
    W -= I*Eigen::Map<MR>( dwi.data() );
    update_others();
    return *this;
  }
  friend Gauge operator-(Gauge v, const Force& w) { v -= w; return v; }

  Gauge& operator+=(const Gauge& rhs) {
    W += rhs.W;
    update_others();
    return *this;
  }
  friend Gauge operator+(Gauge v, const Gauge& w) { v += w; return v; }

  Gauge& operator-=(const Gauge& rhs) {
    W -= rhs.W;
    update_others();
    return *this;
  }
  friend Gauge operator-(Gauge v, const Gauge& w) { v -= w; return v; }

  inline MC id() const { return MC::Identity(); }
  inline Complex u1( const double alpha ) const { return std::exp(I*alpha); }
  inline Complex u() const { return u1(theta); }
  inline MC operator()() const { return W; }
  inline Complex operator()(const int i, const int j) const { return W(i,j); }
  inline Complex& operator()(const int i, const int j) { return W(i,j); }
  inline double norm() const { return W.norm(); }

  void get_qij( int& q, int& i, int& j, const int qij ) const {
    if(qij<Nc*Nc){
      q=0;
      j=qij%Nc;
      i=(qij-j+1)/Nc;
    }
    else if(qij<2*Nc*Nc) {
      q=1;
      j=qij%Nc;
      i=(qij-j+1-Nc*Nc)/Nc;
    }
    else assert( false );
  }

  double operator[](const int qij) const {
    int q,i,j;
    get_qij( q,i,j, qij );
    if(q==0) return re(W(i,j));
    else if(q==1) return im(W(i,j));
    else assert( false );
  }

  double& operator[](const int qij) {
    int q,i,j;
    get_qij( q,i,j, qij );
    if(q==0) return re(W(i,j));
    else if(q==1) return im(W(i,j));
    else assert( false );
  }

  double mod2pi( const double alpha ) const {
    double res = alpha + 4.0*M_PI;
    res -= int(std::floor(res/(2.0*M_PI)))*2.0*M_PI;
    if(res>= 2.0*M_PI) res -= 2.0*M_PI;
    return res;
  }

  //   std::mt19937_64 mt_;
  // std::uniform_real_distribution<> dist_01_;
  // std::uniform_real_distribution<> dist_m11_;
  // // uniform_real_distribution<> dist_x_;
  // std::uniform_real_distribution<> dist_cos_;
  // std::uniform_real_distribution<> dist_phi_;
  // //
  // inline double uniform(){ return dist_01_(mt_); }
  // inline double betagen0(){ return dist_m11_(mt_); }
  // inline double gencos(){ return dist_cos_(mt_); }
  // inline double genphi(){ return dist_phi_(mt_); }
  // void generate_uniform( double &a0, double &cos, double &phi );
  // Eigen::MatrixXcd calcSU2( const double a0, const double &cos, const double &phi );
  // std::vector<Eigen::MatrixXcd> get_single( const int K);

  // void GenerateSUN::generate_uniform
// (double &a0,
//  double &cosine,
//  double &phi
//  ){
//   bool is_accept = false;

//   while(!is_accept){
//     a0 = betagen0();
//     double r = uniform();
//     if( r < sqrt( 1.0-a0*a0 ) ){
//       is_accept = true;
//       break; // break while
//     } // end if
//   } // end while

//   cosine = gencos();
//   phi = genphi();
// }

// Eigen::MatrixXcd GenerateSUN::calcSU2
// (const double a0,
//  const double &cosine,
//  const double &phi
//  ){
//   double r = sqrt( 1.0 - a0*a0);
//   double sine = sqrt( 1.0 - cosine*cosine);
//   return
//     a0 * Eigen::MatrixXcd::Identity(2,2) +
//     Basic::I() * r * ( sine * cos(phi) * sigma1() +
//                        sine * sin(phi) * sigma2() +
//                        cosine * sigma3() );
// };

  void randomize( Rng& rng, const double width=1.0, const double c = 1.0 ){
    // MC tmp;
    // while(true){
    //   for(int i=0; i<Nc; i++){
    // 	for(int j=0; j<Nc; j++){
    // 	  tmp(i, j) = rng.gaussian() + I * rng.gaussian();
    // 	  tmp(i, j) *= width;
    // 	}}
    //   tmp = 0.5*(tmp + tmp.adjoint());
    //   if( tmp.determinant().real()>0 ) break;
    // }
    // Phi = tmp;

    // for(int i=0; i<Nc; i++){
    //   for(int j=0; j<Nc; j++){
    // 	tmp(i, j) = rng.gaussian() + I * rng.gaussian();
    // 	tmp(i, j) *= width;
    //   }}
    // tmp = 0.5*(tmp - tmp.adjoint());
    // U = tmp.exp();
    // theta = rng.uniform() * 2.0*M_PI;

    for(int i=0; i<Nc; i++){
      for(int j=0; j<Nc; j++){
	(*this)(i, j) = rng.gaussian() + I * rng.gaussian();
	(*this)(i, j) *= width;
      }}
    W += c*id();
    update_others();
    theta += 2.0*M_PI/Nc * (rng.mt()%Nc);
    update();
    update_others();
    // update();
    // update_others();
    // for(int i=0; i<Nc; i++){
    //   for(int j=0; j<Nc; j++){
    //     W(i, j) = f1() + I*f2();
    //   }}
    // update_others();
  }

  void check_consistency( const double TOL=1.0e-10 ) const {
    const MC check = u()*Phi*U;
    const double norm = (check-W).norm()/(std::sqrt(2.0)*Nc);
    if(norm > TOL) {
      std::clog << "norm = " << norm << std::endl;
      std::clog << "det Phi = " << Phi.determinant() << std::endl;
    }
    assert( norm<TOL );
  }


  void decomposition(){
    // Eigen::JacobiSVD<MC> svd;
    Eigen::BDCSVD<MC, Eigen::ComputeFullU | Eigen::ComputeFullV> svd;
    // svd.compute<MC, >(W, Eigen::ComputeFullU | Eigen::ComputeFullV); // U S V^\dagger
    svd.compute(W); // U S V^\dagger
    {
      const MC check = svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().adjoint();
      double norm = (check-W).norm();
      if(norm>1.0e-10) {
        std::clog << "W = " << W << std::endl;
        std::clog << "det W = " << W.determinant() << std::endl;
        std::clog << "norm = " << norm << std::endl;
      }
      assert( norm<1.0e-10 );
    }
    Phi = svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixU().adjoint();
    const MC Omega = u1(-theta) * svd.matrixU() * svd.matrixV().adjoint();
    const double dtheta = std::arg( Omega.determinant() ) / Nc;
    theta = mod2pi( theta + dtheta );
    U = u1(-dtheta) * Omega;
  }

  void update(){
    W = u()*Phi*U;
    check_consistency();
  }

  void update_from( const MC& Wnew ){
    W = Wnew;
    decomposition();
    check_consistency();
  }

  void update_others(){
    decomposition();
    check_consistency();
  }

  MG J() const {
    MG res = MG::Zero();
    for(int a=0; a<NA; a++){
      // const MC mat = ( I*u()*Phi*t[a]*U );
      const MC mat = ( u()*Phi*t[a]*U );
      const MR Re = mat.real(); // to avoid bugs of Eigen; no const, no .real().data()
      const MR Im = mat.imag(); // to avoid bugs of Eigen; no const, no .real().data()
      res.block(a, 0, 1, Nc*Nc) = Eigen::Map<const VH>( Re.data() ).transpose();
      res.block(a, Nc*Nc, 1, Nc*Nc) = Eigen::Map<const VH>( Im.data() ).transpose();
    }
    { // 0th element
      const MC mat = ( I*u()*Phi*U );
      const MR Re = mat.real(); // to avoid bugs of Eigen; no const, no .real().data()
      const MR Im = mat.imag(); // to avoid bugs of Eigen; no const, no .real().data()
      res.block(Nc*Nc-1, 0, 1, Nc*Nc) = Eigen::Map<const VH>( Re.data() ).transpose();
      res.block(Nc*Nc-1, Nc*Nc, 1, Nc*Nc) = Eigen::Map<const VH>( Im.data() ).transpose();
    }
    for(int a=0; a<NA; a++){
      const MC mat = ( -I*u()*t[a]*U );
      const MR Re = mat.real(); // to avoid bugs of Eigen; no const, no .real().data()
      const MR Im = mat.imag(); // to avoid bugs of Eigen; no const, no .real().data()
      res.block(Nc*Nc+a, 0, 1, Nc*Nc) = Eigen::Map<const VH>( Re.data() ).transpose();
      res.block(Nc*Nc+a, Nc*Nc, 1, Nc*Nc) = Eigen::Map<const VH>( Im.data() ).transpose();
    }
    { // 0th element
      const MC mat = ( u()*U );
      const MR Re = mat.real(); // to avoid bugs of Eigen; no const, no .real().data()
      const MR Im = mat.imag(); // to avoid bugs of Eigen; no const, no .real().data()
      res.block(2*Nc*Nc-1, 0, 1, Nc*Nc) = Eigen::Map<const VH>( Re.data() ).transpose();
      res.block(2*Nc*Nc-1, Nc*Nc, 1, Nc*Nc) = Eigen::Map<const VH>( Im.data() ).transpose();
    }
    return res;
  }

  friend std::ostream& operator<<(std::ostream& os, const Gauge& v){ os << v.W; return os; }

};







// struct LinkConf { // Force=ForceSingleLink
//   using Gauge=LinkConf;
//   using Force=LinkForce;

//   using Rng = SingleRng;

//   using Complex = std::complex<double>;
//   static constexpr Complex I = Complex(0.0, 1.0);

//   using MC = Eigen::Matrix<Complex, Nc, Nc, Eigen::RowMajor>;
//   using MR = Eigen::Matrix<double, Nc, Nc, Eigen::RowMajor>;
//   using VG = Eigen::Matrix<double, NG, 1>;
//   using VH = Eigen::Matrix<double, NH, 1>;
//   using MG = Eigen::Matrix<double, NG, NG, Eigen::RowMajor>;

//   inline double& re(std::complex<double>& c){ return reinterpret_cast<double (&)[2]>(c)[0]; }
//   inline double& im(std::complex<double>& c){ return reinterpret_cast<double (&)[2]>(c)[1]; }
//   inline double re(const std::complex<double>& c) const { return reinterpret_cast<const double (&)[2]>(c)[0]; }
//   inline double im(const std::complex<double>& c) const { return reinterpret_cast<const double (&)[2]>(c)[1]; }

//   const Generators2 t;
//   MC W;
//   double theta;
//   MC U;
//   MC Phi;

//   LinkConf()
//     : t()
//     , W( id() )
//     , theta( 0.0 )
//     , U( id() )
//     , Phi( id() )
//   {
//     check_consistency();
//   }

//   LinkConf( const LinkConf& other )
//     : t( other.t )
//     , W( other.W )
//     , theta( other.theta )
//     , U( other.U )
//     , Phi( other.Phi )
//   {
//     check_consistency();
//   }

//   LinkConf& operator=(const LinkConf& other) {
//     if (this == &other) return *this;

//     W = other.W;
//     theta = other.theta;
//     U = other.U;
//     Phi = other.Phi;
//     check_consistency();

//     return *this;
//   }

//   Gauge& operator+=(const Force& f) {
//     const VG wbasis = f.wbasis( J() );
//     const VH dwr = wbasis.segment(0, Nc*Nc);
//     const VH dwi = wbasis.segment(Nc*Nc, Nc*Nc);
//     W += Eigen::Map<const MR>( dwr.data() );
//     W += I*Eigen::Map<const MR>( dwi.data() );
//     update_others();
//     return *this;
//   }
//   friend Gauge operator+(Gauge v, const Force& w) { v += w; return v; }

//   Gauge& operator-=(const Force& f) {
//     const VG wbasis = f.wbasis( J() );
//     const VH dwr = wbasis.segment(0, Nc*Nc);
//     const VH dwi = wbasis.segment(Nc*Nc, Nc*Nc);
//     W -= Eigen::Map<const MR>( dwr.data() );
//     W -= I*Eigen::Map<const MR>( dwi.data() );
//     update_others();
//     return *this;
//   }
//   friend Gauge operator-(Gauge v, const Force& w) { v -= w; return v; }

//   Gauge& operator+=(const Gauge& rhs) {
//     W += rhs.W;
//     update_others();
//     return *this;
//   }
//   friend Gauge operator+(Gauge v, const Gauge& w) { v += w; return v; }

//   Gauge& operator-=(const Gauge& rhs) {
//     W -= rhs.W;
//     update_others();
//     return *this;
//   }
//   friend Gauge operator-(Gauge v, const Gauge& w) { v -= w; return v; }

//   inline MC id() const { return MC::Identity(); }
//   inline Complex u1( const double alpha ) const { return std::exp(I*alpha); }
//   inline Complex u() const { return u1(theta); }
//   inline MC operator()() const { return W; }
//   inline Complex operator()(const int i, const int j) const { return W(i,j); }
//   inline Complex& operator()(const int i, const int j) { return W(i,j); }
//   inline double norm() const { return W.norm(); }

//   void get_qij( int& q, int& i, int& j, const int qij ) const {
//     if(qij<Nc*Nc){
//       q=0;
//       j=qij%Nc;
//       i=(qij-j+1)/Nc;
//     }
//     else if(qij<2*Nc*Nc) {
//       q=1;
//       j=qij%Nc;
//       i=(qij-j+1-Nc*Nc)/Nc;
//     }
//     else assert( false );
//   }

//   double operator[](const int qij) const {
//     int q,i,j;
//     get_qij( q,i,j, qij );
//     if(q==0) return re(W(i,j));
//     else if(q==1) return im(W(i,j));
//     else assert( false );
//   }

//   double& operator[](const int qij) {
//     int q,i,j;
//     get_qij( q,i,j, qij );
//     if(q==0) return re(W(i,j));
//     else if(q==1) return im(W(i,j));
//     else assert( false );
//   }

//   double mod2pi( const double alpha ) const {
//     double res = alpha + 4.0*M_PI;
//     res -= int(std::floor(res/(2.0*M_PI)))*2.0*M_PI;
//     return res;
//   }

//   // void randomize( Rng& rng, const double width=1.0, const double c = 1.0 ){
//   //   for(int i=0; i<Nc; i++){
//   //     for(int j=0; j<Nc; j++){
//   //       (*this)(i, j) = rng.gaussian() + I * rng.gaussian();
//   //       (*this)(i, j) *= width;
//   //     }}
//   //   update_others();
//   //   // for(int i=0; i<Nc; i++){
//   //   //   for(int j=0; j<Nc; j++){
//   //   //     W(i, j) = f1() + I*f2();
//   //   //   }}
//   //   // update_others();
//   // }

//   void check_consistency( const double TOL=1.0e-10 ) const {
//     const MC check = u()*Phi*U;
//     const double norm = (check-W).norm()/(std::sqrt(2.0)*Nc);
//     if(norm > TOL) {
//       std::clog << "norm = " << norm << std::endl;
//       std::clog << "det Phi = " << Phi.determinant() << std::endl;
//     }
//     assert( norm<TOL );
//   }


//   void decomposition(){
//     // Eigen::JacobiSVD<MC> svd;
//     Eigen::BDCSVD<MC, Eigen::ComputeFullU | Eigen::ComputeFullV> svd;
//     svd.compute(W); // U S V^\dagger
//     {
//       const MC check = svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().adjoint();
//       double norm = (check-W).norm();
//       if(norm>1.0e-10) {
//         std::clog << "W = " << W << std::endl;
//         std::clog << "det W = " << W.determinant() << std::endl;
//         std::clog << "norm = " << norm << std::endl;
//       }
//       assert( norm<1.0e-10 );
//     }
//     Phi = svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixU().adjoint();
//     const MC Omega = u1(-theta) * svd.matrixU() * svd.matrixV().adjoint();
//     const double dtheta = std::arg( Omega.determinant() ) / Nc;
//     theta = mod2pi( theta + dtheta );
//     U = u1(-dtheta) * Omega;
//   }

//   void update(){
//     W = u()*Phi*U;
//     check_consistency();
//   }

//   void update_from( const MC& Wnew ){
//     W = Wnew;
//     decomposition();
//     check_consistency();
//   }

//   void update_others(){
//     decomposition();
//     check_consistency();
//   }

//   MG J() const {
//     MG res = MG::Zero();
//     for(int a=0; a<NA; a++){
//       const MC mat = ( I*u()*Phi*t[a]*U );
//       const MR Re = mat.real(); // to avoid bugs of Eigen; no const, no .real().data()
//       const MR Im = mat.imag(); // to avoid bugs of Eigen; no const, no .real().data()
//       res.block(a, 0, 1, Nc*Nc) = Eigen::Map<const VH>( Re.data() ).transpose();
//       res.block(a, Nc*Nc, 1, Nc*Nc) = Eigen::Map<const VH>( Im.data() ).transpose();
//     }
//     { // 0th element
//       const MC mat = ( I*u()*Phi*U );
//       const MR Re = mat.real(); // to avoid bugs of Eigen; no const, no .real().data()
//       const MR Im = mat.imag(); // to avoid bugs of Eigen; no const, no .real().data()
//       res.block(Nc*Nc-1, 0, 1, Nc*Nc) = Eigen::Map<const VH>( Re.data() ).transpose();
//       res.block(Nc*Nc-1, Nc*Nc, 1, Nc*Nc) = Eigen::Map<const VH>( Im.data() ).transpose();
//     }
//     for(int a=0; a<NA; a++){
//       const MC mat = ( u()*t[a]*U );
//       const MR Re = mat.real(); // to avoid bugs of Eigen; no const, no .real().data()
//       const MR Im = mat.imag(); // to avoid bugs of Eigen; no const, no .real().data()
//       res.block(Nc*Nc+a, 0, 1, Nc*Nc) = Eigen::Map<const VH>( Re.data() ).transpose();
//       res.block(Nc*Nc+a, Nc*Nc, 1, Nc*Nc) = Eigen::Map<const VH>( Im.data() ).transpose();
//     }
//     { // 0th element
//       const MC mat = ( u()*U );
//       const MR Re = mat.real(); // to avoid bugs of Eigen; no const, no .real().data()
//       const MR Im = mat.imag(); // to avoid bugs of Eigen; no const, no .real().data()
//       res.block(2*Nc*Nc-1, 0, 1, Nc*Nc) = Eigen::Map<const VH>( Re.data() ).transpose();
//       res.block(2*Nc*Nc-1, Nc*Nc, 1, Nc*Nc) = Eigen::Map<const VH>( Im.data() ).transpose();
//     }
//     return res;
//   }

//   friend std::ostream& operator<<(std::ostream& os, const Gauge& v){ os << v.W; return os; }

// };






// template<int DIM>
template<typename M=LinkConfig>
struct GaugeField {
  using Idx = std::size_t;
  using Coord=std::array<int, DIM>;
  // using M=LinkConfig;
  using Gauge=GaugeField<M>;
  using Force = ForceField<ForceSingleLink>; // <ForceSingleLink>;
  using Rng = ParallelRngLink;

  using Complex = std::complex<double>;
  static constexpr Complex I = Complex(0.0, 1.0);

  const Lattice& lattice;
  std::vector<M> field;

  GaugeField( const Lattice& lattice )
    : lattice( lattice )
    , field( lattice.n_links() )
  {}

  GaugeField( const GaugeField& other )
    : lattice( other.lattice )
    , field( other.field )
  {}

  GaugeField& operator=(const GaugeField& other) {
    if (this == &other) return *this;

    assert(&lattice==&other.lattice);
    field = other.field;
    return *this;
  }


  inline M operator[](const Idx i) const { return field[i]; }
  inline M& operator[](const Idx i) { return field[i]; }

  M operator()(const Coord& x, const int mu) const {
    if(mu>=0) return field[ DIM*lattice.idx(x) + mu];
    else return field[ DIM*lattice.idx( lattice.cshift(x,mu) ) -mu-1];
  }
  M& operator()(const Coord& x, const int mu) {
    if(mu>=0) return field[ DIM*lattice.idx(x) + mu];
    else return field[ DIM*lattice.idx( lattice.cshift(x,mu) ) -mu-1];
  }

  M operator()(const Idx i, const int mu) const {
    assert( mu>=0 );
    return field[ DIM*i + mu];
  }
  M& operator()(const Idx i, const int mu) {
    assert( mu>=0 );
    return field[ DIM*i + mu];
  }

  Gauge& operator+=(const Gauge& rhs){
    assert(&lattice==&rhs.lattice);
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(Idx il=0; il<lattice.n_links(); il++) field[il] += rhs.field[il];
    return *this;
  }
  friend Gauge operator+(Gauge v, const Gauge& w) { v += w; return v; }

  Gauge& operator-=(const Gauge& rhs){
    assert(&lattice==&rhs.lattice);
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(Idx il=0; il<lattice.n_links(); il++) field[il] -= rhs.field[il];
    return *this;
  }
  friend Gauge operator-(Gauge v, const Gauge& w) { v -= w; return v; }
  
  Gauge& operator+=(const Force& rhs){
    assert(&lattice==&rhs.lattice);
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(Idx il=0; il<lattice.n_links(); il++) field[il] += rhs.field[il];
    return *this;
  }
  friend Gauge operator+(Gauge v, const Force& w) { v += w; return v; }

  Gauge& operator-=(const Force& rhs){
    assert(&lattice==&rhs.lattice);
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(Idx il=0; il<lattice.n_links(); il++) field[il] -= rhs.field[il];
    return *this;
  }
  friend Gauge operator-(Gauge v, const Force& w) { v -= w; return v; }

  double norm() const {
    double res = 0.0;

    // for(Idx il=0; il<lattice.n_links(); il++) res += field[il].norm();
    std::vector<double> tmp( lattice.n_links() );
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(Idx i=0; i<lattice.n_links(); i++) tmp[i] = field[i].norm();

    for(auto elem : tmp ) res += elem;

    res /= std::sqrt( lattice.n_links() );
    return res;
  }

  void update(){
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(auto it = field.begin(); it!=field.end(); it++ ) it->update();
  }
  void update_others(){
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(auto it = field.begin(); it!=field.end(); it++ ) it->update_others();
  }

  // void randomize( Rng& rng, const double width=1.0 ){
  void randomize( Rng& rng, const double width=1.0, const double c = 1.0 ){
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(auto it = field.begin(); it!=field.end(); it++ ){
      const Idx il = std::distance( field.begin(), it );
      it->randomize( rng[il], width, c );
      // for(int i=0; i<Nc; i++){
      //   for(int j=0; j<Nc; j++){
      //     (*it)(i, j) = rng[il].gaussian() + I * rng[il].gaussian();
      //     (*it)(i, j) *= width;
      //   }}
      // it->update_others();
    }
  }

};
