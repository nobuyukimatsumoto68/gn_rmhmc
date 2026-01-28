#pragma once

#include <Eigen/Dense>

/*
  Force objects should have:

  operator+=(const Force& rhs);
  double square() const;
  void rand();
  Force& operator+=(const Force& rhs);
  friend Force operator*(const double a, Force v);
  friend Force operator-(Force v, const Force& w);
*/


struct ForceSingleLink{
  using Force = ForceSingleLink;

  using Complex = std::complex<double>;
  using MC = Eigen::Matrix<Complex, Nc, Nc, Eigen::RowMajor>;
  using VG = Eigen::Matrix<double, NG, 1>;
  using MG = Eigen::Matrix<double, NG, NG, Eigen::RowMajor>;

  VG pi;

  ForceSingleLink() { pi = VG::Zero(); }

  ForceSingleLink(const VG& pi_) { pi = pi_; }

  ForceSingleLink(const ForceSingleLink& other)
    : pi(other.pi)
  {}

  inline double norm() const { return pi.norm(); }

  Force& operator=(const Force& other) {
    if (this == &other) return *this;
    pi = other.pi;
    return *this;
  }

  // void Lmult(const MR& mat) {
  //   pi = mat*pi;
  // }
  void invert(const MG& mat) {
    const Eigen::PartialPivLU<MG> llt(mat);
    pi = llt.solve(this->pi);
  }


  inline double operator[](const int i) const { return pi[i]; }
  inline double& operator[](const int i) { return pi[i]; }
  inline int size() const { return pi.size(); }

  Force& operator+=(const Force& rhs){
    pi += rhs.pi;
    return *this;
  }
  friend Force operator+(Force v, const Force& w) { v += w; return v; }

  Force& operator-=(const Force& rhs){
    pi -= rhs.pi;
    return *this;
  }
  friend Force operator-(Force v, const Force& w) { v -= w; return v; }

  Force& operator*=(const double a){
    pi *= a;
    return *this;
  }

  friend Force operator*(const double a, Force v) { v.pi *= a; return v; }
  friend Force operator*(const MG& mat, Force v) { v.pi = mat*v.pi; return v; }

  friend std::ostream& operator<<(std::ostream& os, const Force& v) { os << v.pi; return os; }
};



struct LinkForce{
  using Force = LinkForce;
  // using Gauge = LinkConfig;

  using Complex = std::complex<double>;
  using MC = Eigen::Matrix<Complex, Nc, Nc, Eigen::RowMajor>;

  using VA = Eigen::Matrix<double, NA, 1>;
  using VG = Eigen::Matrix<double, NG, 1>;
  using MG = Eigen::Matrix<double, NG, NG, Eigen::RowMajor>;

  const Generators t;

  VA pi;
  double pi0;
  VA rho;
  double rho0;

  LinkForce()
    : t()
  {
    pi = VA::Zero();
    pi0 = 0.0;
    rho = VA::Zero();
    rho0 = 0.0;
  }

  // LinkForce(const VG& pi_) {
  //   pi = pi_;
  // }

  LinkForce(const LinkForce& other)
    : pi(other.pi)
    , pi0(other.pi0)
    , rho(other.rho)
    , rho0(other.rho0)
  {}

  LinkForce(const VG& wbasis, const MG& J)
  {
    pi = VA::Zero();
    pi0 = 0.0;
    rho = VA::Zero();
    rho0 = 0.0;
    update_from( wbasis, J );
  }

  VG a0basis() const {
    VG res;
    res.segment(0, NA) = pi;
    res[NA] = pi0;
    res.segment(NA+1, NA) = rho;
    res[2*NA+1] = rho0;
    return res;
  }

  inline VG wbasis( const MG& J ) const {
    const Eigen::PartialPivLU<MG> lu( J );
    return lu.solve( a0basis() );
    // return a0basis().transpose() * J; // lu.solve( a0basis() ).transpose();
  }

  void update_from( const VG& w_basis, const MG& J ) {
    // const Eigen::PartialPivLU<MG> lu( J.transpose() );
    // const VG tmp = lu.solve( w_basis );
    const VG tmp = J * w_basis;
    pi = tmp.segment(0, NA);
    pi0 = tmp[NA];
    rho = tmp.segment(NA+1, NA);
    rho0 = tmp[2*NA+1];
  }

  double norm() const {
    double sq = 0.0;
    sq += pi.squaredNorm();
    sq += pi0*pi0;
    sq += rho.squaredNorm();
    sq += rho0*rho0;
    return std::sqrt(sq);
  }

  double square() const {
    double sq = 0.0;
    sq += pi.squaredNorm();
    sq += pi0*pi0;
    sq += rho.squaredNorm();
    sq += rho0*rho0;
    return std::sqrt(sq);
  }

  Force& operator=(const Force& other) {
    if (this == &other) return *this;
    pi = other.pi;
    pi0 = other.pi0;
    rho = other.rho;
    rho0 = other.rho0;
    return *this;
  }

  MC get_pi() const {
    MC res;
    for(int a=0; a<NA; a++) res += pi(a)*t[a];
    return res;
  }

  inline void update_pi( const MC& piM ) {
    for(int a=0; a<NA; a++) pi(a) = (piM*t[a]).trace().real();
  }

  // void Lmult(const MR& mat) {
  //   pi = mat*pi;
  // }
  // void invert(const MG& mat) {
  //   const Eigen::PartialPivLU<MG> llt(mat);
  //   pi = llt.solve(this->pi);
  // }

  // inline double operator[](const int i) const { return pi[i]; }
  // inline double& operator[](const int i) { return pi[i]; }
  // inline int size() const { return pi.size(); }

  Force& operator+=(const Force& other){
    pi += other.pi;
    pi0 += other.pi0;
    rho += other.rho;
    rho0 += other.rho0;
    return *this;
  }
  friend Force operator+(Force v, const Force& w) { v += w; return v; }

  Force& operator-=(const Force& other){
    pi -= other.pi;
    pi0 -= other.pi0;
    rho -= other.rho;
    rho0 -= other.rho0;
    return *this;
  }
  friend Force operator-(Force v, const Force& w) { v -= w; return v; }

  Force& operator*=(const double a){
    pi *= a;
    pi0 *= a;
    rho *= a;
    rho0 *= a;
    return *this;
  }

  friend Force operator*(const double a, Force v) {
    v.pi *= a;
    v.pi0 *= a;
    v.rho *= a;
    v.rho0 *= a;
    return v; }
  // friend Force operator*(const MG& mat, Force v) { v.pi = mat*v.pi; return v; }

  friend std::ostream& operator<<(std::ostream& os, const Force& v) { os << v.pi; return os; }
};




template<class V=ForceSingleLink>
struct ForceField {
  using Force=ForceField;
  // using V=ForceSingleLink;
  // using V=ForceSingleLink;


  const Lattice& lattice;
  std::vector<V> field;

  ForceField( const Lattice& lattice )
    : lattice( lattice )
    , field( lattice.n_links() )
  {}

  inline V operator[](const Idx il) const { return field[il]; }
  inline V& operator[](const Idx il) { return field[il]; }

  V operator()(const Idx ix, const int mu) const {
    assert( mu>=0 );
    return field[ DIM*ix + mu];
  }
  V& operator()(const Idx ix, const int mu) {
    assert( mu>=0 );
    return field[ DIM*ix + mu];
  }

  V operator()(const Coord& x, const int mu) const {
    assert( mu>=0 );
    return field[ DIM*lattice.idx(x) + mu];
  }
  V& operator()(const Coord& x, const int mu) {
    assert( mu>=0 );
    return field[ DIM*lattice.idx(x) + mu];
  }

  Force& operator=(const Force& other) {
    if (this == &other) return *this;
    assert(&lattice==&other.lattice);
    field = other.field;
    return *this;
  }

  Force& operator+=(const Force& rhs){
    assert(&lattice==&rhs.lattice);
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(Idx il=0; il<lattice.n_links(); il++) field[il] += rhs.field[il];
    return *this;
  }
  friend Force operator+(Force v, const Force& w) { v += w; return v; }

  Force& operator-=(const Force& rhs){
    assert(&lattice==&rhs.lattice);
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(Idx il=0; il<lattice.n_links(); il++) field[il] -= rhs.field[il];
    return *this;
  }
  friend Force operator-(Force v, const Force& w) { v -= w; return v; }


  Force& operator*=(const double a){
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(Idx il=0; il<lattice.n_links(); il++) field[il] *= a;
    return *this;
  }
  friend Force operator*(const double a, Force v) {
    v *= a; return v;
  }

  Force& operator/=(const double a){
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(Idx il=0; il<lattice.n_links(); il++) field[il] /= a;
    return *this;
  }
  friend Force operator/(const double a, Force v) {
    v /= a; return v;
  }

  double norm() const {
    double res = 0.0;

    std::vector<double> tmp( lattice.n_links() );
#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(Idx i=0; i<lattice.n_links(); i++) tmp[i] = field[i].norm();

    for(auto elem : tmp ) res += elem;
    res /= std::sqrt( lattice.n_links() );
    return res;
  }




};



// struct MatrixForceLink{
//   using Force = MatrixForceLink;

//   using Complex = std::complex<double>;
//   using MC = Eigen::Matrix<Complex, Nc, Nc, Eigen::RowMajor>;
//   // using VG = Eigen::Matrix<double, NG, 1>;
//   // using MG = Eigen::Matrix<double, NG, NG, Eigen::RowMajor>;

//   MC pi;

//   MatrixForceLink() { pi = MC::Zero(); }

//   MatrixForceLink(const MC& pi_) { pi = pi_; }

//   MatrixForceLink(const MatrixForceLink& other)
//     : pi(other.pi)
//   {}

//   inline double norm() const { return pi.norm(); }

//   Force& operator=(const Force& other) {
//     if (this == &other) return *this;
//     pi = other.pi;
//     return *this;
//   }

//   // void Lmult(const MR& mat) {
//   //   pi = mat*pi;
//   // }
//   // void invert(const MG& mat) {
//   //   const Eigen::PartialPivLU<MG> llt(mat);
//   //   pi = llt.solve(this->pi);
//   // }

//   // inline double operator[](const int i) const { return pi[i]; }
//   // inline double& operator[](const int i) { return pi[i]; }
//   // inline int size() const { return pi.size(); }

//   Force& operator+=(const Force& rhs){
//     pi += rhs.pi;
//     return *this;
//   }
//   friend Force operator+(Force v, const Force& w) { v += w; return v; }

//   Force& operator-=(const Force& rhs){
//     pi -= rhs.pi;
//     return *this;
//   }
//   friend Force operator-(Force v, const Force& w) { v -= w; return v; }

//   Force& operator*=(const double a){
//     pi *= a;
//     return *this;
//   }

//   Force& operator/=(const double a){
//     pi /= a;
//     return *this;
//   }

//   friend Force operator*(const double a, Force v) { v.pi *= a; return v; }
//   friend Force operator*(const MC& mat, Force v) {
//     v.pi = mat * v.pi;
//     return v;
//   }
//   friend Force operator*( Force v, const MC& mat ) {
//     v.pi = v.pi * mat;
//     return v;
//   }

//   friend std::ostream& operator<<(std::ostream& os, const Force& v) { os << v.pi; return os; }
// };
