#pragma once

#include <random>

struct SingleRng {
  std::mt19937_64 mt;

  SingleRng( const int seed_=0){
    seed( seed_ );
  };

  std::normal_distribution<double> dist_gaussian;
  inline double gaussian(){ return dist_gaussian(mt); }

  std::uniform_real_distribution<> dist_01;
  inline double uniform(){ return dist_01(mt); }

  void seed(const int seed) { mt.seed(seed); }
};


struct ParallelRngLink {
  using Rng = SingleRng;

  const Lattice& lattice;
  std::mt19937_64 mt;
  std::vector<Rng> field;

  ParallelRngLink(const Lattice& lattice_, const int seed_=0 )
    : lattice(lattice_)
    , field( lattice.n_links() )
  {
    seed( seed_ );
  }

  inline Rng operator[](const Idx il) const { return field[il]; }
  inline Rng& operator[](const Idx il) { return field[il]; }

  std::normal_distribution<double> dist_gaussian;
  double gaussian(){ return dist_gaussian(mt); }

  std::uniform_real_distribution<> dist_01;
  double uniform(){ return dist_01(mt); }

  void seed(const int seed_) {
    mt.seed(seed_);
    for(auto& elem : field) elem.seed( mt() );
  }
};
