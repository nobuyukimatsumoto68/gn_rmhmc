#pragma once

#include <array>

struct Lattice {
  const Coord size;
  const Idx vol;

  Lattice( const Coord& size )
    : size(size)
    , vol(get_volume(size))
  {}

  Idx get_volume( const Coord& size ) const {
    int res = 1;
    assert( size.size()==DIM );
    for(auto elem : size) res *= elem;
    return res;
  }

  inline Idx n_sites() const { return vol; }
  inline Idx n_links() const { return vol*DIM; }

  Coord get_coord( const Idx i ) const {
    Coord x;
    Idx tmp = i;
    for(int mu=DIM-1; mu>=0; mu--){
      x[mu] = tmp%size[mu];
      tmp /= size[mu];
    }
    return x;
  }

  inline int operator[](const int mu) const { return size[mu]; }

  Coord cshift( const Coord& x, const int mu ) const {
    Coord xn( x );
    const int sign = 2*(mu>=0)-1;
    const int mu0 = (sign==1) ? mu : -mu-1;
    xn[mu0] = (xn[mu0]+sign+size[mu0]) % size[mu0];
    return xn;
  }

  inline Idx cshift( const Idx& ix, const int mu ) const {
    return idx(cshift( get_coord(ix), mu ));
  }


  Idx idx(const Coord& x) const {
    Idx res = 0;
    for(int mu=0; mu<DIM; mu++) res = res*size[mu] + x[mu];
    return res;
  }

};
