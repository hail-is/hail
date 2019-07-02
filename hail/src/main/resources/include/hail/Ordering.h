#ifndef HAIL_ORDERING_H
#define HAIL_ORDERING_H 1

#include <cmath>
#include <cstring>

namespace hail {

  class IntOrd {
  public:
    using T = int;

    static bool lt(int l, int r) { return l < r; }

    static bool lteq(int l, int r) { return l <= r; }

    static bool gt(int l, int r) { return l > r; }

    static bool gteq(int l, int r) { return l >= r; }

    static bool eq(int l, int r) { return l == r; }

    static bool neq(int l, int r) { return l != r; }

    static int compare(int l, int r) {
      return l < r ? -1 : (l > r ? 1 : 0);
    }
  };

  class LongOrd {
  public:
    using T = long;

    static bool lt(long l, long r) { return l < r; }

    static bool lteq(long l, long r) { return l <= r; }

    static bool gt(long l, long r) { return l > r; }

    static bool gteq(long l, long r) { return l >= r; }

    static bool eq(long l, long r) { return l == r; }

    static bool neq(long l, long r) { return l != r; }

    static int compare(long l, long r) {
      return l < r ? -1 : (l > r ? 1 : 0);
    }
  };

  static inline int float_to_int_bits(float f) {
    return std::isnan(f) ? 0x7fc00000 : *reinterpret_cast<int *>(&f);
  }

  static inline long double_to_long_bits(double d) {
    return std::isnan(d) ? 0x7ff8000000000000l : *reinterpret_cast<long *>(&d);
  }

  class FloatOrd {
  public:
    using T = float;

    static bool lt(float l, float r) { return l < r; }

    static bool lteq(float l, float r) { return l <= r; }

    static bool gt(float l, float r) { return l > r; }

    static bool gteq(float l, float r) { return l >= r; }

    static bool eq(float l, float r) { return l == r; }

    static bool neq(float l, float r) { return l != r; }

    static int compare(float l, float r) {
      if (l < r)
        return -1;
      if (l > r)
        return 1;

      return IntOrd::compare(float_to_int_bits(l), float_to_int_bits(r));
    }
  };

  class DoubleOrd {
  public:
    using T = double;

    static bool lt(double l, double r) { return l < r; }

    static bool lteq(double l, double r) { return l <= r; }

    static bool gt(double l, double r) { return l > r; }

    static bool gteq(double l, double r) { return l >= r; }

    static bool eq(double l, double r) { return l == r; }

    static bool neq(double l, double r) { return l != r; }

    static int compare(double l, double r) {
      if (l < r)
        return -1;
      if (l > r)
        return 1;

      return LongOrd::compare(double_to_long_bits(l), double_to_long_bits(r));
    }
  };

  class BoolOrd {
  public:
    using T = bool;

    static bool lt(bool l, bool r) { return l < r; }

    static bool lteq(bool l, bool r) { return l <= r; }

    static bool gt(bool l, bool r) { return l > r; }

    static bool gteq(bool l, bool r) { return l >= r; }

    static bool eq(bool l, bool r) { return l == r; }

    static bool neq(bool l, bool r) { return l != r; }

    static int compare(bool l, bool r) {
      return l < r ? -1 : (l > r ? 1 : 0);
    }
  };

  template<typename Comp> class CompareOrd {
  public:
    using T = typename Comp::T;

    static bool lt(T l, T r) { return Comp::compare(l, r) < 0; }

    static bool lteq(T l, T r) { return Comp::compare(l, r) <= 0; }

    static bool gt(T l, T r) { return Comp::compare(l, r) > 0; }

    static bool gteq(T l, T r) { return Comp::compare(l, r) >= 0; }

    static bool eq(T l, T r) { return Comp::compare(l, r) == 0; }

    static bool neq(T l, T r) { return Comp::compare(l, r) != 0; }

    static int compare(T l, T r) { return Comp::compare(l, r); }
  };

  class BinaryCompare {
  public:
    using T = const char *;

    static int compare(const char *l, const char *r) {
      int llen = *(const int *)l;
      int rlen = *(const int *)r;

      int c = memcmp(l + 4, r + 4, llen < rlen ? llen : rlen);
      if (c != 0)
        return c;

      if (llen < rlen)
        return -1;
      else if (llen > rlen)
        return 1;
      else
        return 0;
    }
  };

  using BinaryOrd = CompareOrd<BinaryCompare>;

  template<typename Ord> class ExtOrd {
  public:
    using T = typename Ord::T;

    static bool lt(bool lm, T l, bool rm, T r) {
      if (lm)
        return false;
      else if (rm)
        return true;
      else
        return Ord::lt(l, r);
    }

    static bool lteq(bool lm, T l, bool rm, T r) {
      if (lm) {
        if (rm)
          return true;
        else
          return false;
      } else if (rm)
        return true;
      else
        return Ord::lteq(l, r);
    }

    // FIXME
    static bool gt(bool lm, T l, bool rm, T r) { return lt(rm, r, lm, l); }

    static bool gteq(bool lm, T l, bool rm, T r) { return lteq(rm, r, lm, l); }

    static bool eq(bool lm, T l, bool rm, T r) {
      if (lm)
        return rm;
      else
        return !rm && Ord::eq(l, r);
    }

    static bool neq(bool lm, T l, bool rm, T r) { return !eq(lm, l, rm, r); }

    static int compare(bool lm, T l, bool rm, T r) {
      if (lm) {
        if (rm)
          return 0;
        else
          return 1;
      } else {
        if (rm)
          return -1;
        else
          return Ord::compare(l, r);
      }
    }
  };

  template<typename AL, typename AR, typename ElemOrd>
  class ArrayOrd {
  public:
    using T = const char *;

    static bool lt(const char *l, const char *r) {
      int ln = AL::load_length(l);
      int rn = AR::load_length(r);
      int n = ln < rn ? ln : rn;
      for (int i = 0; i < n; ++i) {
        bool lm = AL::is_element_missing(l, i);
        typename ElemOrd::T lx;
        if (!lm)
          lx = AL::load_element(l, i);
        bool rm = AR::is_element_missing(r, i);
        typename ElemOrd::T rx;
        if (!rm)
          rx = AR::load_element(r, i);

        if (ElemOrd::lt(lm, lx, rm, rx))
          return true;
        if (!ElemOrd::eq(lm, lx, rm, rx))
          return false;
      }

      return ln < rn;
    }

    static bool lteq(const char *l, const char *r) {
      int ln = AL::load_length(l);
      int rn = AR::load_length(r);
      int n = ln < rn ? ln : rn;
      for (int i = 0; i < n; ++i) {
        bool lm = AL::is_element_missing(l, i);
        typename ElemOrd::T lx;
        if (!lm)
          lx = AL::load_element(l, i);
        bool rm = AR::is_element_missing(r, i);
        typename ElemOrd::T rx;
        if (!rm)
          rx = AR::load_element(r, i);

        if (ElemOrd::lt(lm, lx, rm, rx))
          return true;
        if (!ElemOrd::eq(lm, lx, rm, rx))
          return false;
      }

      return ln <= rn;
    }

    static bool gt(const char *l, const char *r) {
      int ln = AL::load_length(l);
      int rn = AR::load_length(r);
      int n = ln < rn ? ln : rn;
      for (int i = 0; i < n; ++i) {
        bool lm = AL::is_element_missing(l, i);
        typename ElemOrd::T lx;
        if (!lm)
          lx = AL::load_element(l, i);
        bool rm = AR::is_element_missing(r, i);
        typename ElemOrd::T rx;
        if (!rm)
          rx = AR::load_element(r, i);

        if (ElemOrd::gt(lm, lx, rm, rx))
          return true;
        if (!ElemOrd::eq(lm, lx, rm, rx))
          return false;
      }

      return ln > rn;
    }

    static bool gteq(const char *l, const char *r) {
      int ln = AL::load_length(l);
      int rn = AR::load_length(r);
      int n = ln < rn ? ln : rn;
      for (int i = 0; i < n; ++i) {
        bool lm = AL::is_element_missing(l, i);
        typename ElemOrd::T lx;
        if (!lm)
          lx = AL::load_element(l, i);
        bool rm = AR::is_element_missing(r, i);
        typename ElemOrd::T rx;
        if (!rm)
          rx = AR::load_element(r, i);

        if (ElemOrd::gt(lm, lx, rm, rx))
          return true;
        if (!ElemOrd::eq(lm, lx, rm, rx))
          return false;
      }

      return ln >= rn;
    }

    static bool eq(const char *l, const char *r) {
      int ln = AL::load_length(l);
      int rn = AR::load_length(r);
      int n = ln < rn ? ln : rn;
      for (int i = 0; i < n; ++i) {
        bool lm = AL::is_element_missing(l, i);
        typename ElemOrd::T lx;
        if (!lm)
          lx = AL::load_element(l, i);
        bool rm = AR::is_element_missing(r, i);
        typename ElemOrd::T rx;
        if (!rm)
          rx = AR::load_element(r, i);

        if (!ElemOrd::eq(lm, lx, rm, rx))
          return false;
      }

      return ln == rn;
    }

    static bool neq(const char *l, const char *r) { return !eq(l, r); }

    static int compare(const char *l, const char *r) {
      int ln = AL::load_length(l);
      int rn = AR::load_length(r);
      int n = ln < rn ? ln : rn;
      for (int i = 0; i < n; ++i) {
        bool lm = AL::is_element_missing(l, i);
        typename ElemOrd::T lx;
        if (!lm)
          lx = AL::load_element(l, i);
        bool rm = AR::is_element_missing(r, i);
        typename ElemOrd::T rx;
        if (!rm)
          rx = AR::load_element(r, i);

        int c = ElemOrd::compare(lm, lx, rm, rx);
        if (c)
          return c;
      }

      return IntOrd::compare(ln, rn);
    }
  };

}

#endif
