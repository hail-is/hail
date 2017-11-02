package is.hail.utils

package object lensing {
  def on12[X,Y,Z](f: (X) => Z): ((X,Y)) => (Z,Y) =
    { case (x,y) => (f(x), y) }
  def on22[X,Y,Z](f: (Y) => Z): ((X,Y)) => (X,Z) =
    { case (x,y) => (x, f(y)) }
  def on13[A,B,C,D](f: (A) => D): ((A,B,C)) => (D,B,C) =
    { case (a,b,c) => (f(a), b, c) }
  def on23[A,B,C,D](f: (B) => D): ((A,B,C)) => (A,D,C) =
    { case (a,b,c) => (a, f(b), c) }
  def on33[A,B,C,D](f: (C) => D): ((A,B,C)) => (A,B,D) =
    { case (a,b,c) => (a, b, f(c)) }
  def onBoth[A,B](f: (A) => B): ((A,A)) => (B,B) =
    { case (x,y) => (f(x), f(y)) }
}
