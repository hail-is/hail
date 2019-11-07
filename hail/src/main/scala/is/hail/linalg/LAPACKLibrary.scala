package is.hail.linalg

import com.sun.jna.{Library, Native, Pointer}
import com.sun.jna.ptr.DoubleByReference

object LAPACKLibrary {
  val instance = Native.loadLibrary("lapack", classOf[LAPACKLibrary]).asInstanceOf[LAPACKLibrary]
}

trait LAPACKLibrary extends Library {
  def dsyevd(JOBZ: Char, UPLO: Char, N: Int, A: Pointer, LDA: Int, W: Pointer, WORK: Pointer, LWORK: Int, IWORK: Pointer, LIWORK: Int, INFO: Int)
  //def disnan(DIN: Double)
  def dlapy2(X: DoubleByReference, Y: DoubleByReference): Double
  def dgeqrf(M: Long, N: Long, A: Long, LDA: Long, TAU: Long, WORK: Long, LWORK: Long, INFO: Long)
}