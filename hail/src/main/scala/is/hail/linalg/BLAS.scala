package is.hail.linalg

import com.sun.jna.{FunctionMapper, Library, Native}
import com.sun.jna.ptr.{DoubleByReference, IntByReference}
import is.hail.utils._

import scala.util.{Failure, Success, Try}


object BLAS {
  lazy val libraryInstance = {
    val standard = Native.loadLibrary("blas", classOf[BLASLibrary]).asInstanceOf[BLASLibrary]

    verificationTest(standard) match {
      case Success(_) =>
        log.info("Imported BLAS with standard names")
        standard
      case Failure(exc) =>
        val underscoreAfterMap = new java.util.HashMap[String, FunctionMapper]()
        underscoreAfterMap.put(Library.OPTION_FUNCTION_MAPPER, new UnderscoreFunctionMapper)
        val underscoreAfter = Native.loadLibrary("blas", classOf[BLASLibrary], underscoreAfterMap).asInstanceOf[BLASLibrary]
        verificationTest(underscoreAfter) match {
          case Success(_) =>
            log.info("Imported BLAS with underscore names")
            underscoreAfter
          case Failure(exception) =>
            throw exception
        }
    }
  }

  private def verificationTest(libInstance: BLASLibrary): Try[Unit] = {
    val n = new IntByReference(2)
    val incx = new IntByReference(1)
    val x = Array(3.0, 4.0)
    TryAll {
      val norm = libInstance.dnrm2(n, x, incx)
      assert(Math.abs(norm - 5.0) < .1)
    }
  }
}

trait BLASLibrary extends Library {
  // Not clear on types for TRANSA and TRANSB. Maybe they can be regular chars?
  def dgemm(TRANSA: IntByReference, TRANSB: IntByReference, M: IntByReference, N: IntByReference, K: IntByReference,
    ALPHA: DoubleByReference, A: Long, LDA: IntByReference, B: Long, LDB: IntByReference,
    BETA: DoubleByReference, C: Long, LDC: IntByReference)

  def dnrm2(N: IntByReference, X: Array[Double], INCX: IntByReference): Double
}
