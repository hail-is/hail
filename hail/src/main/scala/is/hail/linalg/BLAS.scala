package is.hail.linalg

import is.hail.utils._

import java.util.function._
import scala.util.{Failure, Success, Try}

import com.sun.jna.{FunctionMapper, Library, Native}
import com.sun.jna.ptr.{DoubleByReference, FloatByReference, IntByReference}

object BLAS {
  private[this] val libraryInstance = ThreadLocal.withInitial(new Supplier[BLASLibrary]() {
    def get() = {
      val standard = Native.load("blas", classOf[BLASLibrary]).asInstanceOf[BLASLibrary]

      verificationTest(standard) match {
        case Success(_) =>
          log.info("Imported BLAS with standard names")
          standard
        case Failure(_) =>
          val underscoreAfterMap = new java.util.HashMap[String, FunctionMapper]()
          underscoreAfterMap.put(Library.OPTION_FUNCTION_MAPPER, new UnderscoreFunctionMapper)
          val underscoreAfter =
            Native.load("blas", classOf[BLASLibrary], underscoreAfterMap).asInstanceOf[BLASLibrary]
          verificationTest(underscoreAfter) match {
            case Success(_) =>
              log.info("Imported BLAS with underscore names")
              underscoreAfter
            case Failure(exception) =>
              throw exception
          }
      }
    }
  })

  private def verificationTest(libInstance: BLASLibrary): Try[Unit] = {
    val n = new IntByReference(2)
    val incx = new IntByReference(1)
    val x = Array(3.0, 4.0)
    TryAll {
      val norm = libInstance.dnrm2(n, x, incx)
      assert(Math.abs(norm - 5.0) < .1)
    }
  }

  def dcopy(n: Int, X: Long, incX: Int, Y: Long, incY: Int): Unit = {
    val nInt = new IntByReference(n)
    val incXInt = new IntByReference(incX)
    val incYInt = new IntByReference(incY)

    libraryInstance.get.dcopy(nInt, X, incXInt, Y, incYInt)
  }

  def dscal(n: Int, alpha: Double, X: Long, incX: Int): Unit = {
    val nInt = new IntByReference(n)
    val alphaDouble = new DoubleByReference(alpha)
    val incXInt = new IntByReference(incX)

    libraryInstance.get.dscal(nInt, alphaDouble, X, incXInt)
  }

  def dgemv(
    TRANS: String,
    M: Int,
    N: Int,
    ALPHA: Double,
    A: Long,
    LDA: Int,
    X: Long,
    INCX: Int,
    BETA: Double,
    Y: Long,
    INCY: Int,
  ): Unit = {
    val mInt = new IntByReference(M)
    val nInt = new IntByReference(N)
    val alphaDouble = new DoubleByReference(ALPHA)
    val LDAInt = new IntByReference(LDA)
    val betaDouble = new DoubleByReference(BETA)
    val incxInt = new IntByReference(INCX)
    val incyInt = new IntByReference(INCY)

    libraryInstance.get.dgemv(TRANS, mInt, nInt, alphaDouble, A, LDAInt, X, incxInt, betaDouble, Y,
      incyInt)
  }

  def sgemm(
    TRANSA: String,
    TRANSB: String,
    M: Int,
    N: Int,
    K: Int,
    ALPHA: Float,
    A: Long,
    LDA: Int,
    B: Long,
    LDB: Int,
    BETA: Float,
    C: Long,
    LDC: Int,
  ) = {
    val mInt = new IntByReference(M)
    val nInt = new IntByReference(N)
    val kInt = new IntByReference(K)
    val alphaDouble = new FloatByReference(ALPHA)
    val LDAInt = new IntByReference(LDA)
    val LDBInt = new IntByReference(LDB)
    val betaDouble = new FloatByReference(BETA)
    val LDCInt = new IntByReference(LDC)

    libraryInstance.get.sgemm(TRANSA, TRANSB, mInt, nInt, kInt, alphaDouble, A, LDAInt, B, LDBInt,
      betaDouble, C, LDCInt)
  }

  def dgemm(
    TRANSA: String,
    TRANSB: String,
    M: Int,
    N: Int,
    K: Int,
    ALPHA: Double,
    A: Long,
    LDA: Int,
    B: Long,
    LDB: Int,
    BETA: Double,
    C: Long,
    LDC: Int,
  ) = {
    val mInt = new IntByReference(M)
    val nInt = new IntByReference(N)
    val kInt = new IntByReference(K)
    val alphaDouble = new DoubleByReference(ALPHA)
    val LDAInt = new IntByReference(LDA)
    val LDBInt = new IntByReference(LDB)
    val betaDouble = new DoubleByReference(BETA)
    val LDCInt = new IntByReference(LDC)

    libraryInstance.get.dgemm(TRANSA, TRANSB, mInt, nInt, kInt, alphaDouble, A, LDAInt, B, LDBInt,
      betaDouble, C, LDCInt)
  }

  def dtrmm(
    side: String,
    uplo: String,
    transA: String,
    diag: String,
    m: Int,
    n: Int,
    alpha: Double,
    A: Long,
    ldA: Int,
    B: Long,
    ldB: Int,
  ) = {
    val mInt = new IntByReference(m)
    val nInt = new IntByReference(n)
    val alphaDouble = new DoubleByReference(alpha)
    val ldAInt = new IntByReference(ldA)
    val ldBInt = new IntByReference(ldB)

    libraryInstance.get.dtrmm(side, uplo, transA, diag, mInt, nInt, alphaDouble, A, ldAInt, B,
      ldBInt)
  }
}

trait BLASLibrary extends Library {
  def dcopy(n: IntByReference, X: Long, incX: IntByReference, Y: Long, incY: IntByReference)
  def dscal(n: IntByReference, alpha: DoubleByReference, X: Long, incX: IntByReference)

  def dgemv(
    TRANS: String,
    M: IntByReference,
    N: IntByReference,
    ALPHA: DoubleByReference,
    A: Long,
    LDA: IntByReference,
    X: Long,
    INCX: IntByReference,
    BETA: DoubleByReference,
    Y: Long,
    INCY: IntByReference,
  )

  def sgemm(
    TRANSA: String,
    TRANSB: String,
    M: IntByReference,
    N: IntByReference,
    K: IntByReference,
    ALPHA: FloatByReference,
    A: Long,
    LDA: IntByReference,
    B: Long,
    LDB: IntByReference,
    BETA: FloatByReference,
    C: Long,
    LDC: IntByReference,
  )

  def dgemm(
    TRANSA: String,
    TRANSB: String,
    M: IntByReference,
    N: IntByReference,
    K: IntByReference,
    ALPHA: DoubleByReference,
    A: Long,
    LDA: IntByReference,
    B: Long,
    LDB: IntByReference,
    BETA: DoubleByReference,
    C: Long,
    LDC: IntByReference,
  )

  def dtrmm(
    side: String,
    uplo: String,
    transA: String,
    diag: String,
    m: IntByReference,
    n: IntByReference,
    alpha: DoubleByReference,
    A: Long,
    ldA: IntByReference,
    B: Long,
    ldB: IntByReference,
  )

  def dnrm2(N: IntByReference, X: Array[Double], INCX: IntByReference): Double
}
