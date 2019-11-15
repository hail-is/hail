package is.hail.linalg

import java.lang.reflect.Method

import com.sun.jna.{FunctionMapper, Library, Native, NativeLibrary, Pointer}
import com.sun.jna.ptr.{DoubleByReference, IntByReference}

import scala.util.{Failure, Success, Try}
import is.hail.utils._

class UnderscoreFunctionMapper extends FunctionMapper {
  override def getFunctionName(library: NativeLibrary, method: Method): String = {
    method.getName() + "_"
  }
}

object TryAll {
  def apply[K](f: => K): Try[K] =
    try {
      Success(f)
    } catch {
      case e: Throwable => Failure(e)
    }
}

object LAPACKLibrary {
  lazy val instance = {
    val standard = Native.loadLibrary("lapack", classOf[LAPACKLibrary]).asInstanceOf[LAPACKLibrary]

    versionTest(standard) match {
      case Success(version) =>
        log.info(s"Imported LAPACK version ${version} with standard names")
        standard
      case Failure(exception) =>
        val underscoreAfterMap = new java.util.HashMap[String, FunctionMapper]()
        underscoreAfterMap.put(Library.OPTION_FUNCTION_MAPPER, new UnderscoreFunctionMapper)
        val underscoreAfter = Native.loadLibrary("lapack", classOf[LAPACKLibrary], underscoreAfterMap).asInstanceOf[LAPACKLibrary]
        versionTest(underscoreAfter) match {
          case Success(version) =>
            log.info(s"Imported LAPACK version ${version} with underscore names")
            underscoreAfter
          case Failure(exception) =>
            throw exception
        }
    }
  }
  def getInstance() = instance

  private def versionTest(libInstance: LAPACKLibrary): Try[String] = {
    val major = new IntByReference()
    val minor = new IntByReference()
    val patch = new IntByReference()

    TryAll({
      libInstance.ilaver(major, minor, patch)
      s"${major.getValue}.${minor.getValue}.${patch.getValue}"
    })
  }
}

trait LAPACKLibrary extends Library {
  def dsyevd(JOBZ: Char, UPLO: Char, N: Int, A: Pointer, LDA: Int, W: Pointer, WORK: Pointer, LWORK: Int, IWORK: Pointer, LIWORK: Int, INFO: Int)
  def dlapy2(X: DoubleByReference, Y: DoubleByReference): Double
  def dgeqrf(M: Long, N: Long, A: Long, LDA: Long, TAU: Long, WORK: Long, LWORK: Long, INFO: Long)
  def ilaver(MAJOR: IntByReference, MINOR: IntByReference, PATCH: IntByReference)
}
//
//trait LAPACKLibraryUnderscore extends Library {
//  def dgeqrf_(M: Long, N: Long, A: Long, LDA: Long, TAU: Long, WORK: Long, LWORK: Long, INFO: Long)
//  def ilaver_(MAJOR: IntByReference, MINOR: IntByReference, PATCH: IntByReference)
//}