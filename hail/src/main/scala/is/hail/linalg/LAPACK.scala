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

object LAPACK {
  lazy val libraryInstance = {
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

  def dgeqrf(M: Int, N: Int, A: Long, LDA: Int, TAU: Long, WORK: Long, LWORK: Int): Int = {
    val mInt = new IntByReference(M)
    val nInt = new IntByReference(N)
    val LDAInt = new IntByReference(LDA)
    val LWORKInt = new IntByReference(LWORK)
    val infoInt = new IntByReference(1)
    libraryInstance.dgeqrf(mInt, nInt, A, LDAInt, TAU, WORK, LWORKInt, infoInt)
    infoInt.getValue()
  }

  def dorgqr(M: Int, N: Int, K: Int, A: Long, LDA: Int, TAU: Long, WORK: Long, LWORK: Int): Int = {
    val mInt = new IntByReference(M)
    val nInt = new IntByReference(N)
    val kInt = new IntByReference(K)
    val LDAInt = new IntByReference(LDA)
    val LWORKInt = new IntByReference(LWORK)
    val infoInt = new IntByReference(1)
    libraryInstance.dorgqr(mInt, nInt, kInt, A, LDAInt, TAU, WORK, LWORKInt, infoInt)
    infoInt.getValue()
  }

  private def versionTest(libInstance: LAPACKLibrary): Try[String] = {
    val major = new IntByReference()
    val minor = new IntByReference()
    val patch = new IntByReference()

    TryAll {
      libInstance.ilaver(major, minor, patch)
      s"${major.getValue}.${minor.getValue}.${patch.getValue}"
    }
  }
}

trait LAPACKLibrary extends Library {
  def dgeqrf(M: IntByReference, N: IntByReference, A: Long, LDA: IntByReference, TAU: Long, WORK: Long, LWORK: IntByReference, INFO: IntByReference)
  def dorgqr(M: IntByReference, N: IntByReference, K: IntByReference, A: Long, LDA: IntByReference, TAU: Long, WORK: Long, LWORK: IntByReference, Info: IntByReference)
  def ilaver(MAJOR: IntByReference, MINOR: IntByReference, PATCH: IntByReference)
}