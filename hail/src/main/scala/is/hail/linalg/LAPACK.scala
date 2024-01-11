package is.hail.linalg

import is.hail.utils._

import java.lang.reflect.Method
import java.util.function._
import scala.util.{Failure, Success, Try}

import com.sun.jna.{FunctionMapper, Library, Native, NativeLibrary}
import com.sun.jna.ptr.{DoubleByReference, IntByReference}

class UnderscoreFunctionMapper extends FunctionMapper {
  override def getFunctionName(library: NativeLibrary, method: Method): String =
    method.getName() + "_"
}

// ALL LAPACK C function args must be passed by address, not value
// see: https://software.intel.com/content/www/us/en/develop/documentation/mkl-linux-developer-guide/top/language-specific-usage-options/mixed-language-programming-with-the-intel-math-kernel-library/calling-lapack-blas-and-cblas-routines-from-c-c-language-environments.html
object LAPACK {
  private[this] val libraryInstance = ThreadLocal.withInitial(new Supplier[LAPACKLibrary]() {
    def get() = {
      val mklEnvVar = "HAIL_MKL_PATH"
      val openblasEnvVar = "HAIL_OPENBLAS_PATH"
      val libraryName = if (sys.env.contains(mklEnvVar)) {
        val mklDir = sys.env(mklEnvVar)
        val mklLibName = "mkl_rt"
        NativeLibrary.addSearchPath(mklLibName, mklDir)
        mklLibName
      } else if (sys.env.contains(openblasEnvVar)) {
        val openblasDir = sys.env(openblasEnvVar)
        val openblasLibName = "lapack"
        NativeLibrary.addSearchPath(openblasLibName, openblasDir)
        openblasLibName
      } else {
        "lapack"
      }
      val standard = Native.load(libraryName, classOf[LAPACKLibrary])

      versionTest(standard) match {
        case Success(version) =>
          log.info(s"Imported LAPACK library $libraryName, version $version, with standard names")
          standard
        case Failure(_) =>
          val underscoreAfterMap = new java.util.HashMap[String, FunctionMapper]()
          underscoreAfterMap.put(Library.OPTION_FUNCTION_MAPPER, new UnderscoreFunctionMapper)
          val underscoreAfter = Native.load(libraryName, classOf[LAPACKLibrary], underscoreAfterMap)
          versionTest(underscoreAfter) match {
            case Success(version) =>
              log.info(
                s"Imported LAPACK library $libraryName, version $version, with underscore names"
              )
              underscoreAfter
            case Failure(exception) =>
              throw exception
          }
      }
    }
  })

  def dgeqrf(M: Int, N: Int, A: Long, LDA: Int, TAU: Long, WORK: Long, LWORK: Int): Int = {
    val mInt = new IntByReference(M)
    val nInt = new IntByReference(N)
    val LDAInt = new IntByReference(LDA)
    val LWORKInt = new IntByReference(LWORK)
    val infoInt = new IntByReference(1)
    libraryInstance.get.dgeqrf(mInt, nInt, A, LDAInt, TAU, WORK, LWORKInt, infoInt)
    infoInt.getValue()
  }

  def dorgqr(M: Int, N: Int, K: Int, A: Long, LDA: Int, TAU: Long, WORK: Long, LWORK: Int): Int = {
    val mInt = new IntByReference(M)
    val nInt = new IntByReference(N)
    val kInt = new IntByReference(K)
    val LDAInt = new IntByReference(LDA)
    val LWORKInt = new IntByReference(LWORK)
    val infoInt = new IntByReference(1)
    libraryInstance.get.dorgqr(mInt, nInt, kInt, A, LDAInt, TAU, WORK, LWORKInt, infoInt)
    infoInt.getValue()
  }

  def dgeqrt(m: Int, n: Int, nb: Int, A: Long, ldA: Int, T: Long, ldT: Int, work: Long): Int = {
    val mInt = new IntByReference(m)
    val nInt = new IntByReference(n)
    val nbInt = new IntByReference(nb)
    val ldAInt = new IntByReference(ldA)
    val ldTInt = new IntByReference(ldT)
    val infoInt = new IntByReference(1)
    libraryInstance.get.dgeqrt(mInt, nInt, nbInt, A, ldAInt, T, ldTInt, work, infoInt)
    infoInt.getValue()
  }

  def dgemqrt(
    side: String,
    trans: String,
    m: Int,
    n: Int,
    k: Int,
    nb: Int,
    V: Long,
    ldV: Int,
    T: Long,
    ldT: Int,
    C: Long,
    ldC: Int,
    work: Long,
  ): Int = {
    val mInt = new IntByReference(m)
    val nInt = new IntByReference(n)
    val kInt = new IntByReference(k)
    val nbInt = new IntByReference(nb)
    val ldVInt = new IntByReference(ldV)
    val ldTInt = new IntByReference(ldT)
    val ldCInt = new IntByReference(ldC)
    val infoInt = new IntByReference(1)
    libraryInstance.get.dgemqrt(side, trans, mInt, nInt, kInt, nbInt, V, ldVInt, T, ldTInt, C,
      ldCInt, work, infoInt)
    infoInt.getValue()
  }

  def dgeqr(m: Int, n: Int, A: Long, ldA: Int, T: Long, Tsize: Int, work: Long, lWork: Int): Int = {
    val mInt = new IntByReference(m)
    val nInt = new IntByReference(n)
    val ldAInt = new IntByReference(ldA)
    val TsizeInt = new IntByReference(Tsize)
    val lWorkInt = new IntByReference(lWork)
    val infoInt = new IntByReference(1)
    libraryInstance.get.dgeqr(mInt, nInt, A, ldAInt, T, TsizeInt, work, lWorkInt, infoInt)
    infoInt.getValue()
  }

  def dgemqr(
    side: String,
    trans: String,
    m: Int,
    n: Int,
    k: Int,
    A: Long,
    ldA: Int,
    T: Long,
    Tsize: Int,
    C: Long,
    ldC: Int,
    work: Long,
    Lwork: Int,
  ): Int = {
    val mInt = new IntByReference(m)
    val nInt = new IntByReference(n)
    val kInt = new IntByReference(k)
    val ldAInt = new IntByReference(ldA)
    val TsizeInt = new IntByReference(Tsize)
    val ldCInt = new IntByReference(ldC)
    val LworkInt = new IntByReference(Lwork)
    val infoInt = new IntByReference(1)
    libraryInstance.get.dgemqr(side, trans, mInt, nInt, kInt, A, ldAInt, T, TsizeInt, C, ldCInt,
      work, LworkInt, infoInt)
    infoInt.getValue()
  }

  def dtpqrt(
    m: Int,
    n: Int,
    l: Int,
    nb: Int,
    A: Long,
    ldA: Int,
    B: Long,
    ldB: Int,
    T: Long,
    ldT: Int,
    work: Long,
  ): Int = {
    val mInt = new IntByReference(m)
    val nInt = new IntByReference(n)
    val lInt = new IntByReference(l)
    val nbInt = new IntByReference(nb)
    val ldAInt = new IntByReference(ldA)
    val ldBInt = new IntByReference(ldB)
    val ldTInt = new IntByReference(ldT)
    val infoInt = new IntByReference(1)
    libraryInstance.get.dtpqrt(mInt, nInt, lInt, nbInt, A, ldAInt, B, ldBInt, T, ldTInt, work,
      infoInt)
    infoInt.getValue()
  }

  def dtpmqrt(
    side: String,
    trans: String,
    m: Int,
    n: Int,
    k: Int,
    l: Int,
    nb: Int,
    V: Long,
    ldV: Int,
    T: Long,
    ldT: Int,
    A: Long,
    ldA: Int,
    B: Long,
    ldB: Int,
    work: Long,
  ): Int = {
    val mInt = new IntByReference(m)
    val nInt = new IntByReference(n)
    val kInt = new IntByReference(k)
    val lInt = new IntByReference(l)
    val nbInt = new IntByReference(nb)
    val ldVInt = new IntByReference(ldV)
    val ldTInt = new IntByReference(ldT)
    val ldAInt = new IntByReference(ldA)
    val ldBInt = new IntByReference(ldB)
    val infoInt = new IntByReference(1)
    libraryInstance.get.dtpmqrt(side, trans, mInt, nInt, kInt, lInt, nbInt, V, ldVInt, T, ldTInt, A,
      ldAInt, B, ldBInt, work, infoInt)
    infoInt.getValue()
  }

  def dgetrf(M: Int, N: Int, A: Long, LDA: Int, IPIV: Long): Int = {
    val Mref = new IntByReference(M)
    val Nref = new IntByReference(N)
    val LDAref = new IntByReference(LDA)
    val INFOref = new IntByReference(1)

    libraryInstance.get.dgetrf(Mref, Nref, A, LDAref, IPIV, INFOref)
    INFOref.getValue()
  }

  def dgetri(N: Int, A: Long, LDA: Int, IPIV: Long, WORK: Long, LWORK: Int): Int = {
    val Nref = new IntByReference(N)
    val LDAref = new IntByReference(LDA)
    val LWORKref = new IntByReference(LWORK)
    val INFOref = new IntByReference(1)

    libraryInstance.get.dgetri(Nref, A, LDAref, IPIV, WORK, LWORKref, INFOref)
    INFOref.getValue()
  }

  def dgesdd(
    JOBZ: String,
    M: Int,
    N: Int,
    A: Long,
    LDA: Int,
    S: Long,
    U: Long,
    LDU: Int,
    VT: Long,
    LDVT: Int,
    WORK: Long,
    LWORK: Int,
    IWORK: Long,
  ): Int = {
    val Mref = new IntByReference(M)
    val Nref = new IntByReference(N)
    val LDAref = new IntByReference(LDA)
    val LDUref = new IntByReference(LDU)
    val LDVTref = new IntByReference(LDVT)
    val LWORKRef = new IntByReference(LWORK)
    val INFOref = new IntByReference(1)

    libraryInstance.get.dgesdd(JOBZ, Mref, Nref, A, LDAref, S, U, LDUref, VT, LDVTref, WORK,
      LWORKRef, IWORK, INFOref)

    INFOref.getValue()
  }

  def dgesv(N: Int, NHRS: Int, A: Long, LDA: Int, IPIV: Long, B: Long, LDB: Int): Int = {
    val Nref = new IntByReference(N)
    val NHRSref = new IntByReference(NHRS)
    val LDAref = new IntByReference(LDA)
    val LDBref = new IntByReference(LDB)
    val INFOref = new IntByReference(1)

    libraryInstance.get.dgesv(Nref, NHRSref, A, LDAref, IPIV, B, LDBref, INFOref)

    INFOref.getValue()
  }

  def dsyevr(
    jobz: String,
    range: String,
    uplo: String,
    n: Int,
    A: Long,
    ldA: Int,
    vl: Double,
    vu: Double,
    il: Int,
    iu: Int,
    abstol: Double,
    W: Long,
    Z: Long,
    ldZ: Int,
    ISuppZ: Long,
    Work: Long,
    lWork: Int,
    IWork: Long,
    lIWork: Int,
  ): Int = {
    val nRef = new IntByReference(n)
    val ldARef = new IntByReference(ldA)
    val vlRef = new DoubleByReference(vl)
    val vuRef = new DoubleByReference(vu)
    val ilRef = new IntByReference(il)
    val iuRef = new IntByReference(iu)
    val abstolRef = new DoubleByReference(abstol)
    val ldZRef = new IntByReference(ldZ)
    val lWorkRef = new IntByReference(lWork)
    val lIWorkRef = new IntByReference(lIWork)
    val INFOref = new IntByReference(1)
    val mRef = new IntByReference(0)

    libraryInstance.get.dsyevr(jobz, range, uplo, nRef, A, ldARef, vlRef, vuRef, ilRef, iuRef,
      abstolRef, mRef, W, Z, ldZRef, ISuppZ, Work, lWorkRef, IWork, lIWorkRef, INFOref)

    INFOref.getValue()
  }

  def dtrtrs(
    UPLO: String,
    TRANS: String,
    DIAG: String,
    N: Int,
    NRHS: Int,
    A: Long,
    LDA: Int,
    B: Long,
    LDB: Int,
  ): Int = {
    val Nref = new IntByReference(N)
    val NRHSref = new IntByReference(NRHS)
    val LDAref = new IntByReference(LDA)
    val LDBref = new IntByReference(LDB)
    val INFOref = new IntByReference(1)
    libraryInstance.get.dtrtrs(UPLO, TRANS, DIAG, Nref, NRHSref, A, LDAref, B, LDBref, INFOref)
    INFOref.getValue()
  }

  def dlacpy(uplo: String, M: Int, N: Int, A: Long, ldA: Int, B: Long, ldB: Int): Unit = {
    val Mref = new IntByReference(M)
    val Nref = new IntByReference(N)
    val ldAref = new IntByReference(ldA)
    val ldBref = new IntByReference(ldB)
    libraryInstance.get.dlacpy(uplo, Mref, Nref, A, ldAref, B, ldBref)
  }

  def ilaenv(ispec: Int, name: String, opts: String, n1: Int, n2: Int, n3: Int, n4: Int): Int = {
    val ispecref = new IntByReference(ispec)
    val n1ref = new IntByReference(n1)
    val n2ref = new IntByReference(n2)
    val n3ref = new IntByReference(n3)
    val n4ref = new IntByReference(n4)

    libraryInstance.get.ilaenv(ispecref, name, opts, n1ref, n2ref, n3ref, n4ref)
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
  def dgesv(
    N: IntByReference,
    NHRS: IntByReference,
    A: Long,
    LDA: IntByReference,
    IPIV: Long,
    B: Long,
    LDB: IntByReference,
    INFO: IntByReference,
  ): Unit

  def dgeqrf(
    M: IntByReference,
    N: IntByReference,
    A: Long,
    LDA: IntByReference,
    TAU: Long,
    WORK: Long,
    LWORK: IntByReference,
    INFO: IntByReference,
  ): Unit

  def dorgqr(
    M: IntByReference,
    N: IntByReference,
    K: IntByReference,
    A: Long,
    LDA: IntByReference,
    TAU: Long,
    WORK: Long,
    LWORK: IntByReference,
    INFO: IntByReference,
  ): Unit

  def dgeqrt(
    m: IntByReference,
    n: IntByReference,
    nb: IntByReference,
    A: Long,
    ldA: IntByReference,
    T: Long,
    ldT: IntByReference,
    work: Long,
    info: IntByReference,
  ): Unit

  def dgemqrt(
    side: String,
    trans: String,
    m: IntByReference,
    n: IntByReference,
    k: IntByReference,
    nb: IntByReference,
    V: Long,
    ldV: IntByReference,
    T: Long,
    ldT: IntByReference,
    C: Long,
    ldC: IntByReference,
    work: Long,
    info: IntByReference,
  ): Unit

  def dgeqr(
    m: IntByReference,
    n: IntByReference,
    A: Long,
    ldA: IntByReference,
    T: Long,
    Tsize: IntByReference,
    work: Long,
    lWork: IntByReference,
    info: IntByReference,
  ): Unit

  def dgemqr(
    side: String,
    trans: String,
    m: IntByReference,
    n: IntByReference,
    k: IntByReference,
    A: Long,
    ldA: IntByReference,
    T: Long,
    Tsize: IntByReference,
    C: Long,
    ldC: IntByReference,
    work: Long,
    Lwork: IntByReference,
    info: IntByReference,
  ): Unit

  def dtpqrt(
    M: IntByReference,
    N: IntByReference,
    L: IntByReference,
    NB: IntByReference,
    A: Long,
    LDA: IntByReference,
    B: Long,
    LDB: IntByReference,
    T: Long,
    LDT: IntByReference,
    WORK: Long,
    INFO: IntByReference,
  ): Unit

  def dtpmqrt(
    side: String,
    trans: String,
    M: IntByReference,
    N: IntByReference,
    K: IntByReference,
    L: IntByReference,
    NB: IntByReference,
    V: Long,
    LDV: IntByReference,
    T: Long,
    LDT: IntByReference,
    A: Long,
    LDA: IntByReference,
    B: Long,
    LDB: IntByReference,
    WORK: Long,
    INFO: IntByReference,
  ): Unit

  def dgetrf(
    M: IntByReference,
    N: IntByReference,
    A: Long,
    LDA: IntByReference,
    IPIV: Long,
    INFO: IntByReference,
  ): Unit

  def dgetri(
    N: IntByReference,
    A: Long,
    LDA: IntByReference,
    IPIV: Long,
    WORK: Long,
    LWORK: IntByReference,
    INFO: IntByReference,
  ): Unit

  def dgesdd(
    JOBZ: String,
    M: IntByReference,
    N: IntByReference,
    A: Long,
    LDA: IntByReference,
    S: Long,
    U: Long,
    LDU: IntByReference,
    VT: Long,
    LDVT: IntByReference,
    WORK: Long,
    LWORK: IntByReference,
    IWORK: Long,
    INFO: IntByReference,
  ): Unit

  def dsyevr(
    jobz: String,
    range: String,
    uplo: String,
    n: IntByReference,
    A: Long,
    ldA: IntByReference,
    vl: DoubleByReference,
    vu: DoubleByReference,
    il: IntByReference,
    iu: IntByReference,
    abstol: DoubleByReference,
    m: IntByReference,
    W: Long,
    Z: Long,
    ldZ: IntByReference,
    ISuppZ: Long,
    Work: Long,
    lWork: IntByReference,
    IWork: Long,
    lIWork: IntByReference,
    info: IntByReference,
  ): Unit

  def ilaver(MAJOR: IntByReference, MINOR: IntByReference, PATCH: IntByReference): Unit

  def ilaenv(
    ispec: IntByReference,
    name: String,
    opts: String,
    n1: IntByReference,
    n2: IntByReference,
    n3: IntByReference,
    n4: IntByReference,
  ): Int

  def dtrtrs(
    UPLO: String,
    TRANS: String,
    DIAG: String,
    N: IntByReference,
    NRHS: IntByReference,
    A: Long,
    LDA: IntByReference,
    B: Long,
    LDB: IntByReference,
    INFO: IntByReference,
  ): Unit

  def dlacpy(
    uplo: String,
    M: IntByReference,
    N: IntByReference,
    A: Long,
    ldA: IntByReference,
    B: Long,
    ldB: IntByReference,
  ): Unit
}
