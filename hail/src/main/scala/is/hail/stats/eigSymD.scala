package is.hail.stats

import breeze.generic.UFunc
import breeze.linalg._
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}
import org.netlib.util.intW

/** Computes all eigenvalues (and optionally right eigenvectors) of the given real symmetric matrix
  * X using dsyevd (Divide and Conquer):
  * `http://www.netlib.org/lapack/explore-html/d2/d8a/group__double_s_yeigen_ga694ddc6e5527b6223748e3462013d867.html`
  *
  * Based on eigSym in breeze.linalg.eig but replaces dsyev with dsyevd for higher performance:
  * http://www.netlib.org/lapack/lawnspdf/lawn183.pdf
  */
object eigSymD extends UFunc {
  case class EigSymD[V, M](eigenvalues: V, eigenvectors: M)
  type DenseEigSymD = EigSymD[DenseVector[Double], DenseMatrix[Double]]

  implicit object eigSymD_DM_Impl extends Impl[DenseMatrix[Double], DenseEigSymD] {
    def apply(X: DenseMatrix[Double]): DenseEigSymD =
      doeigSymD(X, rightEigenvectors = true) match {
        case (ev, Some(rev)) => EigSymD(ev, rev)
        case _ => throw new RuntimeException("Shouldn't be here!")
      }

  }

  object justEigenvalues extends UFunc {
    implicit object eigSymD_DM_Impl extends Impl[DenseMatrix[Double], DenseVector[Double]] {
      def apply(X: DenseMatrix[Double]): DenseVector[Double] =
        doeigSymD(X, rightEigenvectors = false)._1
    }

  }

  def doeigSymD(X: Matrix[Double], rightEigenvectors: Boolean)
    : (DenseVector[Double], Option[DenseMatrix[Double]]) = {
    // assumes X is non-empty and symmetric, caller should check if necessary

    val JOBZ =
      if (rightEigenvectors) "V" else "N" /* eigenvalues N, eigenvalues & eigenvectors "V" */
    val UPLO = "L"
    val N = X.rows
    val A = lowerTriangular(X)
    val LDA = scala.math.max(1, N)
    val W = DenseVector.zeros[Double](N)
    val LWORK =
      if (N <= 1)
        1
      else if (JOBZ == "N")
        2 * N + 1
      else
        1 + 6 * N + 2 * N * N
    val WORK = new Array[Double](LWORK)
    val LIWORK =
      if (N <= 1 || JOBZ == "N")
        1
      else
        3 + 5 * N
    val IWORK = new Array[Int](LIWORK)
    val info = new intW(0)

    lapack.dsyevd(JOBZ, UPLO, N, A.data, LDA, W.data, WORK, LWORK, IWORK, LIWORK, info)
    // A value of info.`val` < 0 would tell us that the i-th argument
    // of the call to dsyevd was erroneous (where i == |info.`val`|).
    assert(info.`val` >= 0)

    if (info.`val` > 0)
      throw new NotConvergedException(NotConvergedException.Iterations)

    (W, if (rightEigenvectors) Some(A) else None)
  }
}

/** Computes all eigenvalues (and optionally right eigenvectors) of the given real symmetric matrix
  * X using dsyevr (Relatively Robust Representations):
  * `http://www.netlib.org/lapack/explore-html/d2/d8a/group__double_s_yeigen_ga2ad9f4a91cddbf67fe41b621bd158f5c.html`
  *
  * Based on eigSym in breeze.linalg.eig but replaces dsyev with dsyevr for higher performance:
  * http://www.netlib.org/lapack/lawnspdf/lawn183.pdf
  */
object eigSymR extends UFunc {
  case class EigSymR[V, M](eigenvalues: V, eigenvectors: M)
  type DenseeigSymR = EigSymR[DenseVector[Double], DenseMatrix[Double]]

  implicit object eigSymR_DM_Impl extends Impl[DenseMatrix[Double], DenseeigSymR] {
    def apply(X: DenseMatrix[Double]): DenseeigSymR =
      doeigSymR(X, rightEigenvectors = true) match {
        case (ev, Some(rev)) => EigSymR(ev, rev)
        case _ => throw new RuntimeException("Shouldn't be here!")
      }

  }

  object justEigenvalues extends UFunc {
    implicit object eigSymR_DM_Impl extends Impl[DenseMatrix[Double], DenseVector[Double]] {
      def apply(X: DenseMatrix[Double]): DenseVector[Double] =
        doeigSymR(X, rightEigenvectors = false)._1
    }

  }

  private def doeigSymR(X: Matrix[Double], rightEigenvectors: Boolean)
    : (DenseVector[Double], Option[DenseMatrix[Double]]) = {
    // assumes X is non-empty and symmetric, caller should check if necessary

    val JOBZ =
      if (rightEigenvectors) "V" else "N" /* eigenvalues N, eigenvalues & eigenvectors "V" */
    val RANGE = "A" // supports eigenvalue range, but not implementing that interface here
    val UPLO = "L"
    val N = X.rows
    val A = X.toDenseMatrix.data
    val LDA = scala.math.max(1, N)
    val ABSTOL = -1d // default tolerance
    val W = DenseVector.zeros[Double](N)
    val M = new intW(0)
    val Z = DenseMatrix.zeros[Double](N, N)
    val LDZ = N
    val ISUPPZ = DenseVector.zeros[Int](2 * N)
    val LWORK = 26 * N
    val WORK = new Array[Double](LWORK)
    val LIWORK = 10 * N
    val IWORK = new Array[Int](LIWORK)
    val info = new intW(0)

    lapack.dsyevr(
      JOBZ,
      RANGE,
      UPLO,
      N,
      A,
      LDA,
      0d,
      0d,
      0,
      0,
      ABSTOL,
      M,
      W.data,
      Z.data,
      LDZ,
      ISUPPZ.data,
      WORK,
      LWORK,
      IWORK,
      LIWORK,
      info,
    )
    // A value of info.`val` < 0 would tell us that the i-th argument
    // of the call to dsyevr was erroneous (where i == |info.`val`|).
    assert(info.`val` >= 0)

    if (info.`val` > 0)
      throw new NotConvergedException(NotConvergedException.Iterations)

    (W, if (rightEigenvectors) Some(Z) else None)
  }
}

object TriSolve {
  /* Solve for x in A * x = b with upper triangular A
   * http://www.netlib.org/lapack/explore-html/da/dba/group__double_o_t_h_e_rcomputational_ga4e87e579d3e1a56b405d572f868cd9a1.html */

  def apply(A: DenseMatrix[Double], b: DenseVector[Double]): DenseVector[Double] = {
    require(A.rows == A.cols)
    require(A.rows == b.length)

    val x = DenseVector(b.toArray)

    val info: Int = {
      val info = new intW(0)
      lapack.dtrtrs(
        "U",
        "N",
        "N",
        A.rows,
        1,
        A.toArray,
        A.rows,
        x.data,
        x.length,
        info,
      ) // x := A \ x
      info.`val`
    }

    if (info > 0)
      throw new MatrixSingularException()
    else if (info < 0)
      throw new IllegalArgumentException()

    x
  }
}
