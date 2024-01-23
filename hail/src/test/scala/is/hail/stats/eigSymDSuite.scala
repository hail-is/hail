package is.hail.stats

import is.hail.{HailSuite, TestUtils}
import is.hail.utils._

import breeze.linalg.{eigSym, svd, DenseMatrix, DenseVector}
import org.apache.commons.math3.random.JDKRandomGenerator
import org.testng.annotations.Test

class eigSymDSuite extends HailSuite {
  @Test def eigSymTest(): Unit = {
    val seed = 0

    val rand = new JDKRandomGenerator()
    rand.setSeed(seed)

    val n = 5
    val m = 10
    val W = DenseMatrix.fill[Double](n, m)(rand.nextGaussian())
    val K = W * W.t
    K.forceSymmetry()

    val svdW = svd(W)
    val svdK = svd(K)
    val eigSymK = eigSym(K)
    val eigSymDK = eigSymD(K)
    val eigSymRK = eigSymR(K)

    // eigSymD = svdW
    for (j <- 0 until n) {
      assert(D_==(svdW.S(j) * svdW.S(j), eigSymDK.eigenvalues(n - j - 1)))
      for (i <- 0 until n)
        assert(D_==(math.abs(svdW.U(i, j)), math.abs(eigSymDK.eigenvectors(i, n - j - 1))))
    }

    // eigSymR = svdK
    for (j <- 0 until n) {
      assert(D_==(svdK.S(j), eigSymDK.eigenvalues(n - j - 1)))
      for (i <- 0 until n)
        assert(D_==(math.abs(svdK.U(i, j)), math.abs(eigSymDK.eigenvectors(i, n - j - 1))))
    }

    // eigSymD = eigSym
    for (j <- 0 until n) {
      assert(D_==(eigSymK.eigenvalues(j), eigSymDK.eigenvalues(j)))
      for (i <- 0 until n)
        assert(D_==(math.abs(eigSymK.eigenvectors(i, j)), math.abs(eigSymDK.eigenvectors(i, j))))
    }

    // small example
    val K2 = DenseMatrix((2.0, 1.0), (1.0, 2.0))
    val c = 1 / math.sqrt(2)

    val eigSymDK2 = eigSymD(K2)
    val eigSymRK2 = eigSymR(K2)
    assert(D_==(eigSymDK2.eigenvalues(0), 1.0))
    assert(D_==(eigSymDK2.eigenvalues(1), 3.0))
    assert(D_==(math.abs(eigSymDK2.eigenvectors(0, 0)), c))
    assert(D_==(math.abs(eigSymDK2.eigenvectors(1, 0)), c))
    assert(D_==(math.abs(eigSymDK2.eigenvectors(0, 1)), c))
    assert(D_==(math.abs(eigSymDK2.eigenvectors(1, 1)), c))

    assert(D_==(eigSymRK2.eigenvalues(0), 1.0))
    assert(D_==(eigSymRK2.eigenvalues(1), 3.0))
    assert(D_==(math.abs(eigSymRK2.eigenvectors(0, 0)), c))
    assert(D_==(math.abs(eigSymRK2.eigenvectors(1, 0)), c))
    assert(D_==(math.abs(eigSymRK2.eigenvectors(0, 1)), c))
    assert(D_==(math.abs(eigSymRK2.eigenvectors(1, 1)), c))
  }

  def symEigSpeedTest(): Unit = {
    val seed = 0

    val rand = new JDKRandomGenerator()
    rand.setSeed(seed)

    def timeSymEig(): Unit = {
      for (n <- 500 to 5500 by 500) {
        val W = DenseMatrix.fill[Double](n, n)(rand.nextGaussian())
        val K = W * W.t
        K.forceSymmetry()

        println(s"$n dim")
        print("svd:     ")
        printTime({ svd(W) })
        print("svdK:    ")
        printTime({ svd(K) })
        print("eigSym:  ")
        printTime({ eigSymD(K) })
        print("eigSymR: ")
        printTime({ eigSymR(K) })
        print("eigSymD: ")
        printTime({ eigSym(K) })
        println()
      }
    }

    timeSymEig()
  }

  @Test def triSolveTest(): Unit = {
    val seed = 0

    val rand = new JDKRandomGenerator()
    rand.setSeed(seed)

    (1 to 5).foreach { n =>
      val A = DenseMatrix.zeros[Double](n, n)
      (0 until n).foreach(i => (i until n).foreach(j => A(i, j) = rand.nextGaussian()))

      val x = DenseVector.fill[Double](n)(rand.nextGaussian())

      TestUtils.assertVectorEqualityDouble(x, TriSolve(A, A * x))
    }
  }
}
