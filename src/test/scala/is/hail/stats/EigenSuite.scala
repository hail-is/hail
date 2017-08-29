package is.hail.stats

import breeze.linalg._
import breeze.stats.mean
import is.hail.{SparkSuite, TestUtils, stats}
import is.hail.annotations.Annotation
import is.hail.expr.TString
import org.apache.commons.math3.random.JDKRandomGenerator
import org.testng.annotations.Test

class EigenSuite extends SparkSuite {
  def assertEqual(e1: Eigen, e2: Eigen) {
    assert(e1.rowSignature == e2.rowSignature)
    assert(e1.rowIds sameElements e2.rowIds)
    assert(e1.evects == e2.evects)
    assert(e1.evals == e2.evals)
  }
  
  def assertEigenEqualityUpToSign(e1: Eigen, e2: Eigen, r: Range, tolerance: Double = 1e-6) {
    assert(e1.rowSignature == e2.rowSignature)
    assert(e1.rowIds sameElements e2.rowIds)
    assert(math.abs(max(e1.evals - e2.evals)) < 1e-6)
    r.foreach(j => TestUtils.assertVectorEqualityUpToSignDouble(e1.evects(::, j), e2.evects(::, j), tolerance))
  }

  def makeEigen(n: Int, m: Int, seed: Int = 0): Eigen = {
    val seed = 0
    val rand = new JDKRandomGenerator()
    rand.setSeed(seed)

    val samplesIds: Array[Annotation] = Array.tabulate(n)(_.toString)
    val W = DenseMatrix.fill[Double](n, m)(rand.nextGaussian())
    val eig = eigSymD(W)

    Eigen(TString, samplesIds, eig.eigenvectors, eig.eigenvalues)
  }

  @Test def localDistributeIdentity() {
    val eig = makeEigen(3, 5)
    
    assertEqual(eig, eig.distribute(sc).localize())
  }
  
  @Test def readWriteIdentity() {
    // testing small example
    val file = tmpDir.createTempFile("test", extension = ".eig")

    val eig = makeEigen(3, 5)
    
    eig.write(hc, file)
    assertEqual(eig, Eigen.read(hc, file))
    
    // testing eigen from kinship matrix and LD matrix
    val fileK = tmpDir.createTempFile("testK", extension = ".eig")
    val fileL = tmpDir.createTempFile("testL", extension = ".eig")
    val fileDistK = tmpDir.createTempFile("testDistK", extension = ".eigd")
    val fileDistL = tmpDir.createTempFile("testDistL", extension = ".eigd")

    val vds = hc.importVCF("src/test/resources/sample.vcf")
    
    val eigK = vds.rrm().eigen()
    eigK.write(hc, fileK)
    assertEqual(eigK, Eigen.read(hc, fileK))

    val eigL = vds.ldMatrix().eigen()
    eigL.write(hc, fileL)
    assertEqual(eigL, Eigen.read(hc, fileL))
    
    val eigDistK = eigK.distribute(hc.sc)
    eigDistK.write(fileDistK)
    assertEqual(eigDistK.localize(), EigenDistributed.read(hc, fileDistK).localize())

    val eigDistL = eigL.distribute(hc.sc)
    eigDistL.write(fileDistL)
    assertEqual(eigDistL.localize(), EigenDistributed.read(hc, fileDistL).localize())
  }
}