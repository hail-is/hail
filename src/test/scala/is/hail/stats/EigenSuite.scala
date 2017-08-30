package is.hail.stats

import breeze.linalg._
import is.hail.{SparkSuite, TestUtils}
import is.hail.annotations.Annotation
import is.hail.expr.TString
import org.testng.annotations.Test

class EigenSuite extends SparkSuite {
  def assertEigenEqual(e1: Eigen, e2: Eigen) {
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

  @Test def testFunctions() {
    val evals = DenseVector(0.0, 1.0, 2.0, 97.0)
    val W = diag(evals)
    val E = eigSymD(W)
    val samplesIds: Array[Annotation] = Array.tabulate(4)(_.toString)
    
    val eig = Eigen(TString, samplesIds, E.eigenvectors, E.eigenvalues)
    val eig1 = Eigen(TString, samplesIds, DenseMatrix(0.0, 0.0, 0.0, 1.0), DenseVector(97.0))
    val eig2 = Eigen(TString, Array("2".asInstanceOf[Annotation]), DenseMatrix((0.0, 0.0, 1.0, 0.0)), evals)

    TestUtils.assertMatrixEqualityDouble(eig.evects, DenseMatrix.eye[Double](4))
    TestUtils.assertVectorEqualityDouble(eig.evals, evals)
    
    assertEigenEqual(eig.takeTop(1), eig1)
    assertEigenEqual(eig.filterRows(TString, _.asInstanceOf[String] == "2"), eig2)
    assert(eig.dropThreshold(2.0).evals == DenseVector(97.0))
    assert(eig.dropProportion(0.02).evals == DenseVector(2.0, 97.0))
    
    assertEigenEqual(eig, eig.distribute(sc).localize())

    val file = tmpDir.createTempFile("test", extension = ".eig")
    eig.write(hc, file)
    assertEigenEqual(eig, Eigen.read(hc, file))
  }

  // testing read/write on eigen from kinship matrix and LD matrix
  @Test def readWriteIdentity() {
    val fileK = tmpDir.createTempFile("testK", extension = ".eig")
    val fileL = tmpDir.createTempFile("testL", extension = ".eig")
    val fileDistK = tmpDir.createTempFile("testDistK", extension = ".eigd")
    val fileDistL = tmpDir.createTempFile("testDistL", extension = ".eigd")

    val vds = hc.importVCF("src/test/resources/sample.vcf")
    
    val kin = vds.rrm()
    val eigK = kin.eigen()
    eigK.write(hc, fileK)
    assertEigenEqual(eigK, Eigen.read(hc, fileK))

    val ld = vds.ldMatrix()
    val eigL = ld.eigen()
    eigL.write(hc, fileL)
    assertEigenEqual(eigL, Eigen.read(hc, fileL))
    
    val eigDistK = eigK.distribute(hc.sc)
    eigDistK.write(fileDistK)
    assertEigenEqual(eigDistK.localize(), EigenDistributed.read(hc, fileDistK).localize())

    val eigDistL = eigL.distribute(hc.sc)
    eigDistL.write(fileDistL)
    assertEigenEqual(eigDistL.localize(), EigenDistributed.read(hc, fileDistL).localize())
  }
}