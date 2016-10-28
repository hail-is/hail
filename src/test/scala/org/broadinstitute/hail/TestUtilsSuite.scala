package org.broadinstitute.hail

import breeze.linalg.{DenseMatrix, DenseVector}
import org.broadinstitute.hail.variant.Variant
import org.testng.annotations.Test

class TestUtilsSuite extends SparkSuite {

  @Test def vdsFromMatrixTest() {
    val G = DenseMatrix((0, 1), (2, -1), (0, 1))
    val vds = TestUtils.vdsFromMatrix(sc)(G)

    val G1 = DenseMatrix.zeros[Int](3, 2)
    vds.rdd.collect().foreach{ case (v, (va, gs)) => gs.zipWithIndex.foreach { case (g, i) => G1(i, v.start - 1) = g.gt.getOrElse(-1) } }

    assert(vds.sampleIds == IndexedSeq("0", "1", "2"))
    assert(vds.variants.collect().toSet == Set(Variant("1", 1, "A", "C"), Variant("1", 2, "A", "C")))

    assert(G == G1)
  }

  @Test def matrixEqualityTest() {
    val M = DenseMatrix((1d, 0d), (0d, 1d))
    val M1 = DenseMatrix((1d, 0d), (0d, 1.0001d))
    val V = DenseVector(0d, 1d)
    val V1 = DenseVector(0d, 0.5d)

    TestUtils.assertMatrixEqualityDouble(M, DenseMatrix.eye(2))
    TestUtils.assertMatrixEqualityDouble(M, M1, 0.001)
    TestUtils.assertVectorEqualityDouble(V, 2d * V1)

    intercept[Exception](TestUtils.assertVectorEqualityDouble(V, V1))
    intercept[Exception](TestUtils.assertMatrixEqualityDouble(M, M1))
  }
}
