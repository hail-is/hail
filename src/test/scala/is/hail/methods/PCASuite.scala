package is.hail.methods

import breeze.linalg.DenseMatrix
import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.expr.{TArray, TFloat64, TStruct}
import is.hail.table.Table
import is.hail.variant.{Variant, MatrixTable}
import org.testng.annotations.Test

object PCASuite {
  def samplePCA(vsm: MatrixTable, k: Int = 10, computeLoadings: Boolean = false,
    asArray: Boolean = false): (IndexedSeq[Double], DenseMatrix[Double], Option[Table]) = {
    
    val prePCA = vsm.annotateVariantsExpr("va.AC = gs.map(g => g.GT.gt).sum(), va.nCalled = gs.filter(g => isDefined(g.GT)).count()")
      .filterVariantsExpr("va.AC > 0 && va.AC < 2 * va.nCalled").persist()
    val nVariants = prePCA.countVariants()
    val expr = s"let mean = va.AC / va.nCalled in if (isDefined(g.GT)) (g.GT.gt - mean) / sqrt(mean * (2 - mean) * $nVariants / 2) else 0"
    
    PCA(prePCA, expr, k, computeLoadings, asArray)
  }
}

class PCASuite extends SparkSuite {

  @Test def test() {

    val vds = hc.importVCF("src/test/resources/tiny_m.vcf")
        .filterVariantsExpr("v.isBiallelic")
    val (eigenvalues, scores, loadings) = PCASuite.samplePCA(vds, 3, true, true)
    val (eigenvaluesStruct, scoresStruct, loadingsStruct) = PCASuite.samplePCA(vds, 3, true, false)

    // comparing against numbers computed via Python script test/resources/PCA.py

    val arrayT = TArray(TFloat64())
    val structT = TStruct("PC1" -> TFloat64(), "PC2" -> TFloat64(), "PC3" -> TFloat64())

    val pyScores = Map[Annotation, IndexedSeq[Double]](
    0 -> IndexedSeq(-0.55141958610810227, 0.6480766747061064, -0.3559869584014231),
    1 -> IndexedSeq(-0.6916959815105279, -0.7626843185339386, -0.13868806289543628),
    2 -> IndexedSeq(1.487286902938744, -0.08212707761864713, -0.09901636248685303),
    3 -> IndexedSeq(-0.2441713353201146, 0.19673472144647947, 0.5936913837837123))

    (0 until 4).foreach { i => assert(arrayT.valuesSimilar((0 until 3).map(j => scores(i,j)), pyScores(i))) }

    val pyLoadings = Map(Variant("20", 10019093, "A", "G") -> IndexedSeq(-0.2804779961843084, 0.41201694824790014, -0.866933750648181),
      Variant("20", 10026348, "A", "G") -> IndexedSeq(-0.27956988837183483, -0.8990945092947515, -0.33685269907155196),
      Variant("20", 10026357, "T", "C") -> IndexedSeq(0.918244396210614, -0.14788880184962383, -0.36736375857627535))
    val pyLoadingsStruct = pyLoadings.mapValues(Annotation.fromSeq)
    val pyLoadingsBc = sc.broadcast(pyLoadings)
    val pyLoadingsStructBc = sc.broadcast(pyLoadingsStruct)
    val arrayTBc = sc.broadcast(arrayT)
    val structTBc = sc.broadcast(structT)

    assert(loadings.get.rdd.map { r => arrayTBc.value.valuesSimilar(r(1), pyLoadingsBc.value(r(0).asInstanceOf[Variant])) }.collect().forall(b => b))
    assert(loadingsStruct.get.rdd.map { r => structTBc.value.valuesSimilar(r(1), pyLoadingsStructBc.value(r(0).asInstanceOf[Variant])) }.collect().forall(b => b))

    val pyEigen = IndexedSeq(3.0541488634265739, 1.0471401535365061, 0.5082347925607319)

    assert(arrayT.valuesSimilar(eigenvalues, pyEigen), s"$eigenvalues")
  }

  @Test def testExpr() {
    val vds = hc.importVCF("src/test/resources/tiny_m.vcf")
        .filterVariantsExpr("v.isBiallelic")
    val (eigenvalues, scores, loadings) = PCA(vds, "if (isDefined(g.GT)) g.GT.gt else 0", 3, true, true)
  }
}
