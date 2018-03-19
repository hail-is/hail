package is.hail.methods

import breeze.linalg.DenseMatrix
import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.expr.types._
import is.hail.table.Table
import is.hail.testUtils._
import is.hail.variant.{ReferenceGenome, Locus, MatrixTable}
import org.apache.spark.sql.Row
import org.testng.annotations.Test

object PCASuite {
  def samplePCA(vsm: MatrixTable, k: Int = 10, computeLoadings: Boolean = false,
    asArray: Boolean = false): (IndexedSeq[Double], DenseMatrix[Double], Option[Table]) = {

    val prePCA = vsm.annotateRowsExpr(
      "AC = AGG.map(g => g.GT.nNonRefAlleles()).sum(), nCalled = AGG.filter(g => isDefined(g.GT)).count()")
      .filterRowsExpr("va.AC > 0 && va.AC.toInt64 < 2L * va.nCalled").persist()
    val nVariants = prePCA.countRows()
    val expr = s"let mean = va.AC.toFloat64 / va.nCalled.toFloat64 in if (isDefined(g.GT)) (g.GT.nNonRefAlleles().toFloat64 - mean) / sqrt(mean * (2d - mean) * ${ nVariants }d / 2d) else 0d"
    
    PCA(prePCA, expr, k, computeLoadings, asArray)
  }
}

class PCASuite extends SparkSuite {

  @Test def test() {

    val vds = hc.importVCF("src/test/resources/tiny_m.vcf")
      .filterRowsExpr("va.alleles.length() == 2")
    val (eigenvalues, scores, loadings) = PCASuite.samplePCA(vds, 3, true, true)
    val (eigenvaluesStruct, scoresStruct, loadingsStruct) = PCASuite.samplePCA(vds, 3, true, false)

    // comparing against numbers computed via Python script test/resources/PCA.py

    val arrayT = TArray(TFloat64())

    val pyScores = Map[Annotation, IndexedSeq[Double]](
    0 -> IndexedSeq(-0.55141958610810227, 0.6480766747061064, -0.3559869584014231),
    1 -> IndexedSeq(-0.6916959815105279, -0.7626843185339386, -0.13868806289543628),
    2 -> IndexedSeq(1.487286902938744, -0.08212707761864713, -0.09901636248685303),
    3 -> IndexedSeq(-0.2441713353201146, 0.19673472144647947, 0.5936913837837123))

    (0 until 4).foreach { i => assert(arrayT.valuesSimilar((0 until 3).map(j => scores(i,j)), pyScores(i))) }

    val pyLoadings = Array(
      Row(Locus("20", 10019093), IndexedSeq("A", "G"), -0.2804779961843084, 0.41201694824790014, -0.866933750648181),
      Row(Locus("20", 10026348), IndexedSeq("A", "G"), -0.27956988837183483, -0.8990945092947515, -0.33685269907155196),
      Row(Locus("20", 10026357), IndexedSeq("T", "C"), 0.918244396210614, -0.14788880184962383, -0.36736375857627535))

    val structT = TStruct("_PC1" -> TFloat64(), "_PC2" -> TFloat64(), "_PC3" -> TFloat64())
    val truth = Table.parallelize(hc, pyLoadings,
      TStruct("locus" -> TLocus(ReferenceGenome.defaultReference), "alleles" -> TArray(TString())) ++ structT,
      Array("locus", "alleles"), None)
    truth.typeCheck()

    assert(truth.join(loadings.get, "outer").forall(
      s"""
         |abs(row.loadings[0] - row._PC1) < 0.001 &&
         |abs(row.loadings[1] - row._PC2) < 0.001 &&
         |abs(row.loadings[2] - row._PC3) < 0.001
       """.stripMargin))

    assert(truth.join(loadingsStruct.get, "outer").forall(
      s"""
         |abs(row.PC1 - row._PC1) < 0.001 &&
         |abs(row.PC2 - row._PC2) < 0.001 &&
         |abs(row.PC3 - row._PC3) < 0.001
       """.stripMargin))

    val pyEigen = IndexedSeq(3.0541488634265739, 1.0471401535365061, 0.5082347925607319)

    assert(arrayT.valuesSimilar(eigenvalues, pyEigen), s"$eigenvalues")
  }

  @Test def testExpr() {
    val vds = hc.importVCF("src/test/resources/tiny_m.vcf")
        .filterRowsExpr("va.alleles.length() == 2")
    val (eigenvalues, scores, loadings) =
      PCA(vds, "if (isDefined(g.GT)) g.GT.nNonRefAlleles().toFloat64 else 0.0", 3, true, true)
  }
}
