package is.hail.methods

import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.expr.{TArray, TFloat64, TStruct}
import is.hail.variant.Variant
import org.testng.annotations.Test

class PCASuite extends SparkSuite {

  @Test def test() {

    val vds = hc.importVCF("src/test/resources/tiny_m.vcf").filterMulti()
    val (scores, loadings, eigenvalues) = SamplePCA(vds, 3, true, true, true)
    val (scoresStruct, loadingsStruct, eigenvaluesStruct) = SamplePCA(vds, 3, true, true, false)

    // comparing against numbers computed via Python script test/resources/PCA.py

    val arrayT = TArray(TFloat64())
    val structT = TStruct("PC1" -> TFloat64(), "PC2" -> TFloat64(), "PC3" -> TFloat64())

    val pyScores = Map[Annotation, IndexedSeq[Double]](
      "C1046::HG02024" -> IndexedSeq(-0.55141958610810227, 0.6480766747061064, -0.3559869584014231),
      "C1046::HG02025" -> IndexedSeq(-0.6916959815105279, -0.7626843185339386, -0.13868806289543628),
      "C1046::HG02026" -> IndexedSeq(1.487286902938744, -0.08212707761864713, -0.09901636248685303),
      "C1047::HG00731" -> IndexedSeq(-0.2441713353201146, 0.19673472144647947, 0.5936913837837123))
    val pyScoresStruct = pyScores.mapValues(Annotation.fromSeq)

    scores.foreach { case (id, score) => assert(arrayT.valuesSimilar(score, pyScores(id))) }
    scoresStruct.foreach { case (id, score) => assert(structT.valuesSimilar(score, pyScoresStruct(id))) }

    val pyLoadings = Map(Variant("20", 10019093, "A", "G") -> IndexedSeq(-0.2804779961843084, 0.41201694824790014, -0.866933750648181),
      Variant("20", 10026348, "A", "G") -> IndexedSeq(-0.27956988837183483, -0.8990945092947515, -0.33685269907155196),
      Variant("20", 10026357, "T", "C") -> IndexedSeq(0.918244396210614, -0.14788880184962383, -0.36736375857627535))
    val pyLoadingsStruct = pyLoadings.mapValues(Annotation.fromSeq)

    loadings.get.collect().foreach { case (v, l) => assert(arrayT.valuesSimilar(l, pyLoadings(v))) }
    loadingsStruct.get.collect().foreach { case (v, l) => assert(structT.valuesSimilar(l, pyLoadingsStruct(v))) }

    val pyEigen = IndexedSeq(3.0541488634265739, 1.0471401535365061, 0.5082347925607319)
    val pyEigenStruct = Annotation.fromSeq(pyEigen)

    assert(arrayT.valuesSimilar(eigenvalues.get, pyEigen))
    assert(structT.valuesSimilar(eigenvaluesStruct.get, pyEigenStruct))
  }
}
