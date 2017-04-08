package is.hail.methods

import is.hail.SparkSuite
import is.hail.expr.TDouble
import is.hail.utils._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class LinearRegressionBurdenSuite extends SparkSuite {

  @Test def testWeightedSum() {
    /*
          123
    Gene1 --
    Gene2  -
    Gene3 ---

    Variant A       B       C       D       E       F       Weight
    1       0.0     1.0     0.0     0.0     0.0     1.0     1.0
    2         .     2.0       .     2.0     0.0     0.0     2.0
    3       0.0       .     1.0     1.0     1.0       .     3.0

    Sample  Cov1    Cov2    Pheno   Gene1   Gene2   Gene3
    A       0.0     -1.0    1.0     0.0     0.0     0.0
    B       2.0     3.0     1.0     5.0     4.0     5.0
    C       1.0     5.0     2.0     0.0     0.0     3.0
    D       -2.0    0.0     2.0     4.0     4.0     7.0
    E       -2.0    -4.0    2.0     0.0     0.0     3.0
    F       4.0     3.0     2.0     1.0     0.0     1.0

    Values from R: fit <- lm(Pheno ~ Gene1 + Cov1 + Cov2, data=df)
    Gene    beta        se       tstat    pval
    Gene1   -0.08164    0.15339  -0.532   0.6478
    Gene2   -0.09900    0.17211  -0.575   0.6233
    Gene3   0.01558     0.18323   0.085   0.940
    */

    val gene1values = IndexedSeq(0.0, 5.0, 0.0, 4.0, 0.0, 1.0, -0.08164, 0.15339, -0.532, 0.6478)
    val gene2values = IndexedSeq(0.0, 4.0, 0.0, 4.0, 0.0, 0.0, -0.09900, 0.17211, -0.575, 0.6233)
    val gene3values = IndexedSeq(0.0, 5.0, 3.0, 7.0, 3.0, 1.0, 0.01558, 0.18323, 0.085, 0.940)

    val vds = hc.importVCF("src/test/resources/regressionLinear.vcf")
      .filterMulti()
      .annotateVariantsIntervals("src/test/resources/regressionLinear.interval_list", "va.gene", all=true)
      .annotateVariantsExpr("va.weight = v.start.toDouble")
      .annotateSamplesTable("src/test/resources/regressionLinear.pheno",
        "Sample",
        root = Some("sa.pheno"),
        config = TextTableConfiguration(types = Map("Pheno" -> TDouble), missing = "0"))
      .annotateSamplesTable("src/test/resources/regressionLinear.cov",
        "Sample",
        root = Some("sa.cov"),
        config = TextTableConfiguration(types = Map("Cov1" -> TDouble, "Cov2" -> TDouble)))

    val kt = vds.linregBurden("gene", "va.gene", "sum()", "va.weight * g.gt", "sa.pheno.Pheno", covSA = Array("sa.cov.Cov1", "sa.cov.Cov2"), dropSamples = false)

    // kt.toDF(hc.sqlContext).show()

    val values = kt.collect().map { r => val s = r.asInstanceOf[Row].toSeq
      s.head.asInstanceOf[String] -> s.tail.asInstanceOf[Seq[Double]] }.toMap

    (0 until 10).foreach(i => assert(D_==(gene1values(i), values("Gene1")(i), tolerance = 1e-3)))
    (0 until 10).foreach(i => assert(D_==(gene2values(i), values("Gene2")(i), tolerance = 1e-3)))
    (0 until 10).foreach(i => assert(D_==(gene3values(i), values("Gene3")(i), tolerance = 1e-3)))
  }

  @Test def testWeightedSumWithImputation() {
    /*
          123
    Gene1 --
    Gene2  -
    Gene3 ---

    Variant A       B       C       D       E       F       Weight
    1       0.0     1.0     0.0     0.0     0.0     1.0     1.0
    2       1.0     2.0     1.0     2.0     0.0     0.0     2.0
    3       0.0     .75     1.0     1.0     1.0     .75     3.0

    Sample  Cov1    Cov2    Pheno   Gene1   Gene2   Gene3
    A       0.0     -1.0    1.0     2.0     2.0     2.0
    B       2.0     3.0     1.0     5.0     4.0     7.25
    C       1.0     5.0     2.0     2.0     2.0     5.0
    D       -2.0    0.0     2.0     4.0     4.0     7.0
    E       -2.0    -4.0    2.0     0.0     0.0     3.0
    F       4.0     3.0     2.0     1.0     0.0     3.25

    Values from R: fit <- lm(Pheno ~ Gene1 + Cov1 + Cov2, data=df)
    Gene    beta        se       tstat    pval
    Gene1   -0.2177     0.1621   -1.343   0.3115
    Gene2   -0.2709     0.1675   -1.617   0.2473
    Gene3   -0.05710    0.21078  -0.271   0.812
    */

    val gene1values = IndexedSeq(2.0, 5.0, 2.0, 4.0, 0.0, 1.0, -0.2177, 0.1621, -1.343, 0.3115)
    val gene2values = IndexedSeq(2.0, 4.0, 2.0, 4.0, 0.0, 0.0, -0.2709, 0.1675, -1.617, 0.2473)
    val gene3values = IndexedSeq(2.0, 7.25, 5.0, 7.0, 3.0, 3.25, -0.05710, 0.21078, -0.271, 0.812)

    val vds = hc.importVCF("src/test/resources/regressionLinear.vcf")
      .filterMulti()
      .annotateVariantsIntervals("src/test/resources/regressionLinear.interval_list", "va.gene", all=true)
      .annotateVariantsExpr("va.weight = v.start")
      .annotateSamplesTable("src/test/resources/regressionLinear.pheno",
        "Sample",
        root = Some("sa.pheno"),
        config = TextTableConfiguration(types = Map("Pheno" -> TDouble), missing = "0"))
      .annotateSamplesTable("src/test/resources/regressionLinear.cov",
        "Sample",
        root = Some("sa.cov"),
        config = TextTableConfiguration(types = Map("Cov1" -> TDouble, "Cov2" -> TDouble)))
      .filterSamplesExpr("isDefined(sa.pheno.Pheno) && isDefined(sa.cov.Cov1) && isDefined(sa.cov.Cov2)")
      .variantQC()

    val kt = vds.linregBurden("Gene", "va.gene", "sum()", "va.weight * orElse(g.gt.toDouble, 2 * va.qc.AF)", "sa.pheno.Pheno", covSA = Array("sa.cov.Cov1", "sa.cov.Cov2"), dropSamples = false)

    // kt.toDF(hc.sqlContext).show()

    val values = kt.collect().map { r => val s = r.asInstanceOf[Row].toSeq
      s.head.asInstanceOf[String] -> s.tail.asInstanceOf[Seq[Double]] }.toMap

    (0 until 10).foreach(i => assert(D_==(gene1values(i), values("Gene1")(i), tolerance = 1e-3)))
    (0 until 10).foreach(i => assert(D_==(gene2values(i), values("Gene2")(i), tolerance = 1e-3)))
    (0 until 10).foreach(i => assert(D_==(gene3values(i), values("Gene3")(i), tolerance = 1e-3)))
    }

  @Test def testMax() {
    /*
          123
    Gene1 --
    Gene2  -
    Gene3 ---

    Variant A       B       C       D       E       F
    1       0.0     1.0     0.0     0.0     0.0     1.0
    2         .     2.0       .     2.0     0.0     0.0
    3       0.0       .     1.0     1.0     1.0       .

    Sample  Cov1    Cov2    Pheno   Gene1   Gene2   Gene3
    A       0.0     -1.0    1.0     0.0       .     0.0
    B       2.0     3.0     1.0     2.0     2.0     2.0
    C       1.0     5.0     2.0     0.0       .     1.0
    D       -2.0    0.0     2.0     2.0     2.0     2.0
    E       -2.0    -4.0    2.0     0.0     0.0     1.0
    F       4.0     3.0     2.0     1.0     0.0     1.0

    Values from R: fit <- lm(Pheno ~ Gene1 + Cov1 + Cov2, data=df)
    Gene    beta        se       tstat    pval
    Gene1   -0.08164    0.15339  -0.532   0.6478
    Gene2   -0.09900    0.17211  -0.575   0.6233
    Gene3   0.01558     0.18323   0.085   0.940
    */

    val gene1values = IndexedSeq(0.0, 2.0, 0.0, 2.0, 0.0, 1.0, -0.08368, 0.36841, -0.227, 0.8414)
    val gene2values = IndexedSeq(null, 2.0, null, 2.0, 0.0, 0.0, -0.5418, 0.3351, -1.617, 0.2473)
    val gene3values = IndexedSeq(0.0, 2.0, 1.0, 2.0, 1.0, 1.0, 0.07474, 0.51528, 0.145, 0.898)

    val vds = hc.importVCF("src/test/resources/regressionLinear.vcf")
      .filterMulti()
      .annotateVariantsIntervals("src/test/resources/regressionLinear.interval_list", "va.gene", all=true)
      .annotateSamplesTable("src/test/resources/regressionLinear.pheno",
        "Sample",
        root = Some("sa.pheno"),
        config = TextTableConfiguration(types = Map("Pheno" -> TDouble), missing = "0"))
      .annotateSamplesTable("src/test/resources/regressionLinear.cov",
        "Sample",
        root = Some("sa.cov"),
        config = TextTableConfiguration(types = Map("Cov1" -> TDouble, "Cov2" -> TDouble)))

    val kt = vds.linregBurden("gene", "va.gene", "max()", "g.gt.toDouble", "sa.pheno.Pheno", covSA = Array("sa.cov.Cov1", "sa.cov.Cov2"), dropSamples = false)

    // kt.toDF(hc.sqlContext).show()

    val values = kt.collect().map { r => val s = r.asInstanceOf[Row].toSeq
      s.head.asInstanceOf[String] -> s.tail }.toMap

    (0 until 10).foreach(i => assert(D_==(gene1values(i), values("Gene1")(i).asInstanceOf[Double], tolerance = 1e-3)))
    (0 until 10).foreach(i => assert((gene2values(i) == null && values("Gene2")(i) == null) || D_==(gene2values(i).asInstanceOf[Double], values("Gene2")(i).asInstanceOf[Double], tolerance = 1e-3)))
    (0 until 10).foreach(i => assert(D_==(gene3values(i), values("Gene3")(i).asInstanceOf[Double], tolerance = 1e-3)))
  }
}
