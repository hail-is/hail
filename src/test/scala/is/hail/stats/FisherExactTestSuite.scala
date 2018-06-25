package is.hail.stats

import is.hail.SparkSuite
import is.hail.check.Gen._
import is.hail.check.Prop._
import is.hail.check.Properties
import is.hail.utils._
import is.hail.testUtils._
import is.hail.variant.{MatrixTable, _}
import org.testng.annotations.Test

import scala.language.postfixOps
import scala.sys.process._

class FisherExactTestSuite extends SparkSuite {

  @Test def testPvalue() {
    val N = 200
    val K = 100
    val k = 10
    val n = 15
    val a = 5
    val b = 10
    val c = 95
    val d = 90

    val result = fisherExactTest(a, b, c, d)

    assert(math.abs(result(0) - 0.2828) < 1e-4)
    assert(math.abs(result(1) - 0.4754059) < 1e-4)
    assert(math.abs(result(2) - 0.122593) < 1e-4)
    assert(math.abs(result(3) - 1.597972) < 1e-4)
  }

  object Spec extends Properties("FisherExactTest") {
    val twoBytwoMatrix = for (n: Int <- choose(10, 500); k: Int <- choose(1, n - 1); x: Int <- choose(1, n - 1)) yield (k, n - k, x, n - x)

    property("import generates same output as export") =
      forAll(twoBytwoMatrix) { case (a, b, c, d) =>

        val rResultTwoSided = s"Rscript src/test/resources/fisherExactTest.r two.sided $a $b $c $d" !!

        val rResultLess = s"Rscript src/test/resources/fisherExactTest.r less $a $b $c $d" !!

        val rResultGreater = s"Rscript src/test/resources/fisherExactTest.r greater $a $b $c $d" !!


        val rTwoSided = rResultTwoSided.split(" ").take(4)
          .map { s => if (s == "Inf") Double.PositiveInfinity else if (s == "NaN") Double.NaN else s.toDouble }
        val rLess = rResultLess.split(" ").take(4)
          .map { s => if (s == "Inf") Double.PositiveInfinity else if (s == "NaN") Double.NaN else s.toDouble }
        val rGreater = rResultGreater.split(" ").take(4)
          .map { s => if (s == "Inf") Double.PositiveInfinity else if (s == "NaN") Double.NaN else s.toDouble }

        val hailTwoSided = fisherExactTest(a, b, c, d, alternative = "two.sided")
        val hailLess = fisherExactTest(a, b, c, d, alternative = "less")
        val hailGreater = fisherExactTest(a, b, c, d, alternative = "greater")

        val hailResults = Array(hailTwoSided, hailLess, hailGreater)
        val rResults = Array(rTwoSided, rLess, rGreater)

        hailResults.zip(rResults).forall { case (h, r) =>
          val res = D0_==(h(0), r(0)) &&
            D0_==(h(1), h(1)) &&
            D0_==(h(2), r(2)) &&
            D0_==(h(3), r(3))
          if (!res) {
            println(h(0), r(0), D0_==(h(0), r(0)))
            println(h(1), r(1), D0_==(h(1), h(1)))
            println(h(2), r(2), D0_==(h(2), r(2)))
            println(h(3), r(3), D0_==(h(3), r(3)))
          }
          res
        }
      }

    property("expr gives same result as class") =
      forAll(MatrixTable.gen(hc, VSMSubgen.random)) { (vds: MatrixTable) =>
        val sampleIds = vds.stringSampleIds
        val phenotypes = sampleIds.zipWithIndex.map { case (sample, i) =>
          if (i % 3 == 0)
            (sample, "ADHD")
          else if (i % 3 == 1)
            (sample, "Control")
          else
            (sample, "NA")
        }

        val phenotypeFile = tmpDir.createTempFile("phenotypeAnnotation", ".txt")
        hadoopConf.writeTextFile(phenotypeFile) { w =>
          w.write("Sample\tPheno1\n")
          phenotypes.foreach { case (sample, p) => w.write(s"$sample\t$p\n") }
        }

        val vds2 = vds.annotateColsTable(hc.importTable(phenotypeFile).keyBy("Sample"), root = "pheno")
          .annotateColsExpr("pheno" -> "sa.pheno.Pheno1")
          .annotateRowsExpr("macCase" ->
            """AGG.filter(g => sa.pheno == "ADHD" && g.GT.isHet()).count() +
              |2L * AGG.filter(g => sa.pheno == "ADHD" && g.GT.isHomVar()).count()""".stripMargin)
          .annotateRowsExpr("majCase" ->
            """AGG.filter(g => sa.pheno == "ADHD" && g.GT.isHet()).count() +
              |2L * AGG.filter(g => sa.pheno == "ADHD" && g.GT.isHomRef()).count()""".stripMargin)
          .annotateRowsExpr("macControl" ->
            """AGG.filter(g => sa.pheno == "Control" && g.GT.isHet()).count() +
              |2L * AGG.filter(g => sa.pheno == "ADHD" && g.GT.isHomVar()).count()""".stripMargin)
          .annotateRowsExpr("majControl" ->
            """AGG.filter(g => sa.pheno == "Control" && g.GT.isHet()).count() +
              |2L * AGG.filter(g => sa.pheno == "ADHD" && g.GT.isHomRef()).count()""".stripMargin)
          .annotateRowsExpr("fet" ->
            """fet(va.macCase.toInt32(), va.majCase.toInt32(), va.macControl.toInt32(), va.majControl.toInt32())""")


        val (_, q1) = vds2.queryVA("va.macCase")
        val (_, q2) = vds2.queryVA("va.majCase")
        val (_, q3) = vds2.queryVA("va.macControl")
        val (_, q4) = vds2.queryVA("va.majControl")
        val (_, q5) = vds2.queryVA("va.fet.p_value")
        val (_, q6) = vds2.queryVA("va.fet.odds_ratio")
        val (_, q7) = vds2.queryVA("va.fet.ci_95_lower")
        val (_, q8) = vds2.queryVA("va.fet.ci_95_upper")

        vds2.variantsAndAnnotations.forall { case (v, va) =>
          val result = fisherExactTest(
            q1(va).asInstanceOf[Long].toInt,
            q2(va).asInstanceOf[Long].toInt,
            q3(va).asInstanceOf[Long].toInt,
            q4(va).asInstanceOf[Long].toInt)

          val annotationResult = Array(
            q5(va).asInstanceOf[Double],
            q6(va).asInstanceOf[Double],
            q7(va).asInstanceOf[Double],
            q8(va).asInstanceOf[Double])

          result.zip(annotationResult).forall{ case (a, b) => D0_==(a, b) }
        }
      }
  }

  @Test def testFisherExactTest() {
    Spec.check()
  }
}
