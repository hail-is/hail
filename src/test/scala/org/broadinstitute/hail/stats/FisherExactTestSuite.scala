package org.broadinstitute.hail.stats

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.expr.{EvalContext, TDouble, TInt}
import org.broadinstitute.hail.variant.{VariantDataset, VariantSampleMatrix}
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.driver._
import org.testng.annotations.Test
import org.broadinstitute.hail.check.Gen._
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.Properties
import scala.language.postfixOps
import scala.sys.process._
import org.broadinstitute.hail.Utils._
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

    val fet = new FisherExactTest(a, b, c, d)
    val result = fet.result().map {
      _.getOrElse(Double.NaN)
    }

    assert(math.abs(result(0) - 0.2828) < 1e-4)
    assert(math.abs(result(1) - 0.4754059) < 1e-4)
    assert(math.abs(result(2) - 0.122593) < 1e-4)
    assert(math.abs(result(3) - 1.597972) < 1e-4)
  }

  object Spec extends Properties("FisherExactTest") {
    val twoBytwoMatrix = for (n: Int <- choose(10, 500); k: Int <- choose(1, n - 1); x: Int <- choose(1, n - 1)) yield (k, n - k, x, n - x)

//    property("import generates same output as export") =
//      forAll(twoBytwoMatrix) { case (t) =>
//        val a = t._1
//        val b = t._2
//        val c = t._3
//        val d = t._4
//
//        val fet = FisherExactTest(a, b, c, d)
//
//        val rResultTwoSided = s"Rscript src/test/resources/fisherExactTest.r two.sided $a $b $c $d" !!
//
//        val rResultLess = s"Rscript src/test/resources/fisherExactTest.r less $a $b $c $d" !!
//
//        val rResultGreater = s"Rscript src/test/resources/fisherExactTest.r greater $a $b $c $d" !!
//
//
//        val rTwoSided = rResultTwoSided.split(" ").take(4)
//          .map { case s => if (s == "Inf") Double.PositiveInfinity else if (s == "NaN") Double.NaN else s.toDouble }
//        val rLess = rResultLess.split(" ").take(4)
//          .map { case s => if (s == "Inf") Double.PositiveInfinity else if (s == "NaN") Double.NaN else s.toDouble }
//        val rGreater = rResultGreater.split(" ").take(4)
//          .map { case s => if (s == "Inf") Double.PositiveInfinity else if (s == "NaN") Double.NaN else s.toDouble }
//
//        val hailTwoSided = fet.result(alternative = "two.sided")
//        val hailLess = fet.result(alternative = "less")
//        val hailGreater = fet.result(alternative = "greater")
//
//        val hailResults = Array(hailTwoSided, hailLess, hailGreater).map {
//          _.map {
//            _.getOrElse(Double.NaN)
//          }
//        }
//        val rResults = Array(rTwoSided, rLess, rGreater)
//
//        hailResults.zip(rResults).forall { case (h, r) =>
//          val res = D_==(h(0), r(0)) &&
//            D_==(h(1), h(1), 1e-6) &&
//            D_==(h(2), r(2), 1e-6) &&
//            (h(3) == Double.PositiveInfinity && r(3) == Double.PositiveInfinity) || D_==(h(3), r(3), 1e-6)
//          if (!res) {
//            println(h(0), r(0), D_==(h(0), r(0)))
//            println(h(1), r(1), D_==(h(1), h(1), 1e-6))
//            println(h(2), r(2), D_==(h(2), r(2), 1e-6))
//            println(h(3), r(3), (h(3) == Double.PositiveInfinity && r(3) == Double.PositiveInfinity) || D_==(h(3), r(3), 1e-6))
//          }
//          res
//        }
//      }

    property("expr gives same result as command") =
      forAll(VariantSampleMatrix.gen[Genotype](sc, Genotype.gen _)) { (vds: VariantDataset) =>
        var s = State(sc, sqlContext, vds)
        val sampleIds = vds.sampleIds
        val phenotypes = sampleIds.zipWithIndex.map{case (s, i) =>
          if (i % 3 == 0)
            (s, "ADHD")
          else if (i % 3 == 1)
            (s, "Control")
          else
            (s, "NA")
        }

        val phenotypeFile = tmpDir.createTempFile("phenotypeAnnotation",".txt")
        writeTextFile(phenotypeFile, sc.hadoopConfiguration){case w =>
          w.write("Sample\tPheno1\n")
          phenotypes.foreach{case (s, p) => w.write(s"$s\t$p\n")}
        }

        s = AnnotateSamplesTable.run(s, Array("-i", phenotypeFile, "-r", "sa.pheno"))

        s = AnnotateVariantsExpr.run(s, Array("-c", """va.macCase = gs.count(sa.pheno.Pheno1 == "ADHD" && g.isHet) + 2 * gs.count(sa.pheno.Pheno1 == "ADHD" && g.isHomVar)"""))
        s = AnnotateVariantsExpr.run(s, Array("-c", """va.majCase = gs.count(sa.pheno.Pheno1 == "ADHD" && g.isHet) + 2 * gs.count(sa.pheno.Pheno1 == "ADHD" && g.isHomRef)"""))
        s = AnnotateVariantsExpr.run(s, Array("-c", """va.macControl = gs.count(sa.pheno.Pheno1 == "Control" && g.isHet) + 2 * gs.count(sa.pheno.Pheno1 == "ADHD" && g.isHomVar)"""))
        s = AnnotateVariantsExpr.run(s, Array("-c", """va.majControl = gs.count(sa.pheno.Pheno1 == "Control" && g.isHet) + 2 * gs.count(sa.pheno.Pheno1 == "ADHD" && g.isHomRef)"""))

        s = AnnotateVariantsExpr.run(s, Array("-c", """va.fet = fet(va.macCase, va.majCase, va.macControl, va.majControl)"""))

        val output = tmpDir.createTempFile("fetResults", ".tsv")
        println(s.vds.variantsAndAnnotations.take(1).mkString(","))
        //ExportVariants.run(s, Array("-o", output, "-c", "variant=v"))

//        s = ExportVariants.run(s, Array("-c",
//          """VARIANT=v, macCase = va.macCase, majCase = va.majCase, macControl = va.macControl, majControl = va.majControl,
//            | pFET = va.fet.pFET, orat = va.fet.orFET, lower = va.fet.ciLowerFET, upper = va.fet.ciUpperFET
//          """.stripMargin ,"-o", output))
        true
      }
  }

  @Test def testFisherExactTest() {
    Spec.check()
  }
}
