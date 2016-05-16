package org.broadinstitute.hail.methods

import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver._
import org.testng.annotations.Test
import sys.process._
import scala.language._
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.Properties
import org.broadinstitute.hail.variant._


class ImputeGenderSuite extends SparkSuite {

  def parsePlinkSexCheck(file: String): RDD[(String, (Option[Int], Option[Double]))] = sc.parallelize(readLines(file, sc.hadoopConfiguration)(_.drop(1).map(_.transform { line =>
    val Array(fid, iid, pedsex, snpsex, status, f) = line.value.trim.split("\\s+")
    val sex = snpsex match {
      case "1" => Option(1)
      case "2" => Option(2)
      case _ => None
    }
    val fMod = f match {
      case "nan" => None
      case x:String => Option(x.toDouble)
      case _ => throw new IllegalArgumentException
    }

    (iid, (sex, fMod))
  }
  ).toIndexedSeq))

  @Test def testImputeGenderPlinkV1 = {
    var s = State(sc, sqlContext)

    val vcfFile = "src/test/resources/chrX_1kg_thinned2.nobadalt.vcf"
    val plinkSexCheckOutput = tmpDir.createTempFile(prefix="plinksexCheck")

    s = ImportVCF.run(s, Array(vcfFile))
    s = SplitMulti.run(s, Array.empty[String])
    s = ImputeGender.run(s, Array("-m","0.0"))

    s"/Users/jigold/plink --vcf $vcfFile --const-fid --check-sex --out $plinkSexCheckOutput" !

    val plinkResult = parsePlinkSexCheck(plinkSexCheckOutput + ".sexcheck")

    val (_, imputedSexQuery) = s.vds.querySA("sa.imputesex.imputedSex")
    val (_, fQuery) = s.vds.querySA("sa.imputesex.F")

    val hailResult = sc.parallelize(s.vds.sampleIdsAndAnnotations.map{case (sample, sa) =>
      (sample, (imputedSexQuery(sa).get, fQuery(sa).get))
    })

    val mergeResults = plinkResult.fullOuterJoin(hailResult)

    val result = mergeResults.map{ case (sample, (data1, data2)) =>

      val (plink_sex, plink_f) = data1.map{case (sex, f) => (sex, f)}.get
      val (hail_sex: Option[Int], hail_f: Option[Double]) = data2.map{case (sex, f) => (sex, f)}.get

      val resultF = if (plink_f.isDefined && hail_f.isDefined) math.abs(plink_f.get - hail_f.get) < 1e-3 else plink_f == hail_f
      val resultSex = plink_sex == hail_sex

      if (resultSex && resultF)
        true
      else {
        println(s"$sample plink=${data1.getOrElse("NA")} hail=${data2.getOrElse("NA")} $resultSex $resultF")
        false
      }
    }.fold(true)(_ && _)

    assert(result)
  }

  object Spec extends Properties("ImputeSex") {

    property("hail generates same results as PLINK v1.9") =
      forAll(VariantSampleMatrix.gen[Genotype](sc, Genotype.gen _)) { case (vds: VariantSampleMatrix[Genotype]) =>

        if (vds.nSamples == 0 || vds.nVariants == 0) {
          true
        } else {
          var s = State(sc, sqlContext).copy(vds = vds.copy(rdd = vds.rdd.map { case (v, va, gs) => (v.copy(contig = "X"), va, gs) }))

          val fileRoot = tmpDir.createTempFile(prefix = "plinksexCheck")
          val vcfOutputFile = fileRoot + ".vcf"
          val plinkSexCheckRoot = fileRoot

          s = SplitMulti.run(s, Array[String]())
          s = ImputeGender.run(s, Array("-m", "0.0"))
          s = ExportVCF.run(s, Array("-o", vcfOutputFile))

          s"/Users/jigold/plink --vcf $vcfOutputFile --const-fid --check-sex --out $plinkSexCheckRoot" !

          val plinkResult = parsePlinkSexCheck(plinkSexCheckRoot + ".sexcheck")

          val (_, imputedSexQuery) = s.vds.querySA("sa.imputesex.imputedSex")
          val (_, fQuery) = s.vds.querySA("sa.imputesex.F")

          val hailResult = sc.parallelize(s.vds.sampleIdsAndAnnotations.map { case (sample, sa) =>
            (sample, (imputedSexQuery(sa).get, fQuery(sa).get))
          })

          val mergeResults = plinkResult.fullOuterJoin(hailResult)

          val result = mergeResults.map { case (sample, (data1, data2)) =>

            val (plink_sex, plink_f) = data1.map { case (sex, f) => (sex, f) }.get
            val (hail_sex: Option[Int], hail_f: Option[Double]) = data2.map { case (sex, f) => (sex, f) }.get

            val resultSex = plink_sex == hail_sex
            val resultF = if (plink_f.isDefined && hail_f.isDefined) math.abs(plink_f.get - hail_f.get) < 1e-3 else plink_f == hail_f

            if (resultSex && resultF)
              true
            else {
              println(s"$sample plink=${data1.getOrElse("NA")} hail=${data2.getOrElse("NA")} $resultSex $resultF")
              false
            }
          }.fold(true)(_ && _)

          result
        }
      }
  }

  @Test def testImputeGenderPlinkVersion() {
    Spec.check(size = 100, count = 100, seed = Option(1), random = true)
  }
}
