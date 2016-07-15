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


class ImputeSexSuite extends SparkSuite {

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

  object Spec extends Properties("ImputeSex") {

    property("hail generates same results as PLINK v1.9") =
      forAll(VariantSampleMatrix.gen[Genotype](sc, VSMSubgen.random)) { case (vds: VariantSampleMatrix[Genotype]) =>

        var s = State(sc, sqlContext).copy(vds = vds.copy(rdd = vds.rdd.map { case (v, va, gs) => (v.copy(contig = "X"), va, gs) }))

        s = SplitMulti.run(s, Array.empty[String])
        s = VariantQC.run(s, Array[String]())
        s = FilterVariantsExpr.run(s, Array("--keep", "-c", "va.qc.AC > 0"))

        if (s.vds.nSamples < 5 || s.vds.nVariants < 5) {
          true
        } else {
          val fileRoot = tmpDir.createTempFile(prefix = "plinksexCheck")
          val vcfOutputFile = fileRoot + ".vcf"
          val plinkSexCheckRoot = fileRoot

          s = ImputeSex.run(s, Array("-m", "0.0", "--include-par"))
          s = ExportVCF.run(s, Array("-o", vcfOutputFile))

          s"plink --vcf $vcfOutputFile --const-fid --check-sex --silent --out $plinkSexCheckRoot" !

          val plinkResult = parsePlinkSexCheck(plinkSexCheckRoot + ".sexcheck")

          val (_, imputedSexQuery) = s.vds.querySA("if (sa.imputesex.isFemale) 2 else 1")
          val (_, fQuery) = s.vds.querySA("sa.imputesex.Fstat")

          val hailResult = sc.parallelize(s.vds.sampleIdsAndAnnotations.map { case (sample, sa) =>
            (sample, (imputedSexQuery(sa), fQuery(sa)))
          })

          val mergeResults = plinkResult.fullOuterJoin(hailResult)

          val result = mergeResults.map { case (sample, (data1, data2)) =>

            val (plink_sex, plink_f) = data1.map { case (sex, f) => (sex, f) }.get
            val (hail_sex, hail_f) = data2.map { case (sex, f) => (sex.map(_.asInstanceOf[Int]),
              f.map(_.asInstanceOf[Double])) }.get

            val resultSex = plink_sex == hail_sex
            val resultF = if (plink_f.isDefined && hail_f.isDefined) math.abs(plink_f.get - hail_f.get) < 1e-3 else plink_f == hail_f

            if (resultSex && resultF)
              true
            else {
              println(s"$sample plink=${data1.getOrElse("NA")} hail=${data2.getOrElse("NA")} $resultSex $resultF")
              false
            }
          }.fold(true)(_ && _)

          val countAnnotated = AnnotateVariantsExpr.run(s,
            Array("-c", "va.maf = gs.stats(g.nNonRefAlleles).sum / (gs.count(true) * 2)"))
          val sexcheck2 = ImputeSex.run(countAnnotated, Array("--pop-freq", "va.maf"))

          result && sexcheck2.vds.sampleAnnotations == s.vds.sampleAnnotations
        }
      }
  }

  @Test def testImputeSexPlinkVersion() {
    Spec.check()
  }
}
