package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.Properties
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.variant._
import org.testng.annotations.Test

import scala.language._
import scala.sys.process._


class ImputeSexSuite extends SparkSuite {

  def parsePlinkSexCheck(file: String): Map[String, (Option[Int], Option[Double])] =
    hadoopConf.readLines(file)(_.drop(1).map(_.map { line =>
      val Array(fid, iid, pedsex, snpsex, status, f) = line.trim.split("\\s+")
      val sex = snpsex match {
        case "1" => Option(1)
        case "2" => Option(2)
        case _ => None
      }
      val fMod = f match {
        case "nan" => None
        case x: String => Option(x.toDouble)
        case _ => throw new IllegalArgumentException
      }

      (iid, (sex, fMod))
    }.value
    ).toMap)

  object Spec extends Properties("ImputeSex") {

    val plinkSafeBiallelicVDS = VariantSampleMatrix.gen(sc, VSMSubgen.plinkSafeBiallelic)
      .resize(1000)
      .map(vds => vds.filterVariants { case (v, va, gs) => v.isAutosomalOrPseudoAutosomal && v.contig.toUpperCase != "X" && v.contig.toUpperCase != "Y" })
      .filter(vds => vds.nVariants > 2 && vds.nSamples >= 2)

    property("hail generates same results as PLINK v1.9") =
      forAll(plinkSafeBiallelicVDS) { case (vds: VariantSampleMatrix[Genotype]) =>

        var s = State(sc, sqlContext).copy(vds = vds.copy(rdd =
          vds.rdd.map { case (v, (va, gs)) => (v.copy(contig = "X"), (va, gs)) }
            .toOrderedRDD))

        s = VariantQC.run(s)
        s = FilterVariantsExpr.run(s, Array("--keep", "-c", "va.qc.AC > 1 && va.qc.AF >= 1e-8 && va.qc.nCalled * 2 - va.qc.AC > 1 && va.qc.AF <= 1 - 1e-8"))

        if (s.vds.nSamples < 5 || s.vds.nVariants < 5) {
          true
        } else {
          val localRoot = tmpDir.createLocalTempFile("plinksexCheck")
          val localVCFFile = localRoot + ".vcf"
          val localSexcheckFile = localRoot + ".sexcheck"

          val root = tmpDir.createTempFile("plinksexCheck")
          val vcfFile = root + ".vcf"
          val sexcheckFile = root + ".sexcheck"

          s = ImputeSex.run(s, Array("-m", "0.0", "--include-par"))
          s = ExportVCF.run(s, Array("-o", vcfFile))
          s = Write.run(s, Array("-o", root + ".vds"))

          hadoopConf.copy(vcfFile, localVCFFile)

          s"plink --vcf ${ uriPath(localVCFFile) } --const-fid --check-sex --silent --out ${ uriPath(localRoot) }" !

          hadoopConf.copy(localSexcheckFile, sexcheckFile)

          val plinkResult = parsePlinkSexCheck(sexcheckFile)

          val (_, imputedSexQuery) = s.vds.querySA("if (sa.imputesex.isFemale) 2 else 1")
          val (_, fQuery) = s.vds.querySA("sa.imputesex.Fstat")

          val hailResult = s.vds.sampleIdsAndAnnotations.map { case (sample, sa) =>
            (sample, (imputedSexQuery(sa).map(_.asInstanceOf[Int]), fQuery(sa).map(_.asInstanceOf[Double])))
          }.toMap

          assert(plinkResult.keySet == hailResult.keySet)

          val result = plinkResult.forall { case (sample, (plinkSex, plinkF)) =>

            val (hailSex, hailF) = hailResult(sample)

            val resultSex = plinkSex == hailSex
            val resultF = plinkF.liftedZip(hailF).forall { case (p, h) => math.abs(p - h) < 1e-3 }

            if (resultSex && resultF)
              true
            else {
              println(s"$sample plink=${
                plinkSex.liftedZip(plinkF).getOrElse("NA")
              } hail=${
                hailSex.liftedZip(hailF).getOrElse("NA")
              } $resultSex $resultF")
              false
            }
          }

          val countAnnotated = AnnotateVariantsExpr.run(s,
            Array("-c", "va.maf = let a = gs.map(g => g.oneHotAlleles(v)).sum() in a[1] / a.sum"))
          val sexcheck2 = ImputeSex.run(countAnnotated, Array("--pop-freq", "va.maf", "--include-par"))

          result &&
            sexcheck2.vds.saSignature == s.vds.saSignature &&
            sexcheck2.vds.sampleAnnotationsSimilar(s.vds)
        }
      }
  }

  @Test def testImputeSexPlinkVersion() {
    Spec.check()
  }
}
