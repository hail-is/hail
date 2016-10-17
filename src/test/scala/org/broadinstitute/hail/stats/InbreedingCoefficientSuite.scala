package org.broadinstitute.hail.stats

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.Properties
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.variant._
import org.testng.annotations.Test

import scala.language._
import scala.sys.process._


class InbreedingCoefficientSuite extends SparkSuite {

  def parsePlinkHet(file: String): Map[String, (Option[Double], Option[Long], Option[Double], Option[Long])] =
    hadoopConf.readLines(file)(_.drop(1).map(_.map { line =>
      val Array(fid, iid, obsHom, expHom, numCalled, f) = line.trim.split("\\s+")
      val fMod = f match {
        case "nan" => None
        case x: String => Option(x.toDouble)
        case _ => throw new IllegalArgumentException
      }

      (iid, (fMod, Option(obsHom.toLong), Option(expHom.toDouble), Option(numCalled.toLong)))
    }.value
    ).toMap)

  object Spec extends Properties("InbreedingCoefficient") {

    val plinkSafeBiallelicVDS = VariantSampleMatrix.gen(sc, VSMSubgen.plinkSafeBiallelic)
      .resize(1000)
      .map(vds => vds.filterVariants { case (v, va, gs) => v.isAutosomalOrPseudoAutosomal && v.contig.toUpperCase != "X" && v.contig.toUpperCase != "Y" })
      .filter(vds => vds.nVariants > 2 && vds.nSamples >= 2)

    property("hail generates same results as PLINK v1.9") =
      forAll(plinkSafeBiallelicVDS) { case (vds: VariantSampleMatrix[Genotype]) =>

        var s = State(sc, sqlContext).copy(vds = vds)

        s = VariantQC.run(s)
        s = FilterVariantsExpr.run(s, Array("--keep", "-c", "va.qc.AC > 1 && va.qc.AF >= 1e-8 && va.qc.nCalled * 2 - va.qc.AC > 1 && va.qc.AF <= 1 - 1e-8"))

        if (s.vds.nSamples < 5 || s.vds.nVariants < 5) {
          true
        } else {
          val localRoot = tmpDir.createLocalTempFile("ibcCheck")
          val localVCFFile = localRoot + ".vcf"
          val localIbcFile = localRoot + ".het"

          val root = tmpDir.createTempFile("ibcCheck")
          val vcfFile = root + ".vcf"
          val ibcFile = root + ".het"

          s = AnnotateSamplesExpr.run(s, Array("-c", "sa.het = gs.ibc(va.qc.AF)"))
          s = ExportVCF.run(s, Array("-o", vcfFile))

          hadoopConf.copy(vcfFile, localVCFFile)

          s"plink --vcf ${ uriPath(localVCFFile) } --allow-extra-chr --const-fid --het --silent --out ${ uriPath(localRoot) }" !

          hadoopConf.copy(localIbcFile, ibcFile)

          val plinkResult = parsePlinkHet(ibcFile)

          val (_, fQuery) = s.vds.querySA("sa.het.Fstat")
          val (_, obsHomQuery) = s.vds.querySA("sa.het.observedHoms")
          val (_, expHomQuery) = s.vds.querySA("sa.het.expectedHoms")
          val (_, nCalledQuery) = s.vds.querySA("sa.het.nCalled")

          val hailResult = s.vds.sampleIdsAndAnnotations.map { case (sample, sa) =>
            (sample, (fQuery(sa).map(_.asInstanceOf[Double]), obsHomQuery(sa).map(_.asInstanceOf[Long]), expHomQuery(sa).map(_.asInstanceOf[Double]), nCalledQuery(sa).map(_.asInstanceOf[Long])))
          }.toMap

          assert(plinkResult.keySet == hailResult.keySet)

          val result = plinkResult.forall { case (sample, (plinkF, plinkObsHom, plinkExpHom, plinkNCalled)) =>

            val (hailF, hailObsHom, hailExpHom, hailNCalled) = hailResult(sample)

            val resultF = plinkF.liftedZip(hailF).forall { case (p, h) => math.abs(p - h) < 1e-2 }
            val resultObsHom = plinkObsHom.liftedZip(hailObsHom).forall { case (p, h) => p == h }
            val resultExpHom = plinkExpHom.liftedZip(hailExpHom).forall { case (p, h) => math.abs(p - h) < 1e-2} //plink only gives two decimal places
            val resultNCalled = plinkNCalled.liftedZip(hailNCalled).forall { case (p, h) => p == h }

            if (resultF && resultObsHom && resultExpHom && resultNCalled)
              true
            else {
              println(s"$sample plink=${
                plinkF.liftedZip(plinkObsHom).liftedZip(plinkExpHom).liftedZip(plinkNCalled).getOrElse("NA")
              } hail=${
                hailF.liftedZip(hailObsHom).liftedZip(hailExpHom).liftedZip(hailNCalled).getOrElse("NA")
              } $resultF $resultObsHom $resultExpHom $resultNCalled")
              false
            }
          }

          result
        }
      }
  }

  @Test def testIbcPlinkVersion() {
    Spec.check()
  }
}

