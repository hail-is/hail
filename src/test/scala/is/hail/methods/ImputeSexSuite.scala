package is.hail.methods

import is.hail.SparkSuite
import is.hail.check.Prop._
import is.hail.check.Properties
import is.hail.utils._
import is.hail.variant._
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

    val plinkSafeBiallelicVDS = VariantSampleMatrix.gen(hc, VSMSubgen.plinkSafeBiallelic)
      .resize(1000)
      .map(vds => vds.filterVariants { case (v, va, gs) => v.isAutosomalOrPseudoAutosomal && v.contig.toUpperCase != "X" && v.contig.toUpperCase != "Y" })
      .filter(vds => vds.countVariants > 2 && vds.nSamples >= 2)

    property("hail generates same results as PLINK v1.9") =
      forAll(plinkSafeBiallelicVDS) { case (vds: VariantDataset) =>

        var mappedVDS = vds.copy[GenotypeMatrixT](rdd =
          vds.rdd.map { case (v, (va, gs)) => (v.copy(contig = "X"), (va, gs)) }
            .toOrderedRDD)
          .variantQC()
          .filterVariantsExpr("va.qc.AC > 1 && va.qc.AF >= 1e-8 && " +
            "va.qc.nCalled * 2 - va.qc.AC > 1 && va.qc.AF <= 1 - 1e-8")

        if (vds.nSamples < 5 || vds.countVariants() < 5) {
          true
        } else {
          val localRoot = tmpDir.createLocalTempFile("plinksexCheck")
          val localVCFFile = localRoot + ".vcf"
          val localSexcheckFile = localRoot + ".sexcheck"

          val root = tmpDir.createTempFile("plinksexCheck")
          val vcfFile = root + ".vcf"
          val sexcheckFile = root + ".sexcheck"

          mappedVDS = mappedVDS.imputeSex(0.0, includePAR = true)
          mappedVDS.exportVCF(vcfFile)
          mappedVDS.write(root + ".vds")

          hadoopConf.copy(vcfFile, localVCFFile)

          s"plink --vcf ${ uriPath(localVCFFile) } --const-fid --check-sex --silent --out ${ uriPath(localRoot) }" !

          hadoopConf.copy(localSexcheckFile, sexcheckFile)

          val plinkResult = parsePlinkSexCheck(sexcheckFile)

          val (_, imputedSexQuery) = mappedVDS.querySA("if (sa.imputesex.isFemale) 2 else 1")
          val (_, fQuery) = mappedVDS.querySA("sa.imputesex.Fstat")

          val hailResult = mappedVDS.sampleIdsAndAnnotations.map { case (sample, sa) =>
            (sample, (Option(imputedSexQuery(sa)).map(_.asInstanceOf[Int]), Option(fQuery(sa)).map(_.asInstanceOf[Double])))
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

          val countAnnotated = mappedVDS.annotateVariantsExpr(
            "va.maf = let a = gs.map(g => g.oneHotAlleles(v)).sum() in a[1] / a.sum()")
          val sexcheck2 = countAnnotated.imputeSex(popFreqExpr = Some("va.maf"), includePAR = true)

          result &&
            sexcheck2.saSignature == mappedVDS.saSignature &&
            sexcheck2.sampleAnnotationsSimilar(mappedVDS)
        }
      }
  }

  @Test def testImputeSexPlinkVersion() {
    Spec.check()
  }
}
