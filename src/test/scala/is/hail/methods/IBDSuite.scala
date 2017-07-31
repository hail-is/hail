package is.hail.methods

import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.check.Prop._
import is.hail.check.{Gen, Properties}
import is.hail.expr.{TFloat64, TInt32, TString}
import is.hail.utils.AbsoluteFuzzyComparable._
import is.hail.utils.{AbsoluteFuzzyComparable, TextTableReader, _}
import is.hail.variant._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

import scala.language._
import scala.sys.process._

class IBDSuite extends SparkSuite {

  def toI(a: Any): Int =
    a.asInstanceOf[Int]

  def toD(a: Any): Double =
    a.asInstanceOf[Double]

  def toS(a: Any): String =
    a.asInstanceOf[String]

  implicit object ibdAbsoluteFuzzyComparable extends AbsoluteFuzzyComparable[IBDInfo] {
    def absoluteEq(tolerance: Double, x: IBDInfo, y: IBDInfo) = {
      def feq(x: Double, y: Double) = AbsoluteFuzzyComparable.absoluteEq(tolerance, x, y)

      def NaNorFeq(x: Double, y: Double) =
        x.isNaN && y.isNaN || feq(x, y)

      NaNorFeq(x.Z0, y.Z0) && NaNorFeq(x.Z1, y.Z1) && NaNorFeq(x.Z2, y.Z2) && NaNorFeq(x.PI_HAT, y.PI_HAT)
    }
  }

  implicit object eibdAbsoluteFuzzyComparable extends AbsoluteFuzzyComparable[ExtendedIBDInfo] {
    def absoluteEq(tolerance: Double, x: ExtendedIBDInfo, y: ExtendedIBDInfo) = {
      def feq(x: Double, y: Double) = AbsoluteFuzzyComparable.absoluteEq(tolerance, x, y)

      AbsoluteFuzzyComparable.absoluteEq(tolerance, x.ibd, y.ibd) &&
        feq(x.ibs0, y.ibs0) && feq(x.ibs1, y.ibs1) && feq(x.ibs2, y.ibs2)
    }
  }

  private def runPlinkIBD(vds: VariantDataset,
    min: Option[Double] = None,
    max: Option[Double] = None): Map[(Annotation, Annotation), ExtendedIBDInfo] = {

    val tmpdir = tmpDir.createTempFile(prefix = "plinkIBD")
    val localTmpdir = tmpDir.createLocalTempFile(prefix = "plinkIBD")

    val vcfFile = tmpdir + ".vcf"
    val localVCFFile = localTmpdir + ".vcf"

    vds.exportVCF(vcfFile)

    hadoopConf.copy(vcfFile, localVCFFile)

    val thresholdString = min.map(x => s"--min $x").getOrElse("") + " " +
      max.map(x => s"--max $x").getOrElse("")

    s"plink --double-id --allow-extra-chr --vcf ${ uriPath(localVCFFile) } --genome full --out ${ uriPath(localTmpdir) } " + thresholdString !

    val genomeFile = tmpdir + ".genome"
    val localGenomeFile = localTmpdir + ".genome"

    hadoopConf.copy(localGenomeFile, genomeFile)

    val (_, rdd) = TextTableReader.read(sc)(Array(tmpdir + ".genome"),
      types = Map(("IID1", TString), ("IID2", TString), ("Z0", TFloat64), ("Z1", TFloat64), ("Z2", TFloat64),
        ("PI_HAT", TFloat64), ("IBS0", TInt32), ("IBS1", TInt32), ("IBS2", TInt32)),
      separator = " +"
    )

    rdd.collect()
      .map(_.value)
      .map { ann =>
        // _, fid1, iid1, fid2, iid2, rt, ez, z0, z1, z2, pihat, phe, dst, ppc, ratio, ibs0, ibs1, ibs2, homhom, hethet
        val row = ann.asInstanceOf[Row]
        val iid1 = toS(row(2)): Annotation
        val iid2 = toS(row(4)): Annotation
        val z0 = toD(row(7))
        val z1 = toD(row(8))
        val z2 = toD(row(9))
        val pihat = toD(row(10))
        val ibs0 = toI(row(15))
        val ibs1 = toI(row(16))
        val ibs2 = toI(row(17))
        ((iid1, iid2), ExtendedIBDInfo(IBDInfo(z0, z1, z2, pihat), ibs0, ibs1, ibs2))
      }
      // if min or max is enabled, we remove NaNs, plink does not
      .filter { case (_, eibd) => (min.isEmpty && max.isEmpty) || !eibd.hasNaNs }
      .toMap
  }

  object Spec extends Properties("IBD") {
    val plinkSafeBiallelicVDS = VariantSampleMatrix.gen(hc, VSMSubgen.plinkSafeBiallelic)
      .resize(1000)
      .map(vds => vds.filterVariants { case (v, va, gs) => v.isAutosomalOrPseudoAutosomal })
      .filter(vds => vds.countVariants > 2 && vds.nSamples >= 2)

    property("hail generates same result as plink 1.9") =
      forAll(plinkSafeBiallelicVDS) { vds =>
        val us = IBD(vds).collect().toMap

        val plink = runPlinkIBD(vds)

        mapSameElements(us, plink,
          (x: ExtendedIBDInfo, y: ExtendedIBDInfo) => AbsoluteFuzzyComparable.absoluteEq(tolerance, x, y))
      }

    property("hail generates same result as plink 1.9 with min and/or max") =
      forAll(plinkSafeBiallelicVDS,
        Gen.option(Gen.choose(0.0, 1.0), 0.8),
        Gen.option(Gen.choose(0.0, 1.0), 0.8)) { (vds, maybeMin, maybeMax) =>
        // ensure min <= max
        val validMax = maybeMax.map(max => maybeMin match {
          case None => max
          case Some(min) if max < min => (max / min) * (1.0 - min) + min
          case Some(min) => max
        })

        val us = IBD(vds, min = maybeMin, max = validMax).collect().toMap

        val plink = runPlinkIBD(vds, maybeMin, validMax)

        mapSameElements(us, plink,
          (x: ExtendedIBDInfo, y: ExtendedIBDInfo) => AbsoluteFuzzyComparable.absoluteEq(tolerance, x, y))
      }
  }

  // plink rounds to the nearest ten-thousandth
  val tolerance = 5e-5

  @Test def testIBDPlink() {
    Spec.check()
  }

  @Test def ibdPlinkSameOnRealVCF() {
    val vds = hc.importVCF("src/test/resources/sample.vcf")

    val us = IBD(vds).collect().toMap

    val plink = runPlinkIBD(vds)

    assert(mapSameElements(us, plink,
      (x: ExtendedIBDInfo, y: ExtendedIBDInfo) => AbsoluteFuzzyComparable.absoluteEq(tolerance, x, y)))
  }

  @Test def ibdSchemaCorrect() {
    val vds = hc.importVCF("src/test/resources/sample.vcf")
    val us = IBD.toKeyTable(vds.hc, IBD(vds)).typeCheck()
  }

  //Maximal Independent Set and IBD are both tested on their own, so this just makes sure expressions work right for orderings.
  @Test def ibdPrune() {
    val vds = hc.baldingNicholsModel(1, 3, 5, seed = 0)

    val prunedVDS = vds.ibdPrune(0.0, Some("if (s1.toInt < s2.toInt) 1 else if (s1.toInt == s2.toInt) 0 else -1"), bounded = true)
    assert(prunedVDS.sampleIds.length == 1)
    assert(prunedVDS.sampleIds(0) == "2")

    val prunedVDS2 = vds.ibdPrune(0.0, None, bounded = true)
    assert(prunedVDS2.sampleIds.length == 1)
    assert(prunedVDS2.sampleIds(0) == "0")
  }
}
