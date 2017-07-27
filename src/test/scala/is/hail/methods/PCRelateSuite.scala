package is.hail.methods

import java.io.InputStream
import java.io.OutputStream
import java.nio.file.Files
import java.nio.file.Paths

import breeze.linalg.{DenseMatrix => BDM, _}
import is.hail.keytable._
import is.hail.annotations.Annotation
import is.hail.expr.{TStruct, _}
import org.apache.spark.mllib.linalg.distributed._
import is.hail.SparkSuite
import org.apache.hadoop
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.mllib.linalg._
import org.testng.annotations.Test
import is.hail.check._
import is.hail.check.Prop._
import is.hail.expr.{TDouble, TInt, TString}
import is.hail.variant.VariantDataset
import is.hail.variant.VSMSubgen
import is.hail.stats._
import is.hail.utils.{TextTableReader, _}
import scala.sys.process._
import is.hail.distributedmatrix.DistributedMatrix
import is.hail.distributedmatrix.DistributedMatrix.implicits._

class PCRelateSuite extends SparkSuite {
  private val blockSize: Int = 8192

  private def toI(a: Any): Int =
    a.asInstanceOf[Int]

  private def toD(a: Any): Double =
    a.asInstanceOf[Double]

  private def toS(a: Any): String =
    a.asInstanceOf[String]

  def runPcRelateHail(vds: VariantDataset, pcs: DenseMatrix, maf: Double): Map[(String, String), (Double, Double, Double, Double)] =
    PCRelate.toPairRdd(vds, pcs, maf, blockSize).collect().toMap.asInstanceOf[Map[(String, String), (Double, Double, Double, Double)]]

  def runPcRelateR(
    vds: VariantDataset,
    maf: Double,
    rFile: String = "src/test/resources/is/hail/methods/runPcRelate.R"): Map[(String, String), (Double, Double, Double, Double)] = {

    val tmpfile = tmpDir.createTempFile(prefix = "pcrelate")
    val localTmpfile = tmpDir.createLocalTempFile(prefix = "pcrelate")
    val pcRelateScript = tmpDir.createLocalTempFile(prefix = "pcrelateScript")

    vds.exportPlink(tmpfile)

    for (suffix <- Seq(".bed", ".bim", ".fam")) {
      hadoopConf.copy(tmpfile + suffix, localTmpfile + suffix)
    }

    s"Rscript $rFile ${uriPath(localTmpfile)} ${maf}" !

    val columns = Map(
      ("ID1", TString),
      ("ID2", TString),
      ("nsnp", TDouble),
      ("kin", TDouble),
      ("k0", TDouble),
      ("k1", TDouble),
      ("k2", TDouble))
    val separator = " +"

    hadoopConf.copy(localTmpfile + ".out", tmpfile + ".out")

    val (_, rdd) = TextTableReader.read(sc)(Array(tmpfile + ".out"), columns, separator=separator)
    rdd.collect()
      .map(_.value)
      .map { ann =>
      val row = ann.asInstanceOf[Row]
      val id1 = toS(row(0))
      val id2 = toS(row(1))
      val nsnp = toD(row(2)).toInt
      val kin = toD(row(3))
      val k0 = toD(row(4))
      val k1 = toD(row(5))
      val k2 = toD(row(6))
      ((id1, id2), (kin, k0, k1, k2))
    }
      .toMap
  }

  def compareDoubleQuartuplets(cmp: (Double, Double) => Boolean)(x: (Double, Double, Double, Double), y: (Double, Double, Double, Double)): Boolean =
    cmp(x._1, y._1) && cmp(x._2, y._2) && cmp(x._3, y._3) && cmp(x._4, y._4)

  @Test
  def trivialReference() {
    val genotypeMatrix = new BDM(4,8,Array(0,0,0,0, 0,0,1,0, 0,1,0,1, 0,1,1,1, 1,0,0,0, 1,0,1,0, 1,1,0,1, 1,1,1,1)) // column-major, columns == variants
    val vds = vdsFromGtMatrix(hc)(genotypeMatrix, Some(Array("s1","s2","s3","s4")))
    val pcsArray = Array(0.0, 1.0, 1.0, 0.0,  1.0, 1.0, 0.0, 0.0) // NB: this **MUST** be the same as the PCs used by the R script
    val pcs = new DenseMatrix(4,2,pcsArray)
    val us = runPcRelateHail(vds, pcs, maf=0.0)
    val truth = PCRelateReferenceImplementation(vds, pcs, maf=0.0)
    assert(mapSameElements(us, truth, compareDoubleQuartuplets((x, y) => math.abs(x - y) < 1e-14)))
  }

  @Test
  def trivialReferenceMatchesR() {
    val genotypeMatrix = new BDM(4,8,Array(0,0,0,0, 0,0,1,0, 0,1,0,1, 0,1,1,1, 1,0,0,0, 1,0,1,0, 1,1,0,1, 1,1,1,1)) // column-major, columns == variants
    val vds = vdsFromGtMatrix(hc)(genotypeMatrix, Some(Array("s1","s2","s3","s4")))
    val pcsArray = Array(0.0, 1.0, 1.0, 0.0,  1.0, 1.0, 0.0, 0.0) // NB: this **MUST** be the same as the PCs used by the R script
    val pcs = new DenseMatrix(4,2,pcsArray)
    val usRef = PCRelateReferenceImplementation(vds, pcs, maf=0.0)
    val truth = runPcRelateR(vds, maf=0.0, "src/test/resources/is/hail/methods/runPcRelateOnTrivialExample.R")
    assert(mapSameElements(usRef, truth, compareDoubleQuartuplets((x, y) => math.abs(x - y) < 1e-14)))
  }

  @Test def baldingNicholsMatchesReference() {
    for {
      n <- Seq(50, 100//, 500
      )
      seed <- Seq(0, 1//, 2
      )
      nVariants <- Seq(1000, 10000//, 50000
      )
    } {
      val vds: VariantDataset = BaldingNicholsModel(hc, 3, n, nVariants, None, None, seed, None, UniformDist(0.1,0.9)).splitMulti()
      val pcs = SamplePCA.justScores(vds, 2)
      val truth = PCRelateReferenceImplementation(vds, pcs, maf=0.01)
      val actual = runPcRelateHail(vds, pcs, maf=0.01)

      printToFile(new java.io.File(s"/tmp/baldingNicholsMatchesReference-$n-$seed-$nVariants.out")) { pw =>
        pw.println(Array("s1","s2","uskin","usz0","usz1","usz2","themkin","themz0","themz1","themz2").mkString(","))
        for ((k, (hkin, hz0, hz1, hz2)) <- actual) {
          val (rkin, rz0, rz1, rz2) = truth(k)
          val (s1, s2) = k
          pw.println(Array(s1,s2,hkin,hz0,hz1,hz2,rkin,rz0,rz1,rz2).mkString(","))
        }
      }

      println(s"$n $seed $nVariants")
      assert(mapSameElements(actual, truth, compareDoubleQuartuplets((x, y) => math.abs(x - y) < 1e-14)))
    }
  }

  @Test def baldingNicholsReferenceMatchesR() {
    for {
      n <- Seq(50, 100//, 500
      )
      seed <- Seq(0, 1//, 2
      )
      nVariants <- Seq(1000, 10000//, 50000
      )
    } {
      val vds: VariantDataset = BaldingNicholsModel(hc, 3, n, nVariants, None, None, seed, None, UniformDist(0.1,0.9)).splitMulti()
      val pcs = SamplePCA.justScores(vds, 2)
      val truth = runPcRelateR(vds, maf=0.01)
      val actual = PCRelateReferenceImplementation(vds, pcs, maf=0.01)

      printToFile(new java.io.File(s"/tmp/baldingNicholsMatchesReference-$n-$seed-$nVariants.out")) { pw =>
        pw.println(Array("s1","s2","uskin","usz0","usz1","usz2","themkin","themz0","themz1","themz2").mkString(","))
        for ((k, (hkin, hz0, hz1, hz2)) <- actual) {
          val (rkin, rz0, rz1, rz2) = truth(k)
          val (s1, s2) = k
          pw.println(Array(s1,s2,hkin,hz0,hz1,hz2,rkin,rz0,rz1,rz2).mkString(","))
        }
      }

      println(s"$n $seed $nVariants")
      assert(mapSameElements(actual, truth, compareDoubleQuartuplets((x, y) => D_==(x, y, tolerance=1e-2))))
    }
  }

  @Test
  def thousandGenomesTriosMatchesReference() {
    val trios = Array("HG00702", "HG00656", "HG00657",
      "HG00733", "HG00731", "HG00732",
      "HG02024", "HG02026", "HG02025",
      "HG03715","HG03713",
      "HG03948","HG03673",
      "NA19240", "NA19239", "NA19238",
      "NA19675", "NA19679", "NA19678",
      "NA19685", "NA19661", "NA19660")

    val siblings = Array("NA19713", "NA19985",
      "NA20289", "NA20341",
      "NA20334", "NA20336")

    val secondOrder = Array("HG01936", "HG01983")

    val r = scala.util.Random

    val profile225 = hc.readVDS("/Users/dking/projects/hail-data/profile225-splitmulti-hardcalls.vds")
    for (fraction <- Seq(0.0625//0.125, 0.25, 0.5
    )) {
      val subset = r.shuffle(profile225.sampleIds).slice(0, (profile225.nSamples * fraction).toInt).toSet

      def underStudy(s: String) =
        subset.contains(s) || trios.contains(s) || siblings.contains(s) || secondOrder.contains(s)

      val vds = profile225
        .filterSamples((s, sa) => underStudy(s.asInstanceOf[String]))
        .filterVariantsExpr("v => va.qc.AF > 0.20 && va.qc.AF < 0.80 && v.isAutosomal")
        .cache()
      val pcs = SamplePCA.justScores(vds.coalesce(10), 2)

      val (truth, pcRelateTime) = time(PCRelateReferenceImplementation(vds, pcs, maf=0.01))
      val (hailPcRelate, hailTime) = time(runPcRelateHail(vds, pcs, maf=0.01))

      println(s"on fraction: $fraction; pc relate: $pcRelateTime, hail: $hailTime, ratio: ${pcRelateTime / hailTime.toDouble}")

      printToFile(new java.io.File(s"/tmp/thousandGenomesTriosMatchesReference-$fraction.out")) { pw =>
        pw.println(Array("s1","s2","uskin","usz0","usz1","usz2","themkin","themz0","themz1","themz2").mkString(","))
        for ((k, (hkin, hz0, hz1, hz2)) <- hailPcRelate) {
          val (rkin, rz0, rz1, rz2) = truth(k)
          val (s1, s2) = k
          pw.println(Array(s1,s2,hkin,hz0,hz1,hz2,rkin,rz0,rz1,rz2).mkString(","))
        }
      }

      assert(mapSameElements(hailPcRelate, truth, compareDoubleQuartuplets((x, y) => math.abs(x - y) < 0.01)))
    }
  }

}
