package is.hail.methods

import java.io.InputStream
import java.io.OutputStream
import java.nio.file.Files
import java.nio.file.Paths

import org.apache.spark.mllib.linalg.distributed._
import is.hail.SparkSuite
import org.apache.hadoop
import org.apache.spark.sql.Row
import org.testng.annotations.Test
import is.hail.check._
import is.hail.check.Prop._
import is.hail.expr.{TDouble, TInt, TString}
import is.hail.variant.VariantDataset
import is.hail.variant.VSMSubgen
import is.hail.stats._
import is.hail.utils.{TextTableConfiguration, TextTableReader, _}
import scala.sys.process._
import is.hail.methods.PCRelate.DistributedMatrix
import is.hail.methods.PCRelate.DistributedMatrixImplicits._

class PCRelateSuite extends SparkSuite {
  private def toI(a: Any): Int =
    a.asInstanceOf[Int]

  private def toD(a: Any): Double =
    a.asInstanceOf[Double]

  private def toS(a: Any): String =
    a.asInstanceOf[String]

  @Test def compareToPCRelateR() {
    val vds: VariantDataset = BaldingNicholsModel(hc, 3, 100, 10000, None, None, 0, None, UniformDist(0.4,0.6)).splitMulti()

    val truth: Map[(String, String), Double] = {
      val tmpfile = tmpDir.createTempFile(prefix = "pcrelate")
      val localTmpfile = tmpDir.createLocalTempFile(prefix = "pcrelate")
      val pcRelateScript = tmpDir.createLocalTempFile(prefix = "pcrelateScript")

      vds.exportPlink(tmpfile)

      for (suffix <- Seq(".bed", ".bim", ".fam")) {
        hadoopConf.copy(tmpfile + suffix, localTmpfile + suffix)
      }

      s"Rscript src/test/resources/is/hail/methods/runPcRelate.R ${uriPath(localTmpfile)}" !

      val genomeFormat = TextTableConfiguration(
        types = Map(
          ("ID1", TString), ("ID2", TString), ("nsnp", TInt), ("kin", TDouble), ("k0", TDouble), ("k1", TDouble), ("k2", TDouble)),
        separator = " +")

      hadoopConf.copy(localTmpfile + ".out", tmpfile + ".out")

      val (_, rdd) = TextTableReader.read(sc)(Array(tmpfile + ".out"), genomeFormat)
      rdd.collect()
        .map(_.value)
        .map { ann =>
          val row = ann.asInstanceOf[Row]
          val id1 = toS(row(0))
          val id2 = toS(row(1))
          val nsnp = toI(row(2))
          val kin = toD(row(3))
          val k0 = toD(row(4))
          val k1 = toD(row(5))
          val k2 = toD(row(6))
          ((id1, id2), kin)
        }
        .toMap
    }

    val indexToId: Map[Int, String] = vds.sampleIds.zipWithIndex.map { case (id, index) => (index, id) }.toMap

    val pcs = SamplePCA.justScores(vds, 2)
    println("First two PCs for every sample:")
    println(pcs)
    val hailPcRelate = DistributedMatrix[BlockMatrix].toCoordinateMatrix(PCRelate[BlockMatrix](vds, pcs).phiHat).entries.collect()
      .filter(me => me.i < me.j)
      .map(me => ((indexToId(me.i.toInt), indexToId(me.j.toInt)), me.value))
      .toMap

    assert(mapSameElements(hailPcRelate, truth, (x: Double, y: Double) => D_==(x, y)))
  }

  // @Test
  def foo() {
    val vds: VariantDataset = hc.read("/Users/dking/projects/hail-data/profile.vds").splitMulti()
    val pcs = SamplePCA.justScores(vds, 3)
    val result = PCRelate(vds, pcs)
  }
}
