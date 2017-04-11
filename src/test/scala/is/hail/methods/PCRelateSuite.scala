package is.hail.methods

import java.io.InputStream
import java.io.OutputStream
import java.nio.file.Files
import java.nio.file.Paths

import org.apache.spark.mllib.linalg.distributed._
import is.hail.SparkSuite
import org.apache.hadoop
import org.apache.spark.sql.Row
import org.apache.spark.mllib.linalg._
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

    println("total samples, total variants, total genotypes: " + vds.count(true))

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

    // val pcs = SamplePCA.justScores(vds, 2)
    // println("First two PCs for every sample:")
    // println(pcs)

    // from pcrelate
    val pcs = new DenseMatrix(100, 2, Array(0.11895518, -0.04465795,
      -0.02330527,  0.16551836,
      0.12272068, -0.03813280,
      -0.11141409, -0.07689534,
      0.11977809, -0.04708180,
      -0.01988176,  0.16353260,
      -0.11623726, -0.07944966,
      0.11725875, -0.04179644,
      0.11342134, -0.04024008,
      -0.11425587, -0.06857095,
      -0.11154459, -0.08184987,
      -0.02261558,  0.16640316,
      0.12195489, -0.03411192,
      -0.02416329,  0.16948604,
      0.11755419, -0.03829568,
      -0.11616325, -0.07583378,
      -0.11474992, -0.07527435,
      -0.01860560,  0.16258323,
      -0.11749020, -0.07256255,
      -0.01874347,  0.16825959,
      -0.02514715,  0.16001995,
      0.11666823, -0.04745648,
      0.11843766, -0.03797916,
      0.11639382, -0.04646813,
      -0.02126929,  0.16348075,
      -0.11502417, -0.07776868,
      -0.02419285,  0.16787277,
      0.11367936, -0.04504793,
      -0.11016314, -0.07313079,
      -0.10743512, -0.06896256,
      0.11922722, -0.04311908,
      -0.02673449,  0.16592825,
      0.11754620, -0.04190583,
      -0.02615306,  0.16869286,
      -0.10994804, -0.06933998,
      0.11278270, -0.03624087,
      -0.02964590,  0.16500273,
      -0.02541646,  0.17094434,
      0.11602380, -0.04081858,
      -0.11046509, -0.07453944,
      -0.11595665, -0.07820502,
      -0.11780090, -0.07925109,
      0.11423876, -0.04431392,
      -0.11275236, -0.06988538,
      -0.11499812, -0.07641669,
      0.11998528, -0.03879976,
      0.11914212, -0.04979651,
      0.10877894, -0.04912006,
      -0.11114438, -0.07661524,
      0.11446803, -0.04548145,
      -0.11426967, -0.07726671,
      -0.11135893, -0.07583894,
      -0.10685058, -0.08316447,
      0.11575838, -0.04601371,
      0.11583665, -0.04775715,
      -0.02423396,  0.17528047,
      -0.11137321, -0.07916517,
      0.11904631, -0.04181044,
      -0.11118143, -0.07712557,
      -0.11502882, -0.08440555,
      -0.02967079,  0.16941726,
      0.11820777, -0.03431504,
      0.11689959, -0.04443531,
      0.12181073, -0.04327931,
      0.11457496, -0.04651510,
      0.11113775, -0.05069890,
      -0.11447934, -0.07096629,
      -0.11116516, -0.08614145,
      -0.02401157,  0.16752636,
      -0.02277920,  0.16848652,
      -0.11030987, -0.07593532,
      -0.01938910,  0.16522360,
      -0.11782106, -0.07302765,
      -0.10980741, -0.07223088,
      0.12008458, -0.04921489,
      -0.11597779, -0.07588200,
      -0.11228507, -0.07312495,
      0.12132585, -0.04445203,
      -0.01894356,  0.17234548,
      0.11978386, -0.03791617,
      -0.02948153,  0.16081175,
      -0.02000453,  0.16803902,
      -0.02095907,  0.16372494,
      -0.02025943,  0.17047066,
      0.12239995, -0.04246086,
      -0.11709772, -0.07065607,
      0.10937650, -0.03937162,
      0.11529775, -0.04775951,
      -0.11663542, -0.07706614,
      -0.10793354, -0.08029881,
      -0.11408043, -0.07766923,
      -0.02340344,  0.16841757,
      0.11682013, -0.04284017,
      0.11638756, -0.04617343,
      0.12361904, -0.03942926,
      0.11893835, -0.04315691,
      -0.02957361,  0.16335841,
      -0.02636824,  0.16859793,
      -0.11404581, -0.06913945,
      0.11787566, -0.04730442), true)

    val hailPcRelate = DistributedMatrix[BlockMatrix].toCoordinateMatrix(PCRelate[BlockMatrix](vds, pcs).phiHat).entries.collect()
      .filter(me => me.i < me.j)
      .map(me => ((indexToId(me.i.toInt), indexToId(me.j.toInt)), me.value))
      .toMap

    assert(mapSameElements(hailPcRelate, truth, (x: Double, y: Double) => D_==(x, y)))
  }

  @Test
  def foo() {
    import is.hail.keytable._
    import is.hail.annotations.Annotation
    import is.hail.expr.{TStruct, _}

    println(s"started foo ${System.nanoTime()}")

    val vds: VariantDataset = hc.read("/Users/dking/projects/hail-data/profile-with-case.vds").splitMulti()

    val indexToId: Map[Int, String] = vds.sampleIds.zipWithIndex.map { case (id, index) => (index, id) }.toMap
    val pcs = SamplePCA.justScores(vds, 2)
    println(s"one with pcs ${System.nanoTime()}")

    val result = DistributedMatrix[BlockMatrix].toCoordinateMatrix(PCRelate[BlockMatrix](vds, pcs).phiHat).entries
      .filter(me => me.i < me.j)
      .map(me => Annotation(indexToId(me.i.toInt), indexToId(me.j.toInt), me.value))

    KeyTable(vds.hc, result, TStruct("i" -> TString, "j" -> TString, "kin" -> TDouble), Array("i", "j"))
      .write("/tmp/profile-with-case-hail-pc-relate.out")
  }
}
