package is.hail.io.compress

import is.hail.SparkSuite
import is.hail.check.Gen
import is.hail.check.Prop.forAll
import is.hail.utils._
import org.apache.commons.io.IOUtils
import org.apache.{hadoop => hd}
import org.testng.annotations.Test

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.io.Source

class TestFileInputFormat extends hd.mapreduce.lib.input.TextInputFormat {
  override def getSplits(job: hd.mapreduce.JobContext): java.util.List[hd.mapreduce.InputSplit] = {
    val hConf = job.getConfiguration

    val splitPoints = hConf.get("bgz.test.splits").split(",").map(_.toLong)
    println(splitPoints.toSeq)

    val splits = new mutable.ArrayBuffer[hd.mapreduce.InputSplit]
    val files = listStatus(job).asScala
    assert(files.length == 1)

    val file = files.head
    val path = file.getPath
    val length = file.getLen
    val fs = path.getFileSystem(hConf)
    val blkLocations = fs.getFileBlockLocations(file, 0, length)

    Array.tabulate(splitPoints.length - 1) { i =>
      val s = splitPoints(i)
      val e = splitPoints(i + 1)
      val splitSize = e - s

      val blkIndex = getBlockIndex(blkLocations, s)
      splits += makeSplit(path, s, splitSize, blkLocations(blkIndex).getHosts, blkLocations(blkIndex).getCachedHosts)
    }

    splits.asJava
  }
}

class BGzipCodecSuite extends SparkSuite {
  @Test def test() {
    sc.hadoopConfiguration.setLong("mapreduce.input.fileinputformat.split.minsize", 1L)

    val uncompPath = "src/test/resources/sample.vcf"

    /*
     * bgz.test.sample.vcf.bgz was created as follows:
     *  - split sample.vcf into 60-line chunks: `split -l 60 sample.vcf sample.vcf.`
     *  - move the line boundary on two chunks by 1 character in different directions
     *  - bgzip compressed the chunks
     *  - stripped the empty terminate block in chunks except ad and ag (the last)
     *  - concatenated the chunks
     */
    val compPath = "src/test/resources/bgz.test.sample.vcf.bgz"

    val uncompHPath = new hd.fs.Path(uncompPath)
    val compHPath = new hd.fs.Path(compPath)

    val fs = uncompHPath.getFileSystem(hadoopConf)

    val uncompIS = fs.open(uncompHPath)
    val uncomp = IOUtils.toByteArray(uncompIS)
    uncompIS.close()

    val decompIS = new BGzipInputStream(fs.open(compHPath))
    val decomp = IOUtils.toByteArray(decompIS)
    decompIS.close()

    assert(uncomp.sameElements(decomp))

    val lines = Source.fromBytes(uncomp).getLines.toArray

    assert(sc.textFile(uncompPath).collectOrdered()
      .sameElements(lines))

    for (i <- 1 until 20) {
      val linesRDD = sc.textFile(compPath, i)
      assert(linesRDD.partitions.length == i)
      assert(linesRDD.collectOrdered().sameElements(lines))
    }

    val compLength = 195353
    val compSplits = Array[Long](6566, 20290, 33438, 41165, 56691, 70278, 77419, 92522, 106310, 112477, 112505, 124593,
      136405, 144293, 157375, 169172, 175174, 186973, 195325)

    val g = for (n <- Gen.oneOfGen(
      Gen.choose(0, 10),
      Gen.choose(0, 100));
      rawSplits <- Gen.buildableOfN[Array](n,
        Gen.oneOfGen(Gen.choose(0L, compLength),
          Gen.applyGen(Gen.oneOf[(Long) => Long](identity, _ - 1, _ + 1),
            Gen.oneOfSeq(compSplits)))))
      yield
        (Array(0L, compLength) ++ rawSplits).distinct.sorted

    val p = forAll(g) { splits =>

      val jobConf = new hd.conf.Configuration(hadoopConf)
      jobConf.set("bgz.test.splits", splits.mkString(","))
      val rdd = sc.newAPIHadoopFile[hd.io.LongWritable, hd.io.Text, TestFileInputFormat](
        compPath,
        classOf[TestFileInputFormat],
        classOf[hd.io.LongWritable],
        classOf[hd.io.Text],
        jobConf)

      val rddLines = rdd.map(_._2.toString).collectOrdered()
      rddLines.sameElements(lines)
    }
    p.check()
  }
}
