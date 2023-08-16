package is.hail.io.compress

import is.hail.HailSuite
import is.hail.check.Gen
import is.hail.check.Prop.forAll
import is.hail.utils._
import htsjdk.samtools.util.BlockCompressedFilePointerUtil
import is.hail.expr.ir.GenericLines
import org.apache.commons.io.IOUtils
import org.apache.spark.sql.Row
import org.apache.{hadoop => hd}
import org.testng.annotations.Test
import is.hail.TestUtils._

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.io.Source

class TestFileInputFormat extends hd.mapreduce.lib.input.TextInputFormat {
  override def getSplits(job: hd.mapreduce.JobContext): java.util.List[hd.mapreduce.InputSplit] = {
    val hConf = job.getConfiguration

    val splitPoints = hConf.get("bgz.test.splits").split(",").map(_.toLong)

    val splits = new mutable.ArrayBuffer[hd.mapreduce.InputSplit]
    val files = listStatus(job).asScala
    assert(files.length == 1)

    val file = files.head
    val path = file.getPath
    val length = file.getLen
    val fileSystem = path.getFileSystem(hConf)
    val blkLocations = fileSystem.getFileBlockLocations(file, 0, length)

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

class BGzipCodecSuite extends HailSuite {
  val uncompPath = "src/test/resources/sample.vcf"

  // is actually a bgz file
  val gzPath = "src/test/resources/sample.vcf.gz"

  /*
   * bgz.test.sample.vcf.bgz was created as follows:
   *  - split sample.vcf into 60-line chunks: `split -l 60 sample.vcf sample.vcf.`
   *  - move the line boundary on two chunks by 1 character in different directions
   *  - bgzip compressed the chunks
   *  - stripped the empty terminate block in chunks except ad and ag (the last)
   *  - concatenated the chunks
   */
  val compPath = "src/test/resources/bgz.test.sample.vcf.bgz"

  def compareLines(lines2: IndexedSeq[String], lines: IndexedSeq[String]): Unit = {
    val n2 = lines2.length
    val n = lines.length
    assert(n2 == n)
    var i = 0
    while (i < lines.length) {
      assert(lines(i) == lines2(i))
      i += 1
    }
  }

  @Test def testGenericLinesSimpleUncompressed() {
    val lines = Source.fromFile(uncompPath).getLines().toFastIndexedSeq

    val uncompStatus = fs.getFileListEntry(uncompPath)
    var i = 0
    while (i < 16) {
      val lines2 = GenericLines.collect(
        fs,
        GenericLines.read(fs, Array(uncompStatus), Some(i), None, None, false, false))
      compareLines(lines2, lines)
      i += 1
    }
  }

  @Test def testGenericLinesSimpleBGZ() {
    val lines = Source.fromFile(uncompPath).getLines().toFastIndexedSeq

    val compStatus = fs.getFileListEntry(compPath)
    var i = 0
    while (i < 16) {
      val lines2 = GenericLines.collect(
        fs,
        GenericLines.read(fs, Array(compStatus), Some(i), None, None, false, false))
      compareLines(lines2, lines)
      i += 1
    }
  }

  @Test def testGenericLinesSimpleGZ() {
    val lines = Source.fromFile(uncompPath).getLines().toFastIndexedSeq

    // won't split, just run once
    val gzStatus = fs.getFileListEntry(gzPath)
    val lines2 = GenericLines.collect(
      fs,
      GenericLines.read(fs, Array(gzStatus), Some(7), None, None, false, true))
    compareLines(lines2, lines)
  }

  @Test def testGenericLinesRefuseGZ() {
    interceptFatal("Cowardly refusing") {
      val gzStatus = fs.getFileListEntry(gzPath)
      GenericLines.read(fs, Array(gzStatus), Some(7), None, None, false, false)
    }
  }

  @Test def testGenericLinesRandom() {
    val lines = Source.fromFile(uncompPath).getLines().toFastIndexedSeq

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

      val contexts = (0 until (splits.length - 1))
        .map { i =>
          val end = makeVirtualOffset(splits(i + 1), 0)
          Row(i, 0, compPath, splits(i), end, true)
        }
      val lines2 = GenericLines.collect(fs, GenericLines.read(fs, contexts, false, false))
      compareLines(lines2, lines)
      true
    }
    p.check()
  }

  @Test def test() {
    sc.hadoopConfiguration.setLong("mapreduce.input.fileinputformat.split.minsize", 1L)

    val uncompIS = fs.open(uncompPath)
    val uncomp = IOUtils.toByteArray(uncompIS)
    uncompIS.close()

    val decompIS = new BGzipInputStream(fs.openNoCompression(compPath))
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

      val jobConf = new hd.conf.Configuration(sc.hadoopConfiguration)
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

  @Test def testVirtualSeek() {
    // real offsets of the start of some blocks, paired with the offset to the next block
    val blockStarts = Array[(Long, Long)]((0, 14653), (69140, 82949), (133703, 146664), (181362, 192983 /* end of file */))
    // NOTE: maxBlockSize is the length of all blocks other than the last
    val maxBlockSize = 65280
    // number determined by counting bytes from sample.vcf from uncompBlockStarts.last to the end of the file
    val lastBlockLen = 55936
    // offsets into the uncompressed file
    val uncompBlockStarts = Array[Int](0, 326400, 652800, 913920)

    val uncompPath = "src/test/resources/sample.vcf"
    val compPath = "src/test/resources/sample.vcf.gz"

    using(fs.openNoCompression(uncompPath)) { uncompIS =>
      using(new BGzipInputStream(fs.openNoCompression(compPath))) { decompIS =>

      val fromEnd = 48 // arbitrary number of bytes from the end of block to attempt to seek to
      for (((cOff, nOff), uOff) <- blockStarts.zip(uncompBlockStarts);
           e <- Seq(0, 1024, maxBlockSize - fromEnd)) {
        val decompData = new Array[Byte](100)
        val uncompData = new Array[Byte](100)
        val extra = if (cOff == blockStarts.last._1 && e == maxBlockSize - fromEnd)
            lastBlockLen - fromEnd
          else
            e
        val vOff = BlockCompressedFilePointerUtil.makeFilePointer(cOff, extra)

        decompIS.virtualSeek(vOff)
        assert(decompIS.getVirtualOffset() == vOff);
        uncompIS.seek(uOff + extra)

        val decompRead = decompIS.readRepeatedly(decompData)
        val uncompRead = uncompIS.readRepeatedly(uncompData)

        assert(decompRead == uncompRead, s"""compressed offset: ${ cOff }
          |decomp bytes read: ${ decompRead }
          |uncomp bytes read: ${ uncompRead }\n""".stripMargin)
        assert(decompData sameElements uncompData, s"data differs for compressed offset: ${ cOff }")
        val expectedVirtualOffset = if (extra == lastBlockLen - fromEnd)
            BlockCompressedFilePointerUtil.makeFilePointer(nOff, 0)
          else if (extra == maxBlockSize - fromEnd)
            BlockCompressedFilePointerUtil.makeFilePointer(nOff, decompRead - fromEnd)
          else
            BlockCompressedFilePointerUtil.makeFilePointer(cOff, extra + decompRead)
        assert(expectedVirtualOffset == decompIS.getVirtualOffset())
      }

      // here we test reading from the middle of a block to it's end
      val decompData = new Array[Byte](maxBlockSize)
      val toSkip = 20000
      val vOff = BlockCompressedFilePointerUtil.makeFilePointer(blockStarts(2)._1, toSkip)
      decompIS.virtualSeek(vOff)
      assert(decompIS.getVirtualOffset() == vOff)
      assert(decompIS.read(decompData) == maxBlockSize - 20000)
      assert(decompIS.getVirtualOffset() == BlockCompressedFilePointerUtil.makeFilePointer(blockStarts(2)._2, 0))

      // Trying to seek to the end of a block should fail
      intercept[java.io.IOException] {
        val vOff = BlockCompressedFilePointerUtil.makeFilePointer(blockStarts(1)._1, maxBlockSize)
        decompIS.virtualSeek(vOff)
      }

      // Trying to seek past the end of a block should fail
      intercept[java.io.IOException] {
        val vOff = BlockCompressedFilePointerUtil.makeFilePointer(blockStarts(0)._1, maxBlockSize + 1)
        decompIS.virtualSeek(vOff)
      }

      // Trying to seek to the end of the last block should fail
      intercept[java.io.IOException] {
        val vOff = BlockCompressedFilePointerUtil.makeFilePointer(blockStarts.last._1, lastBlockLen)
        decompIS.virtualSeek(vOff)
      }

      // trying to seek to the end of file should succeed
      decompIS.virtualSeek(0)
      val eofOffset = BlockCompressedFilePointerUtil.makeFilePointer(blockStarts.last._2, 0)
      decompIS.virtualSeek(eofOffset)
      assert(-1 == decompIS.read())

      // seeking past end of file directly should fail
      decompIS.virtualSeek(0)
      intercept[java.io.IOException] {
        val vOff = BlockCompressedFilePointerUtil.makeFilePointer(blockStarts.last._2, 1)
        decompIS.virtualSeek(vOff)
      }
    }}
  }
}
