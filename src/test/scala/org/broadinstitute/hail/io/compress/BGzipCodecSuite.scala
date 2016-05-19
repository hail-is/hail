package org.broadinstitute.hail.io.compress

import java.io.File
import java.net.URI
import java.nio.file.{Files, Paths}

import org.apache.commons.io.IOUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.compress.SplittableCompressionCodec.READ_MODE
import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test

import scala.collection.mutable
import scala.util.Random

class BGzipCodecSuite extends SparkSuite {
  @Test def test() {
    sc.hadoopConfiguration.set("io.compression.codecs", "org.apache.hadoop.io.compress.DefaultCodec,org.broadinstitute.hail.io.compress.BGzipCodec,org.apache.hadoop.io.compress.GzipCodec")

    val uncompFilename = "src/test/resources/sample.vcf"
    val compFilename = "src/test/resources/sample.vcf.bgz"

    val uncomp: Array[Byte] = Files.readAllBytes(Paths.get(uncompFilename))

    val bgzipCodec = new BGzipCodec()

    val conf = new Configuration()
    val fs = FileSystem.get(URI.create(compFilename), conf)

    val decomp = IOUtils.toByteArray(
      bgzipCodec.createInputStream(fs.open(new Path(compFilename))))
    assert(uncomp.sameElements(decomp))

    val compSize = new File(compFilename).length().toInt

    for (i <- 0 until 100) {
      val nSplits = Random.nextInt(10)
      val splits = Array.fill[Int](nSplits)(Random.nextInt(compSize)).sorted
      val splitRanges = (0 +: splits) zip (splits.map(_ - 1) :+ (compSize - 1))
      // println(splitRanges.mkString(","))
      val decomp: Array[Byte] = splitRanges.flatMap { case (start, end) =>
        val b = new mutable.ArrayBuilder.ofByte()
        val in = bgzipCodec.createInputStream(fs.open(new Path(compFilename)),
          null, start, end, READ_MODE.BYBLOCK)
        var done = false
        do {
          val ch = in.read()
          if (ch != -1 && in.getPos <= end)
            b += ch.toByte
          else
            done = true
        } while (!done)
        b.result()
      }

      assert(uncomp.sameElements(decomp))
    }

    val uncompLines = sc.textFile(uncompFilename)

    val decompLines = sc.textFile(compFilename, 10)
    assert(decompLines.partitions.size == 10)

    assert(uncompLines.collect().toSet
      == decompLines.collect().toSet)
  }
}
