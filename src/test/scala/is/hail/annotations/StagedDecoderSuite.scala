package is.hail.annotations

import is.hail.SparkSuite
import is.hail.expr._
import is.hail.utils._
import org.testng.annotations.{BeforeMethod, Test}
import is.hail.annotations.StagedDecoder._
import is.hail.io._
import is.hail.TestUtils._

class StagedDecoderSuite extends SparkSuite {

  val region = Region()
  val region2 = Region()
  val rv1 = RegionValue(region)
  val rv2 = RegionValue(region2)
  val rvb = new RegionValueBuilder(region)

  val verbose = true
  val staged = true

  @BeforeMethod def before() {
    region.clear()
    region2.clear()
    region.appendBytes(Array.fill[Byte](100)(0))
    region2.appendBytes(Array.fill[Byte](100)(0))
    region.clear()
    region2.clear()
  }

  def checkDecoding(t: Type) {
    if (verbose) {
      printRegion(rv1.region, "1")
      println(rv1.pretty(t))
    }
    val aos = new ArrayOutputStream()
    val en = new Encoder(new LZ4OutputBuffer(aos))
    en.writeRegionValue(t, rv1.region, rv1.offset)
    en.flush()
    val ais = new ArrayInputStream(aos.a, aos.off)
    val in = new LZ4InputBuffer(ais)
    val dec = new Decoder(in)
    region2.clear()
    if (staged) {
      val readRV = StagedDecoder.getRVReader(t)()
      rv2.setOffset(readRV(region2, dec))
    } else
      rv2.setOffset(dec.readRegionValue(t, region2))

    if (verbose) {
      printRegion(rv2.region, "2")
      println(rv2.pretty(t))
    }
    assert(rv1.pretty(t) == rv2.pretty(t))
    assert(java.util.Arrays.equals(region.loadBytes(0, region.size.toInt), region2.loadBytes(0, region2.size.toInt)))

  }

  @Test def decodeArray() {
    val t = TArray(TString())

    rvb.start(t)
    rvb.startArray(2)
    rvb.addString("hello")
    rvb.addString("world")
    rvb.endArray()
    rv1.setOffset(rvb.end())
    checkDecoding(t)

  }

  @Test def decodeStruct() {
    val t = TStruct("a"->TString(), "b"->TArray(TInt32()))

    rvb.start(t)
    rvb.startStruct()
    rvb.addString("hello")
    rvb.startArray(2)
    rvb.addInt(1)
    rvb.addInt(2)
    rvb.endArray()
    rvb.endStruct()
    rv1.setOffset(rvb.end())
    checkDecoding(t)

  }

  @Test def decodeArrayOfStruct() {
    val t = TArray(TStruct("a"->TString(), "b"->TInt32()))

    val strVals = Array[String]("hello", "world", "foo", "bar")

    rvb.start(t)
    rvb.startArray(2)
    for (i <- 0 until 2) {
      rvb.startStruct()
      rvb.addString(strVals(i))
      rvb.addInt(i+1)
      rvb.endStruct()
    }
    rvb.endArray()
    rv1.setOffset(rvb.end())

    checkDecoding(t)
  }

  def performanceComparison1(nCols: Int, nIter: Int): (Long, Long) = {
    val t = TStruct("a" -> TString(), "b" -> TArray(TInt32()))

    rvb.start(t)
    rvb.startStruct()
    rvb.addString("row1")
    rvb.startArray(nCols)
    for (i <- 1 to nCols) {
      rvb.addInt(i)
    }
    rvb.endArray()
    rvb.endStruct()
    rv1.setOffset(rvb.end())

    if (verbose) {
      printRegion(rv1.region, "1")
      println(rv1.pretty(t))
    }
    val aos = new ArrayOutputStream()
    val en = new Encoder(new LZ4OutputBuffer(aos))
    en.writeRegionValue(t, rv1.region, rv1.offset)
    en.flush()
    val ais1 = new ArrayInputStream(aos.a, aos.off)
    val dec1 = new Decoder(new LZ4InputBuffer(ais1))

    val start1 = System.nanoTime()
    for (i <- 0 until nIter) {
      region2.clear()
      ais1.clear()
      rv2.setOffset(dec1.readRegionValue(t, region2))
    }
    val stop1 = System.nanoTime()


    val ais2 = new ArrayInputStream(aos.a, aos.off)
    val dec2 = new Decoder(new LZ4InputBuffer(ais2))
    val readRV = StagedDecoder.getRVReader(t)()

    val start2 = System.nanoTime()
    for (i <- 0 until nIter) {
      region2.clear()
      ais2.clear()
      rv2.setOffset(readRV(region2, dec2))
    }
    val stop2 = System.nanoTime()

    (stop1 - start1, stop2 - start2)
  }

  def performanceComparison2(nCols: Int, nIter: Int): (Long, Long) = {
    val t = TArray(TStruct("a" -> TString(), "b" -> TInt32()))

    rvb.start(t)
    rvb.startArray(nCols)
    for (i <- 1 to nCols) {
      rvb.startStruct()
      rvb.addString("row1")
      rvb.addInt(i)
      rvb.endStruct()
    }
    rvb.endArray()
    rv1.setOffset(rvb.end())

    if (verbose) {
      printRegion(rv1.region, "1")
      println(rv1.pretty(t))
    }

    val aos = new ArrayOutputStream()
    val en = new Encoder(new LZ4OutputBuffer(aos))
    en.writeRegionValue(t, rv1.region, rv1.offset)
    en.flush()
    val ais1 = new ArrayInputStream(aos.a, aos.off)
    val dec1 = new Decoder(new LZ4InputBuffer(ais1))

    val start1 = System.nanoTime()
    for (i <- 0 until nIter) {
      region2.clear()
      ais1.clear()
      rv2.setOffset(dec1.readRegionValue(t, region2))
    }
    val stop1 = System.nanoTime()


    val ais2 = new ArrayInputStream(aos.a, aos.off)
    val dec2 = new Decoder(new LZ4InputBuffer(ais2))
    val readRV = StagedDecoder.getRVReader(t)()

    val start2 = System.nanoTime()
    for (i <- 0 until nIter) {
      region2.clear()
      ais2.clear()
      rv2.setOffset(readRV(region2, dec2))
    }
    val stop2 = System.nanoTime()

    (stop1 - start1, stop2 - start2)
  }

//  @Test
  def testPerformance() {
    val nIter = 1000

    printf("nCols   |  delta1  | percent1 |  delta2  | percent2%n")
    for (i <- Array(100, 1000, 10000)) {
      val (old1, new1) = performanceComparison1(i, nIter)
      val (old2, new2) = performanceComparison2(i, nIter)
      printf("%7d | %+7.5f | %+7.5f | %+7.5f | %+7.5f %n", i, (new1-old1)/1000000000.0, (new1 - old1).toDouble/old1, (new2-old2)/1000000000.0, (new2 - old2).toDouble/old2)
    }
  }
}
