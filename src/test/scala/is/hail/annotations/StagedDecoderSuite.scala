package is.hail.annotations

import is.hail.SparkSuite
import is.hail.expr._
import is.hail.utils._
import org.testng.annotations.{BeforeMethod, Test}
import is.hail.annotations.StagedDecoder._
import is.hail.io._
import is.hail.TestUtils._
import is.hail.check.{Gen, Prop}
import org.apache.spark.sql.Row

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
    assert(java.util.Arrays.equals(region.loadBytes(rv1.offset, (region.size - rv1.offset).toInt), region2.loadBytes(0, region2.size.toInt)))

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

  @Test def decodeBinary() {
    rvb.start(TInt32())
    rvb.addInt(5)
    rv1.setOffset(rvb.end())
    checkDecoding(TInt32())

    rvb.clear()

    rvb.start(TString())
    rvb.addString("hello")
    rv1.setOffset(rvb.end())
    checkDecoding(TString())

    rvb.clear()

    rvb.start(TString())
    rvb.addString("hello")
    rv1.setOffset(rvb.end())
    checkDecoding(TString())



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

  @Test def testStagedCodec() {
    val region = Region()
    val region2 = Region()
    val rvb = new RegionValueBuilder(region)

    val path = tmpDir.createTempFile(extension = "ser")

    val g = Type.genStruct
      .flatMap(t => Gen.zip(Gen.const(t), t.genValue))
      .filter { case (t, a) => a != null }
    val p = Prop.forAll(g) { case (t, a) =>
      t.typeCheck(a)
      val f = t.fundamentalType

      region.clear()
      rvb.start(f)
      rvb.addRow(t, a.asInstanceOf[Row])
      val offset = rvb.end()
      val ur = new UnsafeRow(t, region, offset)

      val aos = new ArrayOutputStream()
      val en = new Encoder(new LZ4OutputBuffer(aos))
      en.writeRegionValue(f, region, offset)
      en.flush()

      region2.clear()
      val ais = new ArrayInputStream(aos.a, aos.off)
      val dec = new Decoder(new LZ4InputBuffer(ais))
      val dF = StagedDecoder.getRVReader(t)()

      val offset2 = dF(region2, dec)
      val ur2 = new UnsafeRow(t, region2, offset2)

      assert(t.valuesSimilar(a, ur2))

      true
    }
    p.check()
  }
}
