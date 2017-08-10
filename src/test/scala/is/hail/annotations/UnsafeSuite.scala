package is.hail.annotations

import is.hail.SparkSuite
import is.hail.check._
import is.hail.expr._
import is.hail.utils._
import org.apache.commons.math3.random.RandomDataGenerator
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class UnsafeSuite extends SparkSuite {
  @Test def testRegionValue() {
    val region = MemoryBuffer()
    val region2 = MemoryBuffer()
    val rvb = new RegionValueBuilder(region)
    val rvb2 = new RegionValueBuilder(region2)

    val g = Type.genStruct
      .flatMap(t => Gen.zip(Gen.const(t), t.genValue))
      .filter { case (t, a) => a != null }
      .resize(10)
    val p = Prop.forAll(g) { case (t, a) =>
      t.typeCheck(a)
      val f = t.fundamentalType

      region.clear()
      rvb.start(f)
      rvb.addRow(t, a.asInstanceOf[Row])
      val offset = rvb.end()

      val ur = new UnsafeRow(BroadcastTypeTree(sc, t), region, offset)
      assert(t.valuesSimilar(a, ur))

      region2.clear()
      rvb2.start(f)
      rvb2.addRow(t, ur)
      val offset2 = rvb2.end()

      val ur2 = new UnsafeRow(BroadcastTypeTree(sc, t), region2, offset2)
      assert(t.valuesSimilar(a, ur2))

      // don't clear, just add on
      rvb2.start(f)
      rvb2.addUnsafeRow(t, ur)
      val offset3 = rvb2.end()

      val ur3 = new UnsafeRow(BroadcastTypeTree(sc, t), region2, offset3)
      assert(t.valuesSimilar(a, ur2))

      true
    }
    p.check()
  }

  @Test def testMemoryBuffer() {
    val buff = MemoryBuffer()

    buff.appendLong(124L)
    buff.appendByte(2)
    buff.appendByte(1)
    buff.appendByte(4)
    buff.appendInt(1234567)
    buff.appendDouble(1.1)

    assert(buff.loadLong(0) == 124L)
    assert(buff.loadByte(8) == 2)
    assert(buff.loadByte(9) == 1)
    assert(buff.loadByte(10) == 4)
    assert(buff.loadInt(12) == 1234567)
    assert(buff.loadDouble(16) == 1.1)
  }

  val g = (for {
    s <- Gen.size
    // prefer smaller type and bigger values
    fraction <- Gen.choose(0.1, 0.3)
    x = (fraction * s).toInt
    y = s - x
    t <- Type.genStruct.resize(x)
    v <- t.genNonmissingValue.resize(y)
  } yield (t, v)).filter(_._2 != null)

  @Test def testPacking() {

    def makeStruct(types: Type*): TStruct = {
      TStruct(types.zipWithIndex.map { case (t, i) => (s"f$i", t) }: _*)
    }

    val t1 = makeStruct( // missing byte is 0
      TInt32, //4-8
      TInt32, //8-12
      TFloat64, //16-24
      TBoolean, //1-2
      TBoolean, //2-3
      TBoolean, //3-4
      TBoolean, //12-13
      TBoolean) //13-14
    assert(t1.byteOffsets.toSeq == Seq(4, 8, 16, 1, 2, 3, 12, 13))
    assert(t1.byteSize == 24)

    val t2 = makeStruct( //missing bytes 0, 1
      TBoolean, //2-3
      TInt32, //4-8
      TInt32, //8-12
      TFloat64, //16-24
      TInt32, //12-16
      TInt32, //24-28
      TFloat64, //32-40
      TInt32, //28-32
      TBoolean, //3-4
      TFloat64, //40-48
      TBoolean) //48-49

    assert(t2.byteOffsets.toSeq == Seq(2, 4, 8, 16, 12, 24, 32, 28, 3, 40, 48))
    assert(t2.byteSize == 49)

    val t3 = makeStruct((0 until 512).map(_ => TFloat64): _*)
    assert(t3.byteSize == (512 / 8) + 512 * 8)
    val t4 = makeStruct((0 until 256).flatMap(_ => Iterator(TInt32, TInt32, TFloat64, TBoolean)): _*)
    assert(t4.byteSize == 256 * 4 / 8 + 256 * 4 * 2 + 256 * 8 + 256)
  }

  @Test def testEmptySize() {
    assert(TStruct().byteSize == 0)
  }
}