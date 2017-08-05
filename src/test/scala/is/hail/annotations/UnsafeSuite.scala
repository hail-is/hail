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
    val rvb = new RegionValueBuilder(region)

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

  @Test def testCreation() {

    val rng = new RandomDataGenerator()
    rng.reSeed(Prop.seed)

    Prop.forAll(g) { case (t, a) =>
      val urb = new UnsafeRowBuilder(t)
      urb.setAll(a.asInstanceOf[Row])
      val unsafeRow = urb.result(sc)

      urb.clear()
      urb.setAll(a.asInstanceOf[Row])
      val ur2 = urb.result(sc)
      val p = unsafeRow == a

      assert(unsafeRow == ur2)

      if (!p) {
        println(
          s"""IN:  $a
             |OUT: $unsafeRow""".stripMargin)
      }

      p
    }.check()
  }

  @Test def testSubset() {

    val rng = new RandomDataGenerator()
    rng.reSeed(Prop.seed)

    val g2 = for {
      x <- g
      r <- Gen.parameterized { p => Gen.const((0 until x._1.size).toArray).filter(_ => p.rng.nextInt(0, 100) < 50) }
    } yield (x, r)

    Prop.forAll(g2) { case ((t, a), r) =>
      val urb = new UnsafeRowBuilder(t)
      val row = a.asInstanceOf[Row]
      var i = 0
      urb.setAll(row)
      val ur1 = urb.result(sc)
      urb.clear()
      val p1 = ur1 == row

      val t2 = TStruct(r.map(t.fields).map(f => f.name -> f.typ): _*)
      val urb2 = new UnsafeRowBuilder(t2)

      i = 0

      while (i < t2.size) {
        urb2.setFromUnsafe(i, r(i), ur1)
        i += 1
      }
      val ur2 = urb2.result(sc)
      urb2.clear()

      i = 0
      while (i < t2.size) {
        urb2.setFromRow(i, r(i), ur1)
        i += 1
      }
      val ur3 = urb2.result(sc)
      val p2 = ur3 == ur2

      val p3 = ur2 == Row.fromSeq(r.map(row.get))

      val p = p1 && p2 && p3

      if (!p) {
        println(
          s"""SCHEMA: $t
             |IN:   $a
             |OUT1: $ur1
             |OUT2: $ur2
             |SIZE: ${ ur1.region.sizeInBytes }/${ ur2.region.sizeInBytes }""".stripMargin)
      }

      p
    }.check()
  }

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