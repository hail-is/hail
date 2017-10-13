package is.hail.annotations

import is.hail.SparkSuite
import is.hail.check._
import is.hail.check.Arbitrary._
import is.hail.expr._
import is.hail.utils._
import org.apache.spark.SparkEnv
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class UnsafeSuite extends SparkSuite {
  @Test def testCodec() {
    val region = MemoryBuffer()
    val region2 = MemoryBuffer()
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
      val ttBc = BroadcastTypeTree(sc, t)
      val ur = new UnsafeRow(ttBc, region, offset)

      val aos = new ArrayOutputStream()
      val en = new Encoder(new LZ4OutputBuffer(aos))
      en.writeRegionValue(f, region, offset)
      en.flush()

      region2.clear()
      val ais = new ArrayInputStream(aos.a, aos.off)
      val dec = new Decoder(new LZ4InputBuffer(ais))
      val offset2 = dec.readRegionValue(f, region2)
      val ur2 = new UnsafeRow(ttBc, region2, offset2)

      assert(t.valuesSimilar(a, ur2))

      true
    }
    p.check()
  }

  @Test def testRegionValue() {
    val region = MemoryBuffer()
    val region2 = MemoryBuffer()
    val rvb = new RegionValueBuilder(region)
    val rvb2 = new RegionValueBuilder(region2)

    val g = Type.genArb
      .flatMap(t => Gen.zip(Gen.const(t), t.genValue))
      .filter { case (t, a) => a != null }
      .resize(10)
    val p = Prop.forAll(g) { case (t, a) =>
      val ttBc = BroadcastTypeTree(sc, t)

      t.typeCheck(a)

      region.clear()
      rvb.start(t)
      rvb.addAnnotation(t, a)
      val offset = rvb.end()

      val ur = UnsafeRow.read(region, offset, t, ttBc)
      assert(t.valuesSimilar(a, ur))

      // test visitor
      val rv = RegionValue(region, offset)
      rv.pretty(t)

      region2.clear()
      rvb2.start(t)
      rvb2.addAnnotation(t, ur)
      val offset2 = rvb2.end()

      val ur2 = UnsafeRow.read(region2, offset2, t, ttBc)
      assert(t.valuesSimilar(a, ur2))

      t match {
        case t: TStruct =>
          // don't clear, just add on
          rvb2.start(t)
          rvb2.addUnsafeRow(t, ur.asInstanceOf[UnsafeRow])
          val offset3 = rvb2.end()

          val ur3 = new UnsafeRow(BroadcastTypeTree(sc, t), region2, offset3)
          assert(t.valuesSimilar(a, ur2))
        case _ =>
      }

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

  @Test def testUnsafeOrdering() {
    val region = MemoryBuffer()
    val region2 = MemoryBuffer()
    val rvb = new RegionValueBuilder(region)
    val rvb2 = new RegionValueBuilder(region2)

    val g = Type.genStruct
      .flatMap(t => Gen.zip(Gen.const(t), Gen.zip(t.genValue, t.genValue), arbitrary[Boolean]))
      .filter { case (t, (a1, a2), b) => a1 != null && a2 != null }
      .resize(10)
    val p = Prop.forAll(g) { case (t, (a1, a2), b) =>

      t.typeCheck(a1)
      t.typeCheck(a2)

      region.clear()
      rvb.start(t.fundamentalType)
      rvb.addRow(t, a1.asInstanceOf[Row])
      val offset = rvb.end()

      val ur1 = new UnsafeRow(BroadcastTypeTree(sc, t), region, offset)
      assert(t.valuesSimilar(a1, ur1))

      region2.clear()
      rvb2.start(t.fundamentalType)
      rvb2.addRow(t, a2.asInstanceOf[Row])
      val offset2 = rvb2.end()

      val ur2 = new UnsafeRow(BroadcastTypeTree(sc, t), region2, offset2)
      assert(t.valuesSimilar(a2, ur2))

      val ord = t.ordering(b)
      val uord = t.unsafeOrdering(b)

      val c1 = ord.compare(a1, a2)
      val c2 = ord.compare(ur1, ur2)
      val c3 = uord.compare(ur1.region, ur1.offset, ur2.region, ur2.offset)

      val p1 = math.signum(c1) == math.signum(c2)
      val p2 = math.signum(c2) == math.signum(c3)

      val p = p1 && p2
      if (!p) {
        println(s"t=$t")
        println(s"a1=$a1")
        println(s"a2=$a2")
        println(s"c1=$c1, c2=$c2, c3=$c3")
      }
      p
    }
    p.check()
  }

  @Test def unsafeSer() {
    val region = MemoryBuffer()
    val rvb = new RegionValueBuilder(region)

    val path = tmpDir.createTempFile(extension = "ser")

    val g = Type.genStruct
      .flatMap(t => Gen.zip(Gen.const(t), t.genValue))
      .filter { case (t, a) => a != null }
    val p = Prop.forAll(g) { case (t, a) =>
      region.clear()
      rvb.start(t)
      rvb.addRow(t, a.asInstanceOf[Row])
      val offset = rvb.end()
      val ur = new UnsafeRow(BroadcastTypeTree(sc, t), region, offset)

      hadoopConf.writeObjectFile(path) { out =>
        out.writeObject(ur)
      }

      val ur2 = hadoopConf.readObjectFile(path) { in =>
        in.readObject().asInstanceOf[UnsafeRow]
      }

      assert(t.valuesSimilar(ur, ur2))

      true
    }
    p.check()
  }

  @Test def unsafeKryo() {
    val conf = sc.getConf // force sc
    val ser = SparkEnv.get.serializer.asInstanceOf[KryoSerializer]
    val kryo = ser.newKryo()

    val region = MemoryBuffer()
    val rvb = new RegionValueBuilder(region)
    val path = tmpDir.createTempFile(extension = "ser")
    val g = Type.genStruct
      .flatMap(t => Gen.zip(Gen.const(t), t.genValue))
      .filter { case (t, a) => a != null }
    val p = Prop.forAll(g) { case (t, a) =>
      region.clear()
      rvb.start(t)
      rvb.addRow(t, a.asInstanceOf[Row])
      val offset = rvb.end()
      val ur = new UnsafeRow(BroadcastTypeTree(sc, t), region, offset)

      hadoopConf.writeKryoFile(path) { out =>
        kryo.writeObject(out, ur)
      }

      val ur2 = hadoopConf.readKryoFile(path) { in =>
        kryo.readObject[UnsafeRow](in, classOf[UnsafeRow])
      }

      assert(t.valuesSimilar(ur, ur2))

      true
    }
    p.check()
  }

  @Test def testRegionSer() {
    val region = MemoryBuffer()
    val path = tmpDir.createTempFile(extension = "ser")
    val g = Gen.buildableOf[Array, Byte](arbitrary[Byte])
    val p = Prop.forAll(g) { (a: Array[Byte]) =>
      region.clear()
      region.appendBytes(a)

      hadoopConf.writeObjectFile(path) { out =>
        out.writeObject(region)
      }

      val region2 = hadoopConf.readObjectFile(path) { in =>
        in.readObject().asInstanceOf[MemoryBuffer]
      }

      assert(region2.size.toInt == a.length)
      val a2 = region2.loadBytes(0, region2.size.toInt)
      assert(a2 sameElements a)

      true
    }
  }

  @Test def testRegionKryo() {
    val conf = sc.getConf // force sc
    val ser = SparkEnv.get.serializer.asInstanceOf[KryoSerializer]
    val kryo = ser.newKryo()

    val region = MemoryBuffer()
    val path = tmpDir.createTempFile(extension = "ser")
    val g = Gen.buildableOf[Array, Byte](arbitrary[Byte])
    val p = Prop.forAll(g) { (a: Array[Byte]) =>
      region.clear()
      region.appendBytes(a)

      hadoopConf.writeKryoFile(path) { out =>
        kryo.writeObject(out, region)
      }

      val region2 = hadoopConf.readKryoFile(path) { in =>
        kryo.readObject[MemoryBuffer](in, classOf[MemoryBuffer])
      }

      assert(region2.size.toInt == a.length)
      val a2 = region2.loadBytes(0, region2.size.toInt)
      assert(a2 sameElements a)

      true
    }
  }
}