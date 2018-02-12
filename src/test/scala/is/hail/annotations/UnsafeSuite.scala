package is.hail.annotations

import is.hail.SparkSuite
import is.hail.check._
import is.hail.check.Arbitrary._
import is.hail.expr.types._
import is.hail.io._
import is.hail.utils._
import is.hail.variant.{GenomeReference, Variant}
import org.apache.spark.SparkEnv
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.Row
import org.json4s.jackson.JsonMethods
import org.testng.annotations.Test

class UnsafeSuite extends SparkSuite {
  @Test def testCodec() {
    val region = Region()
    val region2 = Region()
    val rvb = new RegionValueBuilder(region)

    val path = tmpDir.createTempFile(extension = "ser")

    val g = Type.genStruct
      .flatMap(t => Gen.zip(Gen.const(t), t.genValue))
      .filter { case (t, a) => a != null }
    val p = Prop.forAll(g) { case (t, a) =>
      t.typeCheck(a)

      CodecSpec.codecSpecs.foreach { codecSpec =>
        region.clear()
        rvb.start(t)
        rvb.addRow(t, a.asInstanceOf[Row])
        val offset = rvb.end()
        val ur = new UnsafeRow(t, region, offset)

        val aos = new ArrayOutputStream()
        val en = codecSpec.buildEncoder(aos)
        en.writeRegionValue(t, region, offset)
        en.flush()

        region2.clear()
        val ais = new ArrayInputStream(aos.a, aos.off)
        val dec = codecSpec.buildDecoder(ais)
        val offset2 = dec.readRegionValue(t, region2)
        val ur2 = new UnsafeRow(t, region2, offset2)

        assert(t.valuesSimilar(a, ur2))
      }

      true
    }
    p.check()
  }

  @Test def testRegionValue() {
    val region = Region()
    val region2 = Region()
    val rvb = new RegionValueBuilder(region)
    val rvb2 = new RegionValueBuilder(region2)

    val g = Type.genArb
      .flatMap(t => Gen.zip(Gen.const(t), t.genValue, Gen.choose(0, 100), Gen.choose(0, 100)))
      .filter { case (t, a, n, n2) => a != null }
    val p = Prop.forAll(g) { case (t, a, n, n2) =>
      t.typeCheck(a)

      // test addAnnotation
      region.clear()
      region.allocate(1, n) // preallocate

      rvb.start(t)
      rvb.addAnnotation(t, a)
      val offset = rvb.end()

      val ur = UnsafeRow.read(t, region, offset)
      assert(t.valuesSimilar(a, ur), s"$a vs $ur")

      // test visitor
      val rv = RegionValue(region, offset)
      rv.pretty(t)

      // test addAnnotation from ur
      region2.clear()
      region2.allocate(1, n2) // preallocate
      rvb2.start(t)
      rvb2.addAnnotation(t, ur)
      val offset2 = rvb2.end()

      val ur2 = UnsafeRow.read(t, region2, offset2)
      assert(t.valuesSimilar(a, ur2), s"$a vs $ur2")

      // test addRegionValue
      region2.clear()
      region2.allocate(1, n2) // preallocate
      rvb2.start(t)
      rvb2.addRegionValue(t, region, offset)
      val offset3 = rvb2.end()
      val ur3 = UnsafeRow.read(t, region2, offset3)
      assert(t.valuesSimilar(a, ur3), s"$a vs $ur3")

      // test addRegionValue nested
      t match {
        case t: TStruct =>
          region2.clear()
          region2.allocate(1, n) // preallocate
          rvb2.start(t)
          rvb2.addAnnotation(t, Row.fromSeq(a.asInstanceOf[Row].toSeq))
          val offset4 = rvb2.end()
          val ur4 = new UnsafeRow(t, region2, offset4)
          assert(t.valuesSimilar(a, ur4))
        case _ =>
      }

      // test addRegionValue to same region
      rvb.start(t)
      rvb.addRegionValue(t, region, offset)
      val offset5 = rvb.end()
      val ur5 = UnsafeRow.read(t, region, offset5)
      assert(t.valuesSimilar(a, ur5))

      // test addRegionValue to same region nested
      t match {
        case t: TStruct =>
          rvb.start(t)
          rvb.addAnnotation(t, Row.fromSeq(a.asInstanceOf[Row].toSeq))
          val offset6 = rvb.end()
          val ur6 = new UnsafeRow(t, region, offset6)
          assert(t.valuesSimilar(a, ur6))
        case _ =>
      }

      true
    }
    p.check()
  }

  @Test def testRegion() {
    val buff = Region()

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
      TInt32(), //4-8
      TInt32(), //8-12
      TFloat64(), //16-24
      TBoolean(), //1-2
      TBoolean(), //2-3
      TBoolean(), //3-4
      TBoolean(), //12-13
      TBoolean()) //13-14
    assert(t1.byteOffsets.toSeq == Seq(4, 8, 16, 1, 2, 3, 12, 13))
    assert(t1.byteSize == 24)

    val t2 = makeStruct( //missing bytes 0, 1
      TBoolean(), //2-3
      TInt32(), //4-8
      TInt32(), //8-12
      TFloat64(), //16-24
      TInt32(), //12-16
      TInt32(), //24-28
      TFloat64(), //32-40
      TInt32(), //28-32
      TBoolean(), //3-4
      TFloat64(), //40-48
      TBoolean()) //48-49

    assert(t2.byteOffsets.toSeq == Seq(2, 4, 8, 16, 12, 24, 32, 28, 3, 40, 48))
    assert(t2.byteSize == 49)

    val t3 = makeStruct((0 until 512).map(_ => TFloat64()): _*)
    assert(t3.byteSize == (512 / 8) + 512 * 8)
    val t4 = makeStruct((0 until 256).flatMap(_ => Iterator(TInt32(), TInt32(), TFloat64(), TBoolean())): _*)
    assert(t4.byteSize == 256 * 4 / 8 + 256 * 4 * 2 + 256 * 8 + 256)
  }

  @Test def testEmptySize() {
    assert(TStruct().byteSize == 0)
  }

  @Test def orderingRegression() {
    val region = Region()
    val region2 = Region()
    val rvb = new RegionValueBuilder(region)
    val rvb2 = new RegionValueBuilder(region2)

    val v1 = Variant("1", 1, "T", Array("A", "G"))
    val v2 = Variant("1", 1, "T", "C")

    val t = TVariant(GenomeReference.GRCh37)

    rvb.start(t)
    rvb.addAnnotation(t, v1)
    val rv = RegionValue(region, rvb.end())

    rvb2.start(t)
    rvb2.addAnnotation(t, v2)
    val rv2 = RegionValue(region2, rvb2.end())

    assert(math.signum(t.ordering.compare(v1, v2, missingGreatest = true)) ==
      math.signum(t.unsafeOrdering(missingGreatest = true).compare(rv, rv2)))
  }

  @Test def testUnsafeOrdering() {
    val region = Region()
    val region2 = Region()
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

      val ur1 = new UnsafeRow(t, region, offset)
      assert(t.valuesSimilar(a1, ur1))

      region2.clear()
      rvb2.start(t.fundamentalType)
      rvb2.addRow(t, a2.asInstanceOf[Row])
      val offset2 = rvb2.end()

      val ur2 = new UnsafeRow(t, region2, offset2)
      assert(t.valuesSimilar(a2, ur2))

      val ord = t.ordering
      val uord = t.unsafeOrdering(b)

      val c1 = ord.compare(a1, a2, b)
      val c2 = ord.compare(ur1, ur2, b)
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
    val region = Region()
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
      val ur = new UnsafeRow(t, region, offset)

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

    val region = Region()
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
      val ur = new UnsafeRow(t, region, offset)

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
    val region = Region()
    val path = tmpDir.createTempFile(extension = "ser")
    val g = Gen.buildableOf[Array](arbitrary[Byte])
    val p = Prop.forAll(g) { (a: Array[Byte]) =>
      region.clear()
      region.appendBytes(a)

      hadoopConf.writeObjectFile(path) { out =>
        out.writeObject(region)
      }

      val region2 = hadoopConf.readObjectFile(path) { in =>
        in.readObject().asInstanceOf[Region]
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

    val region = Region()
    val path = tmpDir.createTempFile(extension = "ser")
    val g = Gen.buildableOf[Array](arbitrary[Byte])
    val p = Prop.forAll(g) { (a: Array[Byte]) =>
      region.clear()
      region.appendBytes(a)

      hadoopConf.writeKryoFile(path) { out =>
        kryo.writeObject(out, region)
      }

      val region2 = hadoopConf.readKryoFile(path) { in =>
        kryo.readObject[Region](in, classOf[Region])
      }

      assert(region2.size.toInt == a.length)
      val a2 = region2.loadBytes(0, region2.size.toInt)
      assert(a2 sameElements a)

      true
    }
  }
}
