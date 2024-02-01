package is.hail.annotations

import is.hail.HailSuite
import is.hail.check._
import is.hail.io._
import is.hail.rvd.AbstractRVDSpec
import is.hail.types.physical._
import is.hail.types.virtual.{TArray, TStruct, Type}
import is.hail.utils._

import scala.util.Random

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import org.apache.spark.sql.Row
import org.json4s.jackson.Serialization
import org.testng.annotations.{DataProvider, Test}

class UnsafeSuite extends HailSuite {
  def subsetType(t: Type): Type = {
    t match {
      case t: TStruct =>
        TStruct(
          t.fields.filter(_ => Random.nextDouble() < 0.4)
            .map(f => f.name -> f.typ): _*
        )

      case t: TArray =>
        TArray(subsetType(t.elementType))

      case _ => t
    }
  }

  def subset(t: Type, requestedType: Type, a: Annotation): Annotation = {
    t match {
      case t2: TStruct =>
        val requestedType2 = requestedType.asInstanceOf[TStruct]
        if (a == null)
          null
        else {
          val a2 = a.asInstanceOf[Row]
          Row.fromSeq(requestedType2.fields.map { rf =>
            val f = t2.field(rf.name)
            subset(f.typ, rf.typ, a2.get(f.index))
          })
        }

      case _ => a
    }
  }

  def sm = ctx.stateManager

  @DataProvider(name = "codecs")
  def codecs(): Array[Array[Any]] =
    (BufferSpec.specs ++ Array(TypedCodecSpec(
      PCanonicalStruct("x" -> PInt64()),
      BufferSpec.default,
    )))
      .map(x => Array[Any](x))

  @Test(dataProvider = "codecs") def testCodecSerialization(codec: Spec): Unit = {
    implicit val formats = AbstractRVDSpec.formats
    assert(Serialization.read[Spec](codec.toString) == codec)

  }

  @Test def testCodec(): Unit = {
    val region = Region(pool = pool)
    val region2 = Region(pool = pool)
    val region3 = Region(pool = pool)
    val region4 = Region(pool = pool)

    val g = Type.genStruct
      .flatMap(t => Gen.zip(Gen.const(t), t.genValue(sm)))
      .filter { case (_, a) => a != null }
    val p = Prop.forAll(g) { case (t, a) =>
      assert(t.typeCheck(a))
      val pt = PType.canonical(t).asInstanceOf[PStruct]

      val requestedType = subsetType(t).asInstanceOf[TStruct]

      val a2 = subset(t, requestedType, a)
      assert(requestedType.typeCheck(a2))

      BufferSpec.specs.foreach { bufferSpec =>
        val codec = TypedCodecSpec(pt, bufferSpec)
        region.clear()
        val offset = pt.unstagedStoreJavaObject(sm, a, region)

        val aos = new ByteArrayOutputStream()
        val en = codec.buildEncoder(ctx, pt)(aos, theHailClassLoader)
        en.writeRegionValue(offset)
        en.flush()

        region2.clear()
        val ais2 = new ByteArrayInputStream(aos.toByteArray)
        val (retPType2: PStruct, dec2) = codec.buildDecoder(ctx, t)
        val offset2 = dec2(ais2, theHailClassLoader).readRegionValue(region2)
        val ur2 = new UnsafeRow(retPType2, region2, offset2)
        assert(t.typeCheck(ur2))
        assert(t.valuesSimilar(a, ur2))

        region3.clear()
        val ais3 = new ByteArrayInputStream(aos.toByteArray)
        val (retPType3: PStruct, dec3) = codec.buildDecoder(ctx, requestedType)
        val offset3 = dec3(ais3, theHailClassLoader).readRegionValue(region3)
        val ur3 = new UnsafeRow(retPType3, region3, offset3)
        assert(requestedType.typeCheck(ur3))
        assert(requestedType.valuesSimilar(a2, ur3))

        val codec2 = TypedCodecSpec(PType.canonical(requestedType), bufferSpec)
        val aos2 = new ByteArrayOutputStream()
        val en2 = codec2.buildEncoder(ctx, pt)(aos2, theHailClassLoader)
        en2.writeRegionValue(offset)
        en2.flush()

        region4.clear()
        val ais4 = new ByteArrayInputStream(aos2.toByteArray)
        val (retPType4: PStruct, dec4) = codec2.buildDecoder(ctx, requestedType)
        val offset4 = dec4(ais4, theHailClassLoader).readRegionValue(region4)
        val ur4 = new UnsafeRow(retPType4, region4, offset4)
        assert(requestedType.typeCheck(ur4))
        if (!requestedType.valuesSimilar(a2, ur4)) {
          println(t)
          println(requestedType)
          println(a2)
          println(ur4)
        }
        assert(requestedType.valuesSimilar(a2, ur4))
      }

      true
    }
    p.check()
  }

  @Test def testCodecForNonWrappedTypes(): Unit = {
    val valuesAndTypes = FastSeq(
      5 -> PInt32(),
      6L -> PInt64(),
      5.5f -> PFloat32(),
      5.7d -> PFloat64(),
      "foo" -> PCanonicalString(),
      Array[Byte](61, 62, 63) -> PCanonicalBinary(),
      FastSeq[Int](1, 2, 3) -> PCanonicalArray(PInt32()),
    )

    valuesAndTypes.foreach { case (v, t) =>
      pool.scopedRegion { region =>
        val off = ScalaToRegionValue(sm, region, t, v)
        BufferSpec.specs.foreach { spec =>
          val cs2 = TypedCodecSpec(t, spec)
          val baos = new ByteArrayOutputStream()
          val enc = cs2.buildEncoder(ctx, t)(baos, theHailClassLoader)
          enc.writeRegionValue(off)
          enc.flush()

          val serialized = baos.toByteArray
          val (decT, dec) = cs2.buildDecoder(ctx, t.virtualType)
          assert(decT == t)
          val res =
            dec((new ByteArrayInputStream(serialized)), theHailClassLoader).readRegionValue(region)

          assert(t.unsafeOrdering(sm).equiv(res, off))
        }
      }
    }
  }

  @Test def testBufferWriteReadDoubles(): Unit = {
    val a = Array(1.0, -349.273, 0.0, 9925.467, 0.001)

    BufferSpec.specs.foreach { bufferSpec =>
      val out = new ByteArrayOutputStream()
      val outputBuffer = bufferSpec.buildOutputBuffer(out)
      outputBuffer.writeDoubles(a)
      outputBuffer.flush()

      val in = new ByteArrayInputStream(out.toByteArray)
      val inputBuffer = bufferSpec.buildInputBuffer(in)
      val a2 = new Array[Double](5)
      inputBuffer.readDoubles(a2)

      assert(a sameElements a2)
    }
  }

  @Test def testRegionValue(): Unit = {
    val region = Region(pool = pool)
    val region2 = Region(pool = pool)
    val rvb = new RegionValueBuilder(sm, region)
    val rvb2 = new RegionValueBuilder(sm, region2)

    val g = Type.genArb
      .flatMap(t => Gen.zip(Gen.const(t), t.genValue(sm), Gen.choose(0, 100), Gen.choose(0, 100)))
      .filter { case (_, a, _, _) => a != null }
    val p = Prop.forAll(g) { case (t, a, n, n2) =>
      val pt = PType.canonical(t)
      t.typeCheck(a)

      // test addAnnotation
      region.clear()
      region.allocate(1, n) // preallocate

      val offset = pt.unstagedStoreJavaObject(sm, a, region)

      val ur = UnsafeRow.read(pt, region, offset)
      assert(t.valuesSimilar(a, ur), s"$a vs $ur")

      // test visitor
      val rv = RegionValue(region, offset)
      rv.pretty(pt)

      // test addAnnotation from ur
      region2.clear()
      region2.allocate(1, n2) // preallocate
      val offset2 = pt.unstagedStoreJavaObject(sm, ur, region2)

      val ur2 = UnsafeRow.read(pt, region2, offset2)
      assert(t.valuesSimilar(a, ur2), s"$a vs $ur2")

      // test addRegionValue
      region2.clear()
      region2.allocate(1, n2) // preallocate
      rvb2.start(pt)
      rvb2.addRegionValue(pt, region, offset)
      val offset3 = rvb2.end()
      val ur3 = UnsafeRow.read(pt, region2, offset3)
      assert(t.valuesSimilar(a, ur3), s"$a vs $ur3")

      // test addRegionValue nested
      t match {
        case t: TStruct =>
          val ps = pt.asInstanceOf[PStruct]
          region2.clear()
          region2.allocate(1, n) // preallocate
          val offset4 =
            ps.unstagedStoreJavaObject(sm, Row.fromSeq(a.asInstanceOf[Row].toSeq), region2)
          val ur4 = new UnsafeRow(ps, region2, offset4)
          assert(t.valuesSimilar(a, ur4))
        case _ =>
      }

      // test addRegionValue to same region
      rvb.start(pt)
      rvb.addRegionValue(pt, region, offset)
      val offset5 = rvb.end()
      val ur5 = UnsafeRow.read(pt, region, offset5)
      assert(t.valuesSimilar(a, ur5))

      // test addRegionValue to same region nested
      t match {
        case t: TStruct =>
          val ps = pt.asInstanceOf[PStruct]
          val offset6 =
            ps.unstagedStoreJavaObject(sm, Row.fromSeq(a.asInstanceOf[Row].toSeq), region)
          val ur6 = new UnsafeRow(ps, region, offset6)
          assert(t.valuesSimilar(a, ur6))
        case _ =>
      }

      true
    }
    p.check()
  }

  val g = (for {
    s <- Gen.size
    // prefer smaller type and bigger values
    fraction <- Gen.choose(0.1, 0.3)
    x = (fraction * s).toInt
    y = s - x
    t <- Type.genStruct.resize(x)
    v <- t.genNonmissingValue(sm).resize(y)
  } yield (t, v)).filter(_._2 != null)

  @Test def testPacking(): Unit = {

    def makeStruct(types: PType*): PCanonicalStruct =
      PCanonicalStruct(types.zipWithIndex.map { case (t, i) => (s"f$i", t) }: _*)

    val t1 = makeStruct( // missing byte is 0
      PInt32(), // 4-8
      PInt32(), // 8-12
      PFloat64(), // 16-24
      PBoolean(), // 1-2
      PBoolean(), // 2-3
      PBoolean(), // 3-4
      PBoolean(), // 12-13
      PBoolean(),
    ) // 13-14
    assert(t1.byteOffsets.toSeq == Seq(4, 8, 16, 1, 2, 3, 12, 13))
    assert(t1.byteSize == 24)

    val t2 = makeStruct( // missing bytes 0, 1
      PBoolean(), // 2-3
      PInt32(), // 4-8
      PInt32(), // 8-12
      PFloat64(), // 16-24
      PInt32(), // 12-16
      PInt32(), // 24-28
      PFloat64(), // 32-40
      PInt32(), // 28-32
      PBoolean(), // 3-4
      PFloat64(), // 40-48
      PBoolean(),
    ) // 48-49

    assert(t2.byteOffsets.toSeq == Seq(2, 4, 8, 16, 12, 24, 32, 28, 3, 40, 48))
    assert(t2.byteSize == 49)

    val t3 = makeStruct((0 until 512).map(_ => PFloat64()): _*)
    assert(t3.byteSize == (512 / 8) + 512 * 8)
    val t4 = makeStruct((0 until 256).flatMap(_ =>
      Iterator(PInt32(), PInt32(), PFloat64(), PBoolean())
    ): _*)
    assert(t4.byteSize == 256 * 4 / 8 + 256 * 4 * 2 + 256 * 8 + 256)
  }

  @Test def testEmptySize(): Unit =
    assert(PCanonicalStruct().byteSize == 0)

  @Test def testUnsafeOrdering(): Unit = {
    val region = Region(pool = pool)
    val region2 = Region(pool = pool)

    val g = PType.genStruct
      .flatMap(t => Gen.zip(Gen.const(t), Gen.zip(t.genValue(sm), t.genValue(sm))))
      .filter { case (_, (a1, a2)) => a1 != null && a2 != null }
      .resize(10)
    val p = Prop.forAll(g) { case (t, (a1, a2)) =>
      val tv = t.virtualType

      tv.typeCheck(a1)
      tv.typeCheck(a2)

      region.clear()
      val offset = t.unstagedStoreJavaObject(sm, a1, region)

      val ur1 = new UnsafeRow(t, region, offset)
      assert(tv.valuesSimilar(a1, ur1))

      region2.clear()
      val offset2 = t.unstagedStoreJavaObject(sm, a2, region2)

      val ur2 = new UnsafeRow(t, region2, offset2)
      assert(tv.valuesSimilar(a2, ur2))

      val ord = tv.ordering(sm)
      val uord = t.unsafeOrdering(sm)

      val c1 = ord.compare(a1, a2)
      val c2 = ord.compare(ur1, ur2)
      val c3 = uord.compare(ur1.offset, ur2.offset)

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
}
