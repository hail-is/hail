package is.hail.nativecode

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import is.hail.{HailSuite, variant}
import is.hail.annotations._
import is.hail.cxx._
import is.hail.expr.Parser
import is.hail.expr.types._
import is.hail.expr.types.virtual._
import is.hail.io._
import is.hail.io.compress.LZ4Utils
import org.testng.annotations.Test
import is.hail.utils._
import is.hail.variant.{Locus, ReferenceGenome}
import org.apache.spark.sql.Row

class NativeDecoderSuite extends HailSuite {

  @Test def testCXXInputStream(): Unit = {
    val tub = new TranslationUnitBuilder()
    tub.include("hail/hail.h")
    tub.include("hail/Decoder.h")
    tub.include("hail/Upcalls.h")
    tub.include("hail/ObjectArray.h")
    tub.include("<cstdio>")
    tub.include("<memory>")

    val fb = tub.buildFunction("testInputStream", Array("NativeStatus*" -> "st", "long" -> "array"), "long")

    fb +=
      s"""UpcallEnv up;
         |auto h = reinterpret_cast<ObjectArray*>(${ fb.getArg(1) });
         |auto jis = h->at(0);
         |
         |char buf[15];
         |
         |auto is = std::make_shared<InputStream>(up, jis);
         |is->read(buf, 15);
         |
         |long equals = 1;
         |for (int i = 0; i < 15; i++) {
         |  if (buf[i] != 97 + i) {
         |    equals = 0;
         |  }
         |}
         |
         |return equals;""".stripMargin

    val f = fb.end()

    val mod = tub.end().build("")

    val st = new NativeStatus()
    val testIS = mod.findLongFuncL1(st, f.name)
    assert(st.ok, st.toString())
    mod.close()

    val bais = new ByteArrayInputStream("abcdefghijklmno".getBytes)
    val objArray = new ObjectArray(bais)

    assert(testIS(st, objArray.get()) == 1)
    objArray.close()
    testIS.close()
  }

  @Test def testInputBuffers(): Unit = {
    CodecSpec.bufferSpecs.foreach { spec =>
      val tub = new TranslationUnitBuilder()
      tub.include("hail/hail.h")
      tub.include("hail/Decoder.h")
      tub.include("hail/ObjectArray.h")
      tub.include("<cstdio>")
      tub.include("<memory>")

      val fb = tub.buildFunction("testInputBuffers", Array("NativeStatus*" -> "st", "long" -> "holder"), "long")

      fb +=
        s"""
           |UpcallEnv up;
           |auto h = reinterpret_cast<ObjectArray*>(${ fb.getArg(1) });
           |auto jis = h->at(0);
           |
           |auto is = std::make_shared<InputStream>(up, jis);
           |${ spec.nativeInputBufferType("InputStream") } leb_buf {is};
           |
           |leb_buf.skip_boolean();
           |if (leb_buf.read_boolean() != true) { return 0; }
           |leb_buf.skip_byte();
           |if (leb_buf.read_byte() != 3) { return 0; }
           |leb_buf.skip_int();
           |if (leb_buf.read_int() != 3) { return 0; }
           |leb_buf.skip_long();
           |if (leb_buf.read_long() != 500) { return 0; }
           |leb_buf.skip_float();
           |if (leb_buf.read_float() != 5.5) { return 0; }
           |leb_buf.skip_double();
           |if (leb_buf.read_double() != 5.5) { return 0; }
           |leb_buf.skip_bytes(3);
           |char b[5];
           |leb_buf.read_bytes(b, 5);
           |for (int i = 0; i < 5; i++) {
           |  if (b[i] != 100 + i) {
           |    return 0;
           |  }
           |}
           |return 1;
       """.stripMargin

      val f = fb.end()

      val mod = tub.end().build("-O1 -llz4")

      val data = new ByteArrayOutputStream()
      val ob = spec.buildOutputBuffer(data)
      Region.scoped { region =>
        ob.writeBoolean(false)
        ob.writeBoolean(true)
        ob.writeByte(1)
        ob.writeByte(3)
        ob.writeInt(500)
        ob.writeInt(3)
        ob.writeLong(3)
        ob.writeLong(500)
        ob.writeFloat(3.3f)
        ob.writeFloat(5.5f)
        ob.writeDouble(3.3)
        ob.writeDouble(5.5)
        val b1 = region.allocate(1, 3)
        region.storeBytes(b1, Array[Byte](97, 98, 99))
        ob.writeBytes(region, b1, 3)
        val b2 = region.allocate(1, 5)
        region.storeBytes(b2, Array[Byte](100, 101, 102, 103, 104))
        ob.writeBytes(region, b2, 5)
        ob.flush()
      }

      val bais = new ByteArrayInputStream(data.toByteArray)
      val objArray = new ObjectArray(bais)

      val st = new NativeStatus()
      val testIB = mod.findLongFuncL1(st, f.name)
      assert(st.ok, st.toString())
      mod.close()
      assert(testIB(st, objArray.get()) == 1)
      objArray.close()
      testIB.close()
    }
  }

  @Test def testDecoder(): Unit = {
    val spec = new LEB128BufferSpec(
      new BlockingBufferSpec(32,
        new LZ4BlockBufferSpec(32,
          new StreamBlockBufferSpec)))

    val t = TStruct(
      "a" -> +TArray(+TTuple(TInt32(), TInt32(), TInt32(), TBoolean())),
      "b" -> TString(),
      "c" -> TTuple(+TString(), +TInt32()))
    val rt = TStruct(
      "a" -> +TArray(+TTuple(TInt32(), TInt32(), TInt32(), TBoolean())),
      "c" -> TTuple(+TString(), +TInt32()))
    val a = Row(FastIndexedSeq(Row(5, 4, null, true), Row(7, null, null, null)), "asfdghsjsdfgsfgasdfasdgadfg", Row("foo", 1))
    val requested = Row(a.get(0), a.get(2))

    val baos = new ByteArrayOutputStream()
    val enc = new PackEncoder(t.physicalType, spec.buildOutputBuffer(baos))

    Region.scoped { region =>
      val rvb = new RegionValueBuilder(region)
      rvb.start(t.physicalType)
      rvb.addAnnotation(t, a)
      val off = rvb.end()
      enc.writeRegionValue(region, off)
      enc.flush()
    }

    val bais = new ByteArrayInputStream(baos.toByteArray)
    val dec = new NativePackDecoder(bais, PackDecoder.buildModule(t.physicalType, t.physicalType, spec))
    val bais2 = new ByteArrayInputStream(baos.toByteArray)
    val dec2 = new NativePackDecoder(bais2, PackDecoder.buildModule(t.physicalType, rt.physicalType, spec))

    Region.scoped { region =>
      val off = dec.readRegionValue(region)
      val off2 = dec2.readRegionValue(region)
      assert(SafeRow.read(t.physicalType, region, off) == a)
      assert(SafeRow.read(rt.physicalType, region, off2) == requested)
    }
  }
}
