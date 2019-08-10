package is.hail.nativecode

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import is.hail.HailSuite
import is.hail.annotations._
import is.hail.cxx._
import is.hail.expr.types.virtual.{TInt32, TInterval, TSet, TStruct, _}
import is.hail.io._
import is.hail.io.compress.LZ4Utils
import is.hail.utils._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class NativeEncoderSuite extends HailSuite {

  @Test def testCXXOutputStream(): Unit = {
    val tub = new TranslationUnitBuilder()
    tub.include("hail/hail.h")
    tub.include("hail/Encoder.h")
    tub.include("hail/Upcalls.h")
    tub.include("hail/ObjectArray.h")
    tub.include("<cstdio>")
    tub.include("<memory>")

    val fb = tub.buildFunction("testOutputStream", Array("NativeStatus*" -> "st", "long" -> "array"), "long")

    fb +=
      s"""UpcallEnv up;
         |auto h = reinterpret_cast<ObjectArray*>(${ fb.getArg(1) });
         |auto jos = h->at(0);
         |
         |char * buf = new char[10]{97, 98, 99, 100, 101, 102, 103, 104, 105, 106};
         |
         |auto os = std::make_shared<OutputStream>(up, jos);
         |os->write(buf, 10);
         |
         |return 0;""".stripMargin

    val f = fb.end()

    val mod = tub.end().build("")

    val st = new NativeStatus()
    val testOS = mod.findLongFuncL1(st, f.name)
    assert(st.ok, st.toString())
    mod.close()

    val baos = new ByteArrayOutputStream()
    val objArray = new ObjectArray(baos)

    assert(testOS(st, objArray.get()) == 0)
    objArray.close()
    baos.flush()
    assert(new String(baos.toByteArray) == "abcdefghij")
    testOS.close()
  }

  @Test def testOutputBuffers() {
    CodecSpec.bufferSpecs.foreach { spec =>
      val tub = new TranslationUnitBuilder()
      tub.include("hail/hail.h")
      tub.include("hail/Encoder.h")
      tub.include("hail/ObjectArray.h")
      tub.include("<cstdio>")
      tub.include("<memory>")

      val fb = tub.buildFunction("testOutputBuffers", Array("NativeStatus*" -> "st", "long" -> "holder"), "long")

      val bytes = Array.tabulate[Byte](100)(i => new Integer(i + 97).byteValue())

      fb +=
        s"""
           |UpcallEnv up;
           |auto h = reinterpret_cast<ObjectArray*>(${ fb.getArg(1) });
           |auto jos = h->at(0);
           |
           |auto os = std::make_shared<OutputStream>(up, jos);
           |${ spec.nativeOutputBufferType } buf { os };
           |
           |buf.write_boolean(true);
           |buf.write_byte(3);
           |buf.write_int(3);
           |buf.write_long(3);
           |buf.write_float(3.3f);
           |buf.write_double(3.3);
           |buf.write_bytes(new char[${ bytes.length }] {${ bytes.mkString(", ") }}, ${ bytes.length });
           |buf.flush();
           |buf.close();
           |
           |return 0;
       """.stripMargin

      val f = fb.end()

      val mod = tub.end().build("-O1 -llz4")

      val st = new NativeStatus()
      val testOB = mod.findLongFuncL1(st, f.name)
      assert(st.ok, st.toString())
      mod.close()

      val compiled = new ByteArrayOutputStream()
      val objArray = new ObjectArray(compiled)

      assert(testOB(st, objArray.get()) == 0)
      objArray.close()
      testOB.close()

      val expected = new ByteArrayOutputStream()
      val ob = spec.buildOutputBuffer(expected)
      Region.scoped { region =>
        ob.writeBoolean(true)
        ob.writeByte(3)
        ob.writeInt(3)
        ob.writeLong(3)
        ob.writeFloat(3.3f)
        ob.writeDouble (3.3)
        val off = region.allocate(bytes.length)
        region.storeBytes(off, bytes)
        ob.writeBytes(region, off, bytes.length)
        ob.flush()
      }

      val expectedDecoded = spec.buildInputBuffer(new ByteArrayInputStream(expected.toByteArray))
      val actualDecoded = spec.buildInputBuffer(new ByteArrayInputStream(compiled.toByteArray))
      assert(expectedDecoded.readBoolean() == actualDecoded.readBoolean())
      assert(expectedDecoded.readByte() == actualDecoded.readByte())
      assert(expectedDecoded.readInt() == actualDecoded.readInt())
      assert(expectedDecoded.readLong() == actualDecoded.readLong())
      assert(expectedDecoded.readFloat() == actualDecoded.readFloat())
      assert(expectedDecoded.readDouble() == actualDecoded.readDouble())
      bytes.foreach { b =>
        assert(expectedDecoded.readByte() == b)
        assert(actualDecoded.readByte() == b)
      }
    }
  }

  @Test def testEncoder(): Unit = {
    val spec = new LEB128BufferSpec(
      new BlockingBufferSpec(32,
        new LZ4BlockBufferSpec(32,
          new StreamBlockBufferSpec)))
    val t = TTuple(TInterval(TStruct("x" -> TSet(TInt32()))))

    val a = Row(Interval(Row(Set(-1478292367)), Row(Set(2084728308)), true, true))

    val baos = new ByteArrayOutputStream()
    val enc = new NativePackEncoder(baos, PackEncoder.buildModule(t.physicalType, spec))

    val baos2 = new ByteArrayOutputStream()
    val enc2 = new PackEncoder(t.physicalType, spec.buildOutputBuffer(baos2))

    Region.scoped { region =>
      val rvb = new RegionValueBuilder(region)
      rvb.start(t.physicalType)
      rvb.addAnnotation(t, a)
      val off = rvb.end()
      enc.writeRegionValue(region, off)
      enc.flush()
      enc2.writeRegionValue(region, off)
      enc2.flush()
    }

    val compiled = baos.toByteArray
    val expected = baos.toByteArray

    assert(compiled sameElements expected)
  }

}
