package is.hail.nativecode

import java.io.ByteArrayOutputStream

import is.hail.SparkSuite
import is.hail.annotations._
import is.hail.cxx._
import is.hail.expr.types._
import is.hail.io._
import org.testng.annotations.Test
import is.hail.utils._
import org.apache.spark.sql.Row

class NativeEncoderSuite extends SparkSuite {

  @Test def testCXXOutputStream: Unit = {
    val tub = new TranslationUnitBuilder()
    tub.include("hail/hail.h")
    tub.include("hail/Encoder.h")
    tub.include("hail/Upcalls.h")
    tub.include("hail/ObjectArray.h")
    tub.include("<cstdio>")
    tub.include("<memory>")

    val makeHolderF = FunctionBuilder("makeObjectHolder", Array("NativeStatus*" -> "st", "long" -> "objects"), "NativeObjPtr")

    makeHolderF += Statement(s"return std::make_shared<ObjectHolder>(reinterpret_cast<ObjectArray*>(${makeHolderF.getArg(1)}));")
    val holderF = makeHolderF.result()
    tub += holderF

    val fb = FunctionBuilder("testOutputStream", Array("NativeStatus*" -> "st", "long" -> "holder"), "long")

    fb += s"""UpcallEnv up;
             |auto h = reinterpret_cast<ObjectHolder*>(${fb.getArg(1)});
             |auto jos = h->objects_->at(0);
             |
             |char * buf = new char[10]{97, 98, 99, 100, 101, 102, 103, 104, 105, 106};
             |
             |auto os = std::make_shared<OutputStream>(up, jos);
             |os->write(buf, 10);
             |
             |return 0;""".stripMargin

    val f = fb.result()
    tub += f

    val mod = tub.result().build("")

    val st = new NativeStatus()
    val makeHolder = mod.findPtrFuncL1(st, holderF.name)
    assert(st.ok, st.toString())
    val testOS = mod.findLongFuncL1(st, f.name)
    assert(st.ok, st.toString())
    mod.close()

    val baos = new ByteArrayOutputStream()
    val objArray = new ObjectArray(baos)
    val holder = new NativePtr(makeHolder, st, objArray.get())
    objArray.close()
    makeHolder.close()

    assert(testOS(st, holder.get()) == 0)
    baos.flush()
    assert(new String(baos.toByteArray) == "abcdefghij")
    testOS.close()
  }

  @Test def testOutputBuffers: Unit = {
    val tub = new TranslationUnitBuilder()
    tub.include("hail/hail.h")
    tub.include("hail/Encoder.h")
    tub.include("hail/ObjectArray.h")
    tub.include("<cstdio>")
    tub.include("<memory>")

    val makeHolderF = FunctionBuilder("makeObjectHolder", Array("NativeStatus*" -> "st", "long" -> "objects"), "NativeObjPtr")

    makeHolderF += s"return std::make_shared<ObjectHolder>(reinterpret_cast<ObjectArray*>(${makeHolderF.getArg(1)}));"
    val holderF = makeHolderF.result()
    tub += holderF

    val fb = FunctionBuilder("testOutputBuffers", Array("NativeStatus*" -> "st", "long" -> "holder"), "long")

    val bytes = Array.tabulate[Byte](100)(i => new Integer(i + 97).byteValue())

    fb +=
      s"""
         |UpcallEnv up;
         |auto h = reinterpret_cast<ObjectHolder*>(${fb.getArg(1)});
         |auto jos = h->objects_->at(0);
         |
         |auto os = std::make_shared<OutputStream>(up, jos);
         |auto stream_buf = std::make_shared<StreamOutputBlockBuffer>(os);
         |using LZ4Buf = LZ4OutputBlockBuffer<32, StreamOutputBlockBuffer>;
         |auto lz4_buf = std::make_shared<LZ4Buf>(stream_buf);
         |using BlockBuf = BlockingOutputBuffer<32, LZ4Buf>;
         |auto blocking_buf = std::make_shared<BlockBuf>(lz4_buf);
         |auto leb_buf = LEB128OutputBuffer<BlockBuf>(blocking_buf);
         |
         |leb_buf.write_boolean(true);
         |leb_buf.write_byte(3);
         |leb_buf.write_int(3);
         |leb_buf.write_long(3l);
         |leb_buf.write_float(3.3f);
         |leb_buf.write_double(3.3);
         |leb_buf.write_bytes(new char[${bytes.length}] {${bytes.mkString(", ")}}, ${bytes.length});
         |leb_buf.flush();
         |leb_buf.close();
         |
         |return 0;
       """.stripMargin

    val f = fb.result()
    tub += f

    val mod = tub.result().build("-O1 -llz4")

    val st = new NativeStatus()
    val makeHolder = mod.findPtrFuncL1(st, holderF.name)
    assert(st.ok, st.toString())
    val testOB = mod.findLongFuncL1(st, f.name)
    assert(st.ok, st.toString())
    mod.close()

    val compiled = new ByteArrayOutputStream()
    val objArray = new ObjectArray(compiled)
    val holder = new NativePtr(makeHolder, st, objArray.get())
    objArray.close()
    makeHolder.close()

    assert(testOB(st, holder.get()) == 0)
    testOB.close()

    val expected = new ByteArrayOutputStream()
    Region.scoped { region =>
      val ob = new LEB128OutputBuffer(
        new BlockingOutputBuffer(32,
          new LZ4OutputBlockBuffer(32,
            new StreamBlockOutputBuffer(expected))))
      ob.writeBoolean(true)
      ob.writeByte(3)
      ob.writeInt(3)
      ob.writeLong(3)
      ob.writeFloat(3.3f)
      ob.writeDouble(3.3)
      val off = region.allocate(bytes.length)
      region.storeBytes(off, bytes)
      ob.writeBytes(region, off, bytes.length)
      ob.flush()
    }

    assert(compiled.toByteArray sameElements expected.toByteArray)
  }

  @Test def testEncoder() {
    val spec = new LEB128BufferSpec(
      new BlockingBufferSpec(32,
        new LZ4BlockBufferSpec(32,
          new StreamBlockBufferSpec)))
    val t = TTuple(TInterval(TStruct("x"->TSet(TInt32()))))

    val a = Row(Interval(Row(Set(-1478292367)), Row(Set(2084728308)), true, true))

    val baos = new ByteArrayOutputStream()
    val enc = new NativePackEncoder(baos, NativeEncoder(t.physicalType, spec))

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
