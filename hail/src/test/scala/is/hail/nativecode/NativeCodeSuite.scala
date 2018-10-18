package is.hail.nativecode

import java.io.ByteArrayOutputStream

import is.hail.SparkSuite
import is.hail.annotations._
import is.hail.check.Gen
import is.hail.check.Prop.forAll
import is.hail.cxx._
import is.hail.expr._
import is.hail.expr.types._
import is.hail.io.{BlockingOutputBuffer, LEB128OutputBuffer, StreamBlockOutputBuffer}
import is.hail.nativecode._
import org.apache.spark.SparkException
import org.testng.annotations.Test
import is.hail.utils._
import is.hail.testUtils._

class NativeCodeSuite extends SparkSuite {

  @Test def testNativePtr() = {
    var a = new NativeStatus()
    assert(a.ok)
    assert(a.use_count() == 1)
    var b = new NativeStatus()
    b.copyAssign(a)
    assert(b.get() == a.get())
    assert(b.use_count() == 2)
    assert(a.use_count() == 2)
    var c = new NativeStatus()
    c.moveAssign(b)
    assert(c.get() == a.get())
    assert(b.get() == 0)
    assert(c.use_count() == 2)
    c.close();
    assert(c.get() == 0)
    assert(a.use_count() == 1)
    c.close()
    assert(a.use_count() == 1)
    var d = new NativeStatus()
    d.copyAssign(a)
    assert(d.get() == a.get())
    assert(a.use_count() == 2)
    var e = new NativeStatus()
    e.copyAssign(a)
    assert(a.use_count() == 3)
    e.copyAssign(d)
    assert(a.use_count() == 3)
    e.moveAssign(d)
    assert(d.get() == 0)
    assert(a.use_count() == 2)
    e.close()
    assert(e.get() == 0)
    assert(a.use_count() == 1)
  }

  @Test def testNativeGlobal() = {
    val st = new NativeStatus()
    val globalModule = new NativeModule("global")
    val funcHash1 = globalModule.findLongFuncL1(st, "hailTestHash1")
    assert(st.ok, st.toString())
    val funcHash8 = globalModule.findLongFuncL8(st, "hailTestHash8")
    assert(st.ok, st.toString())
    val ret = funcHash8(st, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8)
    assert(ret == 0x87654321L)
    st.close()
    globalModule.close()
    funcHash1.close()
    funcHash8.close() 
  }

  @Test def testNativeCallSpeed() = {
    val st = new NativeStatus()
    val globalModule = new NativeModule("global")
    val funcHash1 = globalModule.findLongFuncL1(st, "hailTestHash1")
    assert(st.ok, st.toString())
    val t0 = System.currentTimeMillis()
    var sum: Long = 0
    val numCalls = 100*1000000
    var countdown = numCalls
    while (countdown > 0) {
      sum = funcHash1(st, sum)
      countdown -= 1
    }
    val t1 = System.currentTimeMillis()
    val usecsPerJniCall = ((t1 - t0) * 1000.0) / numCalls
    assert(usecsPerJniCall < 0.2)
    st.close()
    globalModule.close()
    funcHash1.close()
  }

  @Test def testNativeBuild() = {
    val sb = new StringBuilder()
    sb.append(
    """#include "hail/hail.h"
      |
      |NAMESPACE_HAIL_MODULE_BEGIN
      |
      |long testFunc1(NativeStatus* st, long a0) { return a0+1; }
      |
      |class MyObj : public NativeObj {
      | public:
      |  int val_;
      |
      |  MyObj(int val) : val_(val) { }
      |  ~MyObj() { }
      |  const char* get_class_name() { return "MyObj"; }
      |};
      |
      |NativeObjPtr makeMyObj(NativeStatus*, long val) {
      |  return std::make_shared<MyObj>(val);
      |}
      |
      |NAMESPACE_HAIL_MODULE_END
      |""".stripMargin
    )
    val options = "-ggdb -O2"
    val st = new NativeStatus()
    val mod = new NativeModule(options, sb.toString())
    mod.findOrBuild(st)
    assert(st.ok, st.toString())
    val testFunc1 = mod.findLongFuncL1(st, "testFunc1")
    assert(st.ok, st.toString())
    assert(testFunc1(st, 6) == 7);
    testFunc1.close()
    val makeMyObj = mod.findPtrFuncL1(st, "makeMyObj")
    assert(st.ok, st.toString())
    val myObj = new NativePtr(makeMyObj, st, 55L)
    assert(myObj.get() != 0)
    // Now try getting the binary
    val key = mod.getKey
    val binary = mod.getBinary
    mod.close()
    val workerMod = new NativeModule(key, binary)
    val workerFunc1 = workerMod.findLongFuncL1(st, "testFunc1")
    assert(st.ok, st.toString())
    workerFunc1.close()
    workerMod.close()
    st.close()
  }
  
  @Test def testNativeUpcall() = {
    val sb = new StringBuilder()
    sb.append(
    """#include "hail/hail.h"
      |
      |NAMESPACE_HAIL_MODULE_BEGIN
      |
      |long testUpcall(NativeStatus* st, long a0) {
      |  set_test_msg("Hello!");
      |  return 1000+a0;
      |}
      |
      |NAMESPACE_HAIL_MODULE_END
      |""".stripMargin
    )
    val st = new NativeStatus()
    val mod = new NativeModule("", sb.toString())
    val testUpcall = mod.findLongFuncL1(st, "testUpcall")
    mod.close()
    assert(st.ok, st.toString())
    Upcalls.testMsg = "InitialValueOfTestMsg";
    assert(testUpcall(st, 99) == 1099)
    assert(Upcalls.testMsg.equals("Hello!"))
    st.close()
    testUpcall.close()
  }
  
  @Test def testObjectArray() = {
    class MyObject(num: Long) {
      def plus(n: Long) = num+n
    }

    val sb = new StringBuilder()
    sb.append(
    """#include "hail/hail.h"
      |#include "hail/ObjectArray.h"
      |#include "hail/Upcalls.h"
      |#include <cstdio>
      |
      |NAMESPACE_HAIL_MODULE_BEGIN
      |
      |class ObjectHolder : public NativeObj {
      | public:
      |  ObjectArrayPtr objects_;
      |
      |  ObjectHolder(ObjectArray* objects) :
      |    objects_(std::dynamic_pointer_cast<ObjectArray>(objects->shared_from_this())) {
      |  }
      |};
      |
      |NativeObjPtr makeObjectHolder(NativeStatus*, long objects) {
      |  return std::make_shared<ObjectHolder>(reinterpret_cast<ObjectArray*>(objects));
      |}
      |
      |long testPlus(NativeStatus* st, long holder, long idx, long val) {
      |  UpcallEnv up;
      |  JNIEnv* env = up.env();
      |  auto h = reinterpret_cast<ObjectHolder*>(holder);
      |  auto obj = h->objects_->at(idx);
      |  auto cl = env->GetObjectClass(obj);
      |  auto plus_method = env->GetMethodID(cl, "plus", "(J)J");
      |  return env->CallLongMethod(obj, plus_method, val);
      |}
      |
      |NAMESPACE_HAIL_MODULE_END
      |""".stripMargin
    )
    val st = new NativeStatus()
    val mod = new NativeModule("", sb.toString())
    val makeObjectHolder = mod.findPtrFuncL1(st, "makeObjectHolder")
    assert(st.ok, st.toString())
    val testPlus = mod.findLongFuncL3(st, "testPlus")
    assert(st.ok, st.toString())
    mod.close()
    val objArray = new ObjectArray(new MyObject(1000), new MyObject(2000))
    val holder = new NativePtr(makeObjectHolder, st, objArray.get())
    objArray.close()
    assert(testPlus(st, holder.get(), 0, 44) == 1044)
    assert(testPlus(st, holder.get(), 1, 55) == 2055)
    testPlus.close()
  }

  @Test def testCXXCodeFunctions: Unit = {
    val tub= new TranslationUnitBuilder()
    tub.include("hail/hail.h")
    tub.include("<cstdio>")

    val fb = FunctionBuilder("testUpcall", Array("NativeStatus*" -> "st", "long" -> "a0"), "long")

    fb += Statement(
      s"""
         |set_test_msg("Hello!");
         |return 1000+${fb.getArg(1)}
       """.stripMargin)

    val f = fb.result()
    tub += f

    val mod = tub.result().build("")

    val st = new NativeStatus()
    val testUpcall = mod.findLongFuncL1(st, f.name)
    mod.close()
    assert(st.ok, st.toString())
    Upcalls.testMsg = "InitialValueOfTestMsg";
    assert(testUpcall(st, 99) == 1099)
    assert(Upcalls.testMsg.equals("Hello!"))
    st.close()
    testUpcall.close()
  }

  @Test def testCXXOutputStream: Unit = {
    val tub = new TranslationUnitBuilder()
    tub.include("hail/hail.h")
    tub.include("hail/Encoder.h")
    tub.include("hail/ObjectArray.h")
    tub.include("<cstdio>")

    val makeHolderF = FunctionBuilder("makeObjectHolder", Array("NativeStatus*" -> "st", "long" -> "objects"), "NativeObjPtr")

    makeHolderF += Statement(s"return std::make_shared<ObjectHolder>(reinterpret_cast<ObjectArray*>(${makeHolderF.getArg(1)}))")
    val holderF = makeHolderF.result()
    tub += holderF

    val fb = FunctionBuilder("testOutputStream", Array("NativeStatus*" -> "st", "long" -> "holder"), "long")

    fb += Statement(
      s"""
         |UpcallEnv up;
         |auto h = reinterpret_cast<ObjectHolder*>(${fb.getArg(1)});
         |auto jos = h->objects_->at(0);
         |
         |char * buf = new char[10]{97, 98, 99, 100, 101, 102, 103, 104, 105, 106};
         |
         |auto os = OutputStream(up, jos);
         |os.write(buf, 10);
         |
         |return 0;
       """.stripMargin)

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

    val makeHolderF = FunctionBuilder("makeObjectHolder", Array("NativeStatus*" -> "st", "long" -> "objects"), "NativeObjPtr")

    makeHolderF += Statement(s"return std::make_shared<ObjectHolder>(reinterpret_cast<ObjectArray*>(${makeHolderF.getArg(1)}))")
    val holderF = makeHolderF.result()
    tub += holderF

    val fb = FunctionBuilder("testOutputBuffers", Array("NativeStatus*" -> "st", "long" -> "holder"), "long")

    val bytes = Array.tabulate[Byte](100)(i => new Integer(i + 97).byteValue())

    fb += Statement(
      s"""
         |UpcallEnv up;
         |auto h = reinterpret_cast<ObjectHolder*>(${fb.getArg(1)});
         |auto jos = h->objects_->at(0);
         |
         |char * buf = new char[10]{97, 98, 99, 100, 101, 102, 103, 104, 105, 106};
         |
         |auto os = OutputStream(up, jos);
         |auto stream_buf = StreamOutputBlockBuffer(os);
         |auto blocking_buf = BlockingOutputBuffer(32, &stream_buf);
         |auto leb_buf = LEB128OutputBuffer(&blocking_buf);
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
       """.stripMargin)

    val f = fb.result()
    tub += f

    val mod = tub.result().build("")

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
      val ob = new LEB128OutputBuffer(new BlockingOutputBuffer(32, new StreamBlockOutputBuffer(expected)))
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

//    println(compiled.toByteArray.map(b => String.format("%02x", new Integer(b.toInt))).mkString(", "))
//    println(expected.toByteArray.map(b => String.format("%02x", new Integer(b.toInt))).mkString(", "))
    assert(compiled.toByteArray sameElements expected.toByteArray)
  }

}
