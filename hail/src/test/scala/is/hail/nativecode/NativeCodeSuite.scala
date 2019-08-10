package is.hail.nativecode

import is.hail.HailSuite
import is.hail.cxx._
import org.testng.annotations.Test

class NativeCodeSuite extends HailSuite {

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
      |#include "hail/Upcalls.h"
      |
      |NAMESPACE_HAIL_MODULE_BEGIN
      |
      |long testUpcall(NativeStatus* st, long a0) {
      |  UpcallEnv up {};
      |  up.set_test_msg("Hello!");
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

  @Test def testCXXCodeFunctions(): Unit = {
    val tub = new TranslationUnitBuilder()
    tub.include("hail/hail.h")
    tub.include("hail/Upcalls.h")
    tub.include("<cstdio>")

    val fb = tub.buildFunction("testUpcall", Array("NativeStatus*" -> "st", "long" -> "a0"), "long")

    fb +=
      s"""
         |UpcallEnv up {};
         |up.set_test_msg("Hello!");
         |return 1000+${fb.getArg(1)};
       """.stripMargin

    val f = fb.end()

    val mod = tub.end().build("")

    val st = new NativeStatus()
    val testUpcall = mod.findLongFuncL1(st, f.name)
    mod.close()
    assert(st.ok, st.toString())
    Upcalls.testMsg = "InitialValueOfTestMsg"
    assert(testUpcall(st, 99) == 1099)
    assert(Upcalls.testMsg.equals("Hello!"))
    st.close()
    testUpcall.close()
  }
}
