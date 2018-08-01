package is.hail.nativecode

import is.hail.SparkSuite
import is.hail.annotations._
import is.hail.check.Gen
import is.hail.check.Prop.forAll
import is.hail.expr._
import is.hail.expr.types._
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
    val mod = new NativeModule(options, sb.toString(), true)
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
    val key = mod.getKey()
    val binary = mod.getBinary()
    mod.close()
    val workerMod = new NativeModule(key, binary)
    val workerFunc1 = workerMod.findLongFuncL1(st, "testFunc1")
    assert(st.ok, st.toString())
    workerFunc1.close()
    workerMod.close()
    st.close()
  }

}
