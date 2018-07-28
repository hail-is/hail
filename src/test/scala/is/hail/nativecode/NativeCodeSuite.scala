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
    var ret: Long = -1
    val st = new NativeStatus()
    val globalModule = new NativeModule("global")
    val funcHash1 = globalModule.findLongFuncL1(st, "hailTestHash1")
    if (st.fail) error(s"${st}")
    assert(st.ok)
    val funcHash8 = globalModule.findLongFuncL8(st, "hailTestHash8")
    if (st.fail) error(s"${st}")
    assert(st.ok)
    ret = funcHash8(st, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8)
    assert(ret == 0x87654321L)
    val t0 = System.currentTimeMillis()
    var sum: Long = 0
    val numCalls = 100*1000000
    var countdown = numCalls
    while (countdown > 0) {
      sum = funcHash1(st, sum)
      countdown -= 1
    }
    val t1 = System.currentTimeMillis()
    val usecsPerCall = ((t1 - t0) * 1000.0) / numCalls
    info(s"funcHash1() ~ ${usecsPerCall}usecs")
    assert(usecsPerCall < 0.2)
  }

  @Test def testNativeBuild() = {
    val sb = new StringBuilder()
    sb.append("#include \"hail/hail.h\"\n")
    sb.append("NAMESPACE_HAIL_MODULE_BEGIN\n")
    sb.append("\n")
    // A very simple function
    sb.append("long testFunc1(NativeStatus* st, long a0) { return a0+1; }\n\n")
    // Now declare our own NativeObj
    sb.append("class MyObj : public NativeObj {\n")
    sb.append("public:\n")
    sb.append("  int val_;\n")
    sb.append("\n")
    sb.append("  MyObj(int val) : val_(val) { }\n")
    sb.append("  ~MyObj() { }\n")
    sb.append("  const char* get_class_name() { return \"MyObj\"; }\n")
    sb.append("};\n")
    sb.append("\n")
    sb.append("NativeObjPtr makeMyObj(NativeStatus*, long val) {\n")
    sb.append("  return std::make_shared<MyObj>(val);\n")
    sb.append("}\n")
    sb.append("\n")
    sb.append("NAMESPACE_HAIL_MODULE_END\n")
    val options = "-ggdb -O2"
    val st = new NativeStatus()
    val mod = new NativeModule(options, sb.toString(), true)
    mod.findOrBuild(st)
    if (st.fail) error(s"${st}")
    assert(st.ok)
    val testFunc1 = mod.findLongFuncL1(st, "testFunc1")
    if (st.fail) error(s"${st}")
    assert(st.ok)
    val ret = testFunc1(st, 6)
    info(s"testFunc(6) returns ${ret}")
    testFunc1.close()
    val makeMyObj = mod.findPtrFuncL1(st, "makeMyObj")
    if (st.fail) error(s"${st}")
    assert(st.ok)
    val myObj = new NativePtr(makeMyObj, st, 55L)
    assert(myObj.get() != 0)
    // Now try getting the binary
    val key = mod.getKey()
    val binary = mod.getBinary()
    val workerMod = new NativeModule(key, binary)
    val workerFunc1 = workerMod.findLongFuncL1(st, "testFunc1")
    if (st.fail) error(s"${st}")
    assert(st.ok)
    workerFunc1.close()
    workerMod.close()
    st.close()
    mod.close()
  }

}
