package is.hail.asm4s

import java.io.PrintWriter

import is.hail.asm4s.Code._
import is.hail.check.{Gen, Prop}
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test
import is.hail.asm4s.FunctionBuilder._
import org.objectweb.asm.tree.InsnNode
import scala.language.postfixOps

trait Z2Z { def apply(z:Boolean): Boolean }

class ASM4SSuite extends TestNGSuite {
  @Test def not(): Unit = {
    val notb = new FunctionBuilder[Z2Z](Array(NotGenericTypeInfo[Boolean]), NotGenericTypeInfo[Boolean], "is/hail/asm4s/Z2Z")
    notb.emit(_return(!notb.getArg[Boolean](1)))
    val not = notb.result()()
    assert(!not(true))
    assert(not(false))
  }

  @Test def mux(): Unit = {
    val gb = functionBuilder[Boolean, Int]
    gb.emit(_return(gb.getArg[Boolean](1).mux(11, -1)))
    val g = gb.result()()
    assert(g(true) == 11)
    assert(g(false) == -1)
  }

  @Test def add(): Unit = {
    val fb = functionBuilder[Int, Int]
    fb.emit(_return(fb.getArg[Int](1) + 5))
    val f = fb.result()()
    assert(f(-2) == 3)
  }

  @Test def iinc(): Unit = {
    val fb = functionBuilder[Int]
    val l = fb.newLocal[Int]
    fb.emit(_return(Code(l := 0, l++, l += 2, l)))
    val f = fb.result()()
    assert(f() == 3)
  }

  @Test def array(): Unit = {
    val hb = functionBuilder[Int, Int]
    val arr = hb.newLocal[Array[Int]]()
    hb.emit(Code(
      arr.store(newArray[Int](3)),
      arr(0) = 6,
      arr(1) = 7,
      arr(2) = -6,
      _return(arr(hb.getArg[Int](1)))
    ))
    val h = hb.result()()
    assert(h(0) == 6)
    assert(h(1) == 7)
    assert(h(2) == -6)
  }

  @Test def get(): Unit = {
    val ib = functionBuilder[A, Int]
    ib.emit(_return(ib.getArg[A](1).get[Int]("i")))
    val i = ib.result()()

    val a = new A
    assert(i(a) == 5)
  }

  @Test def invoke(): Unit = {
    val ib = functionBuilder[A, Int]
    ib.emit(_return(ib.getArg[A](1).invoke[Int]("f")))
    val i = ib.result()()

    val a = new A
    assert(i(a) == 6)
  }

  @Test def invoke2(): Unit = {
    val jb = functionBuilder[A, Int]
    jb.emit(_return(jb.getArg[A](1).invoke[Int, Int]("g", 6)))
    val j = jb.result()()

    val a = new A
    assert(j(a) == 11)
  }

  @Test def newInstance(): Unit = {
    val fb = functionBuilder[Int]
    fb.emit(_return(Code.newInstance[A]().invoke[Int]("f")))
    val f = fb.result()()
    assert(f() == 6)
  }

  @Test def put(): Unit = {
    val fb = functionBuilder[Int]
    val inst = fb.newLocal[A]()
    fb.emit(_return(Code(
      inst.store(Code.newInstance[A]()),
      inst.put("i", -2),
      inst.get[Int]("i"))))
    val f = fb.result()()
    assert(f() == -2)
  }

  @Test def staticPut(): Unit = {
    val fb = functionBuilder[Int]
    val inst = fb.newLocal[A]()
    fb.emit(_return(Code(
      inst.store(Code.newInstance[A]()),
      inst.put("j", -2),
      Code.getStatic[A, Int]("j"))))
    val f = fb.result()()
    assert(f() == -2)
  }

  @Test def f2(): Unit = {
    val fb = functionBuilder[Int, Int, Int]
    fb.emit(_return(fb.getArg[Int](1) + fb.getArg[Int](2)))
    val f = fb.result()()
    assert(f(3, 5) == 8)
  }

  @Test def compare(): Unit = {
    val fb = functionBuilder[Int, Int, Boolean]
    fb.emit(_return(fb.getArg[Int](1) > fb.getArg[Int](2)))
    val f = fb.result()()
    assert(f(5, 2))
    assert(!f(-1, -1))
    assert(!f(2, 5))
  }

  @Test def fact(): Unit = {
    val fb = functionBuilder[Int, Int]
    val i = fb.getArg[Int](1)
    val r = fb.newLocal[Int]()
    fb.emit(_return(Code(
      r.store(1),
      whileLoop(
        fb.getArg[Int](1) > 1,
        Code(
          r.store(r * i),
          i.store(i - 1))),
      r)))
    val f = fb.result()()

    assert(f(3) == 6)
    assert(f(4) == 24)
  }

  @Test def dcmp(): Unit = {
    val fb = functionBuilder[Double, Double, Boolean]
    fb.emit(_return(fb.getArg[Double](1) > fb.getArg[Double](2)))
    val f = fb.result()()
    assert(f(5.2, 2.3))

    val d = -2.3
    assert(!f(d, d))
    assert(!f(2.3, 5.2))
  }

  @Test def anewarray(): Unit = {
    val fb = functionBuilder[Int]
    val arr = fb.newLocal[Array[A]]()
    fb.emit(_return(Code(
      arr.store(newArray[A](2)),
      arr(0) = Code.newInstance[A](),
      arr(1) = Code.newInstance[A](),
      arr(0).get[Int]("i") + arr(1).get[Int]("i")
    )))
    val f = fb.result()()
    assert(f() == 10)
  }

  def fibonacciReference(i: Int): Int = i match {
    case 0 => 0
    case 1 => 1
    case n => fibonacciReference(n-1) + fibonacciReference(n-2)
  }

  @Test def fibonacci(): Unit = {
    val fb = functionBuilder[Int, Int]
    val i = fb.getArg[Int](1)
    val n = fb.newLocal[Int]
    val vn_2 = fb.newLocal[Int]
    val vn_1 = fb.newLocal[Int]
    val temp = fb.newLocal[Int]
    fb.emit(_return(
      (i < 3).mux(1, Code(
        vn_2.store(1),
        vn_1.store(1),
        whileLoop(
          i > 3,
          Code(
            temp.store(vn_2 + vn_1),
            vn_2.store(vn_2),
            vn_1.store(temp),
            i.store(i - 1)
          )
        ),
        vn_2 + vn_1
      ))
    ))
    val f = fb.result()()

    Prop.forAll(Gen.choose(0, 100)) { i =>
      fibonacciReference(i) == f(i)
    }
  }

  @Test def nanAlwaysComparesFalse(): Unit = {
    Prop.forAll { (x: Double) =>
      {
        val fb = functionBuilder[Boolean]
        fb.emit(_return(Double.NaN < x))
        val f = fb.result()()
        assert(!f())
      }
      {
        val fb = functionBuilder[Boolean]
        fb.emit(_return(Double.NaN <= x))
        val f = fb.result()()
        assert(!f())
      }
      {
        val fb = functionBuilder[Boolean]
        fb.emit(_return(Double.NaN > x))
        val f = fb.result()()
        assert(!f())
      }
      {
        val fb = functionBuilder[Boolean]
        fb.emit(_return(Double.NaN >= x))
        val f = fb.result()()
        assert(!f())
      }
      {
        val fb = functionBuilder[Boolean]
        fb.emit(_return(new CodeDouble(Double.NaN).ceq(x)))
        val f = fb.result()()
        assert(!f())
      }
      {
        val fb = functionBuilder[Boolean]
        fb.emit(_return(new CodeDouble(Double.NaN).cne(x)))
        val f = fb.result()()
        assert(f())
      }

      true
    }.check()
  }

  @Test def nanFloatAlwaysComparesFalse(): Unit = {
    Prop.forAll { (x: Float) =>
      {
        val fb = functionBuilder[Boolean]
        fb.emit(_return(Float.NaN < x))
        val f = fb.result()()
        assert(!f())
      }
      {
        val fb = functionBuilder[Boolean]
        fb.emit(_return(Float.NaN <= x))
        val f = fb.result()()
        assert(!f())
      }
      {
        val fb = functionBuilder[Boolean]
        fb.emit(_return(Float.NaN > x))
        val f = fb.result()()
        assert(!f())
      }
      {
        val fb = functionBuilder[Boolean]
        fb.emit(_return(Float.NaN >= x))
        val f = fb.result()()
        assert(!f())
      }
      {
        val fb = functionBuilder[Boolean]
        fb.emit(_return(new CodeFloat(Float.NaN).ceq(x)))
        val f = fb.result()()
        assert(!f())
      }
      {
        val fb = functionBuilder[Boolean]
        fb.emit(_return(new CodeFloat(Float.NaN).cne(x)))
        val f = fb.result()()
        assert(f())
      }

      true
    }.check()
  }

  @Test def makeMethods(): Unit = {
    val fb = FunctionBuilder.functionBuilder[Int]
    val methods = Array.tabulate[MethodBuilder](3)(_ => fb.newMethod[Int, Int, Int])
    val locals = Array.tabulate[LocalRef[Int]](9)(i => methods(i / 3).newLocal[Int])
    var i = 0
    while (i < 3) {
      var j = 0
      while (j < 3) {
        methods(i).emit(locals(3*i + j) := const(i))
        j += 1
      }
      methods(i).emit(locals(3*i))
      methods(i).mn.instructions
      i += 1
    }
    fb.emit(Code._return[Int](methods(1).invoke(0,0)))
    val f = fb.result()()
    assert(f() == 1)
  }

  @Test def defineOpsAsMethods(): Unit = {
    val fb = FunctionBuilder.functionBuilder[Int, Int, Int, Int]
    val add = fb.newMethod[Int, Int, Int]
    val sub = fb.newMethod[Int, Int, Int]
    val mul = fb.newMethod[Int, Int, Int]

    add.emit(add.getArg[Int](1) + add.getArg[Int](2))
    sub.emit(sub.getArg[Int](1) - sub.getArg[Int](2))
    mul.emit(mul.getArg[Int](1) * mul.getArg[Int](2))

    fb.emit(fb.getArg[Int](1).ceq(0).mux(
      add.invoke(fb.getArg[Int](2), fb.getArg[Int](3)),
      fb.getArg[Int](1).ceq(1).mux(
        sub.invoke(fb.getArg[Int](2), fb.getArg[Int](3)),
        mul.invoke(fb.getArg[Int](2), fb.getArg[Int](3))
      )))
    val f = fb.result(Some(new PrintWriter(System.out)))()
    assert(f(0, 1, 1) == 2)
    assert(f(1, 5, 1) == 4)
    assert(f(2, 2, 8) == 16)
  }


  @Test def checkLocalVarsOnMethods(): Unit = {
    val fb = FunctionBuilder.functionBuilder[Int, Int, Int]
    val add = fb.newMethod[Int, Int, Int]

    val v1 = add.newLocal[Int]
    val v2 = add.newLocal[Int]

    add.emit(Code(v1 := add.getArg[Int](1),
      v2 := add.getArg[Int](2),
      v1 + v2))

    fb.emit(add.invoke(fb.getArg[Int](1), fb.getArg[Int](2)))
    val f = fb.result()()
    assert(f(1, 1) == 2)
  }

  @Test def checkClassFields(): Unit = {

    def readField[T: TypeInfo](arg1: Int, arg2: Long, arg3: Boolean): T = {
      val fb = FunctionBuilder.functionBuilder[Int, Long, Boolean, T]
      val intField = fb.newField[Int]
      val longField = fb.newField[Long]
      val booleanField = fb.newField[Boolean]
      val c = Code(
        intField.store(fb.getArg[Int](1)),
        longField.store(fb.getArg[Long](2)),
        booleanField.store(fb.getArg[Boolean](3)))

      typeInfo[T] match {
        case IntInfo => fb.emit(Code(c, intField.load()))
        case LongInfo => fb.emit(Code(c, longField.load()))
        case BooleanInfo => fb.emit(Code(c, booleanField.load()))
      }
      val f = fb.result()()
      f(arg1, arg2, arg3)
    }

    assert(readField[Int](1, 2L, true) == 1)
    assert(readField[Long](1, 2L, true) == 2L)
    assert(readField[Boolean](1, 2L, true))
  }

  @Test def checkClassFieldsFromMethod(): Unit = {
    def readField[T: TypeInfo](arg1: Int, arg2: Long, arg3: Boolean): T = {
      val fb = FunctionBuilder.functionBuilder[Int, Long, Boolean, T]
      val mb = fb.newMethod[Int, Long, Boolean, T]
      val intField = fb.newField[Int]
      val longField = fb.newField[Long]
      val booleanField = fb.newField[Boolean]
      val c = Code(
        intField.store(fb.getArg[Int](1)),
        longField.store(fb.getArg[Long](2)),
        booleanField.store(fb.getArg[Boolean](3)))

      typeInfo[T] match {
        case IntInfo => mb.emit(Code(c, intField.load()))
        case LongInfo => mb.emit(Code(c, longField.load()))
        case BooleanInfo => mb.emit(Code(c, booleanField.load()))
      }
      fb.emit(mb.invoke(fb.getArg[Int](1), fb.getArg[Long](2), fb.getArg[Boolean](3)))
      val f = fb.result()()
      f(arg1, arg2, arg3)
    }

    assert(readField[Int](1, 2L, true) == 1)
    assert(readField[Long](1, 2L, true) == 2L)
    assert(readField[Boolean](1, 2L, true))
  }

  @Test def lazyFieldEvaluatesOnce(): Unit = {
    val fb = FunctionBuilder.functionBuilder[Int]
    val v2 = fb.newField[Int]
    val v1 = fb.newLazyField(v2 + 1)

    fb.emit(v2 := 0)
    fb.emit(v2 := v1)
    fb.emit(v2 := v1)
    fb.emit(v1)

    assert(fb.result()()() == 1)
  }

  @Test def fbFunctionsCanBeNested(): Unit = {
    val fb = FunctionBuilder.functionBuilder[Boolean]
    val fb2 = fb.newDependentFunction[Int, Boolean]
    val localF = fb.newField[AsmFunction1[Int, Boolean]]

    val wrappedInt = Code.invokeStatic[java.lang.Integer, Int, java.lang.Integer]("valueOf", 0)
    val rawOut = localF.load().invoke[java.lang.Object, java.lang.Object]("apply", wrappedInt)

    fb2.emit(true)
    fb.emit(Code(
      localF := fb2.newInstance(),
      checkcast[java.lang.Boolean](rawOut).invoke[Boolean]("booleanValue")
    ))

    val f = fb.result()()
    assert(f())
  }

  @Test def dependentFunctionsCanUseParentsFields(): Unit = {
    val fb = FunctionBuilder.functionBuilder[Int, Int, Int]
    val fb2 = fb.newDependentFunction[Int, Int]

    val localF = fb.newField[AsmFunction1[Int, Int]]

    val field1 = fb.newField[Int]
    val field2 = fb2.addField[Int](field1.load())

    def wrappedCall(c: Code[Int]) =
      localF.load().invoke[java.lang.Object, java.lang.Object]("apply",
        Code.invokeStatic[java.lang.Integer, Int, java.lang.Integer]("valueOf", c))

    fb2.emit(field2 + fb2.getArg[Int](1))
    fb.emit(Code(
      field1 := fb.getArg[Int](1),
      localF := fb2.newInstance(),
      checkcast[java.lang.Integer](wrappedCall(fb.getArg[Int](2))).invoke[Int]("intValue")
    ))

    val f = fb.result()()
    assert(f(1, 2) == 3)
  }
}
