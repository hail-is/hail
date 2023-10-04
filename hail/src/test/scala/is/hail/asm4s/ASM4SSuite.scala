package is.hail.asm4s

import java.io.PrintWriter
import is.hail.HailSuite
import is.hail.asm4s.Code._
import is.hail.asm4s.FunctionBuilder._
import is.hail.check.{Gen, Prop}
import is.hail.utils.HailException
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

import scala.collection.mutable
import scala.language.postfixOps
import scala.util.Random

trait Z2Z { def apply(z:Boolean): Boolean }

class ASM4SSuite extends HailSuite {
  private[this] val theHailClassLoader = new HailClassLoader(getClass().getClassLoader())

  @Test def not(): Unit = {
    val notb = FunctionBuilder[Z2Z]("is/hail/asm4s/Z2Z", Array(NotGenericTypeInfo[Boolean]), NotGenericTypeInfo[Boolean])
    notb.emit(!notb.getArg[Boolean](1))
    val not = notb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assert(!not(true))
    assert(not(false))
  }

  @Test def mux(): Unit = {
    val gb = FunctionBuilder[Boolean, Int]("G")
    gb.emit(gb.getArg[Boolean](1).mux(11, -1))
    val g = gb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assert(g(true) == 11)
    assert(g(false) == -1)
  }

  @Test def add(): Unit = {
    val fb = FunctionBuilder[Int, Int]("F")
    fb.emit(fb.getArg[Int](1) + 5)
    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assert(f(-2) == 3)
  }

  @Test def iinc(): Unit = {
    val fb = FunctionBuilder[Int]("F")
    val l = fb.newLocal[Int]()
    fb.emit(Code(l := 0, l++, l += 2, l))
    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assert(f() == 3)
  }

  @Test def array(): Unit = {
    val hb = FunctionBuilder[Int, Int]("H")
    val arr = hb.newLocal[Array[Int]]()
    hb.emit(Code(
      arr.store(newArray[Int](3)),
      arr(0) = 6,
      arr(1) = 7,
      arr(2) = -6,
      arr(hb.getArg[Int](1))
    ))
    val h = hb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assert(h(0) == 6)
    assert(h(1) == 7)
    assert(h(2) == -6)
  }

  @Test def get(): Unit = {
    val fb = FunctionBuilder[A, Int]("F")
    fb.emit(fb.getArg[A](1).getField[Int]("i"))
    val i = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)

    val a = new A
    assert(i(a) == 5)
  }

  @Test def invoke(): Unit = {
    val fb = FunctionBuilder[A, Int]("F")
    fb.emit(fb.getArg[A](1).invoke[Int]("f"))
    val i = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)

    val a = new A
    assert(i(a) == 6)
  }

  @Test def invoke2(): Unit = {
    val fb = FunctionBuilder[A, Int]("F")
    fb.emit(fb.getArg[A](1).invoke[Int, Int]("g", 6))
    val j = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)

    val a = new A
    assert(j(a) == 11)
  }

  @Test def newInstance(): Unit = {
    val fb = FunctionBuilder[Int]("F")
    fb.emit(Code.newInstance[A]().invoke[Int]("f"))
    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assert(f() == 6)
  }

  @Test def put(): Unit = {
    val fb = FunctionBuilder[Int]("F")
    val inst = fb.newLocal[A]()
    fb.emit(Code(
      inst.store(Code.newInstance[A]()),
      inst.put("i", -2),
      inst.getField[Int]("i")))
    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assert(f() == -2)
  }

  @Test def staticPut(): Unit = {
    val fb = FunctionBuilder[Int]("F")
    val inst = fb.newLocal[A]()
    fb.emit(Code(
      inst.store(Code.newInstance[A]()),
      inst.put("j", -2),
      Code.getStatic[A, Int]("j")))
    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assert(f() == -2)
  }

  @Test def f2(): Unit = {
    val fb = FunctionBuilder[Int, Int, Int]("F")
    fb.emit(fb.getArg[Int](1) + fb.getArg[Int](2))
    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assert(f(3, 5) == 8)
  }

  @Test def compare(): Unit = {
    val fb = FunctionBuilder[Int, Int, Boolean]("F")
    fb.emit(fb.getArg[Int](1) > fb.getArg[Int](2))
    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assert(f(5, 2))
    assert(!f(-1, -1))
    assert(!f(2, 5))
  }

  @Test def fact(): Unit = {
    val fb = FunctionBuilder[Int, Int]("Fact")
    val i = fb.getArg[Int](1)
    fb.emitWithBuilder[Int] { cb =>
      val r = cb.newLocal[Int]("r")
      cb.assign(r, 1)
      cb.while_(i > 1, {
        cb.assign(r, r * i)
        cb.assign(i, i - 1)
      })
      r
    }

    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assert(f(3) == 6)
    assert(f(4) == 24)
  }

  @Test def dcmp(): Unit = {
    val fb = FunctionBuilder[Double, Double, Boolean]("F")
    fb.emit(fb.getArg[Double](1) > fb.getArg[Double](2))
    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assert(f(5.2, 2.3))

    val d = -2.3
    assert(!f(d, d))
    assert(!f(2.3, 5.2))
  }

  @Test def anewarray(): Unit = {
    val fb = FunctionBuilder[Int]("F")
    val arr = fb.newLocal[Array[A]]()
    fb.emit(Code(
      arr.store(newArray[A](2)),
      arr(0) = Code.newInstance[A](),
      arr(1) = Code.newInstance[A](),
      arr(0).getField[Int]("i") + arr(1).getField[Int]("i")
    ))
    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assert(f() == 10)
  }

  def fibonacciReference(i: Int): Int = i match {
    case 0 => 0
    case 1 => 1
    case n => fibonacciReference(n-1) + fibonacciReference(n-2)
  }

  @Test def fibonacci(): Unit = {
    val fb = FunctionBuilder[Int, Int]("Fib")

    val i = fb.getArg[Int](1)
    fb.emitWithBuilder[Int] { cb =>
      val n = cb.newLocal[Int]("n")
      cb.ifx(i < 3, cb.assign(n, 1), {
        val vn_1 = cb.newLocal[Int]("vn_1")
        val vn_2 = cb.newLocal[Int]("vn_2")
        cb.assign(vn_1, 1)
        cb.assign(vn_2, 1)
        cb.while_(i > 3, {
          val temp = fb.newLocal[Int]()
          cb.assign(temp, vn_2 + vn_1)
          cb.assign(vn_1, temp)
          cb.assign(i, i - 1)
        })
        cb.assign(n, vn_2 + vn_1)
      })
      n
    }
    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)

    Prop.forAll(Gen.choose(0, 100)) { i =>
      fibonacciReference(i) == f(i)
    }
  }

  @Test def nanAlwaysComparesFalse(): Unit = {
    Prop.forAll { (x: Double) =>
      {
        val fb = FunctionBuilder[Boolean]("F")
        fb.emit(Double.NaN < x)
        val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
        assert(!f())
      }
      {
        val fb = FunctionBuilder[Boolean]("F")
        fb.emit(Double.NaN <= x)
        val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
        assert(!f())
      }
      {
        val fb = FunctionBuilder[Boolean]("F")
        fb.emit(Double.NaN > x)
        val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
        assert(!f())
      }
      {
        val fb = FunctionBuilder[Boolean]("F")
        fb.emit(Double.NaN >= x)
        val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
        assert(!f())
      }
      {
        val fb = FunctionBuilder[Boolean]("F")
        fb.emit(new CodeDouble(Double.NaN).ceq(x))
        val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
        assert(!f())
      }
      {
        val fb = FunctionBuilder[Boolean]("F")
        fb.emit(new CodeDouble(Double.NaN).cne(x))
        val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
        assert(f())
      }

      true
    }.check()
  }

  @Test def nanFloatAlwaysComparesFalse(): Unit = {
    Prop.forAll { (x: Float) =>
      {
        val fb = FunctionBuilder[Boolean]("F")
        fb.emit(Float.NaN < x)
        val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
        assert(!f())
      }
      {
        val fb = FunctionBuilder[Boolean]("F")
        fb.emit(Float.NaN <= x)
        val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
        assert(!f())
      }
      {
        val fb = FunctionBuilder[Boolean]("F")
        fb.emit(Float.NaN > x)
        val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
        assert(!f())
      }
      {
        val fb = FunctionBuilder[Boolean]("F")
        fb.emit(Float.NaN >= x)
        val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
        assert(!f())
      }
      {
        val fb = FunctionBuilder[Boolean]("F")
        fb.emit(new CodeFloat(Float.NaN).ceq(x))
        val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
        assert(!f())
      }
      {
        val fb = FunctionBuilder[Boolean]("F")
        fb.emit(new CodeFloat(Float.NaN).cne(x))
        val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
        assert(f())
      }

      true
    }.check()
  }

  @Test def defineOpsAsMethods(): Unit = {
    val fb = FunctionBuilder[Int, Int, Int, Int]("F")
    val add = fb.genMethod[Int, Int, Int]("add")
    val sub = fb.genMethod[Int, Int, Int]("sub")
    val mul = fb.genMethod[Int, Int, Int]("mul")

    add.emit(add.getArg[Int](1) + add.getArg[Int](2))
    sub.emit(sub.getArg[Int](1) - sub.getArg[Int](2))
    mul.emit(mul.getArg[Int](1) * mul.getArg[Int](2))

    fb.emitWithBuilder { cb =>
      val a = fb.getArg[Int](1)
      val b = fb.getArg[Int](2)
      val c = fb.getArg[Int](3)
      val res = cb.newLocal[Int]("res")
      cb.ifx(a.ceq(0), {
        cb.assign(res, add.invoke(cb, b, c))
      }, {
        cb.ifx(a.ceq(1),
          cb.assign(res, sub.invoke(cb, b, c)),
          cb.assign(res, mul.invoke(cb, b, c)))
      })
      res
    }
    val f = fb.result(ctx.shouldWriteIRFiles(), Some(new PrintWriter(System.out)))(theHailClassLoader)
    assert(f(0, 1, 1) == 2)
    assert(f(1, 5, 1) == 4)
    assert(f(2, 2, 8) == 16)
  }

  @Test def checkLocalVarsOnMethods(): Unit = {
    val fb = FunctionBuilder[Int, Int, Int]("F")
    val add = fb.genMethod[Int, Int, Int]("add")

    val v1 = add.newLocal[Int]()
    val v2 = add.newLocal[Int]()

    add.emit(Code(v1 := add.getArg[Int](1),
      v2 := add.getArg[Int](2),
      v1 + v2))

    fb.emitWithBuilder(add.invoke(_, fb.getArg[Int](1), fb.getArg[Int](2)))
    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assert(f(1, 1) == 2)
  }

  @Test def checkClassFields(): Unit = {

    def readField[T: TypeInfo](arg1: Int, arg2: Long, arg3: Boolean): T = {
      val fb = FunctionBuilder[Int, Long, Boolean, T]("F")
      val intField = fb.genFieldThisRef[Int]()
      val longField = fb.genFieldThisRef[Long]()
      val booleanField = fb.genFieldThisRef[Boolean]()
      val c = Code(
        intField.store(fb.getArg[Int](1)),
        longField.store(fb.getArg[Long](2)),
        booleanField.store(fb.getArg[Boolean](3)))

      typeInfo[T] match {
        case IntInfo => fb.emit(Code(c, intField.load()))
        case LongInfo => fb.emit(Code(c, longField.load()))
        case BooleanInfo => fb.emit(Code(c, booleanField.load()))
      }
      val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
      f(arg1, arg2, arg3)
    }

    assert(readField[Int](1, 2L, true) == 1)
    assert(readField[Long](1, 2L, true) == 2L)
    assert(readField[Boolean](1, 2L, true))
  }

  @Test def checkClassFieldsFromMethod(): Unit = {
    def readField[T: TypeInfo](arg1: Int, arg2: Long, arg3: Boolean): T = {
      val fb = FunctionBuilder[Int, Long, Boolean, T]("F")
      val mb = fb.genMethod[Int, Long, Boolean, T]("m")
      val intField = fb.genFieldThisRef[Int]()
      val longField = fb.genFieldThisRef[Long]()
      val booleanField = fb.genFieldThisRef[Boolean]()
      val c = Code(
        intField.store(fb.getArg[Int](1)),
        longField.store(fb.getArg[Long](2)),
        booleanField.store(fb.getArg[Boolean](3)))

      typeInfo[T] match {
        case IntInfo => mb.emit(Code(c, intField.load()))
        case LongInfo => mb.emit(Code(c, longField.load()))
        case BooleanInfo => mb.emit(Code(c, booleanField.load()))
      }
      fb.emitWithBuilder(mb.invoke(_, fb.getArg[Int](1), fb.getArg[Long](2), fb.getArg[Boolean](3)))
      val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
      f(arg1, arg2, arg3)
    }

    assert(readField[Int](1, 2L, true) == 1)
    assert(readField[Long](1, 2L, true) == 2L)
    assert(readField[Boolean](1, 2L, true))
  }

  @Test def lazyFieldEvaluatesOnce(): Unit = {
    val F = FunctionBuilder[Int]("LazyField")
    val a = F.genFieldThisRef[Int]("a")
    val lzy = F.genLazyFieldThisRef(a + 1, "lzy")

    F.emit(Code(
      a := 0,
      a := lzy,
      a := lzy,
      lzy
    ))

    val f = F.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assert(f() == 1)
  }

  @Test def testInitialize(): Unit = {
    val fb = FunctionBuilder[Boolean, Int]("F")
    fb.emitWithBuilder { cb =>
      val a = cb.newLocal[Int]("a")
      cb.ifx(!fb.getArg[Boolean](1), cb.assign(a, 5))
      a
    }
    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assert(f(true) == 0)
    assert(f(false) == 5)
  }

  @Test def testInit(): Unit = {
    val Main = FunctionBuilder[Int]("Main")
    val a = Main.genFieldThisRef[Int]("a")
    Main.emitInit { a := 1 }
    Main.emit { a }

    val test = Main.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assert(test() == 1)
  }

  @Test def testClinit(): Unit = {
    val Main = FunctionBuilder[Int]("Main")
    val a = Main.newStaticField[Int]("a")
    Main.emitClinit { a.put(1) }
    Main.emit { a.get() }

    val test = Main.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assert(test() == 1)
  }

  @Test def testIf(): Unit = {
    val Main = FunctionBuilder[Int, Int]("If")
    Main.emitWithBuilder[Int] { cb =>
      val a = cb.mb.getArg[Int](1)
      val t = cb.newLocal[Int]("t")
      cb.ifx(a > 0, cb.assign(t, a), cb.assign(t, -a))
      t
    }

    val abs = Main.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    Prop.forAll { (x: Int) => abs(x) == x.abs }.check()
  }

  @Test def testWhile(): Unit = {
    val Main = FunctionBuilder[Int, Int, Int]("While")
    Main.emitWithBuilder[Int] { cb =>
      val a = cb.mb.getArg[Int](1)
      val b = cb.mb.getArg[Int](2)

      val acc = cb.newLocal[Int]("signum")
      cb.ifx(a > 0, cb.assign(acc, 1), cb.assign(acc, -1))

      cb.while_(a cne 0, {
        cb.assign(a, a - acc)
        cb.assign(b, b + acc)
      })

      b
    }

    val add = Main.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    Prop.forAll(Gen.choose(-10, 10), Gen.choose(-10, 10))
      { (x, y) => add(x, y) == x + y }
      .check()
  }

  @Test def testFor(): Unit = {
    val Main = FunctionBuilder[Int, Int, Int]("For")
    Main.emitWithBuilder[Int] { cb =>
      val a = cb.mb.getArg[Int](1)
      val b = cb.mb.getArg[Int](2)

      val acc = cb.newLocal[Int]("signum")

      cb.for_(
        setup = cb.ifx(a > 0, cb.assign(acc, 1), cb.assign(acc, -1)),
        cond = a cne 0,
        incr = cb.assign(a, a - acc),
        body = cb.assign(b, b + acc)
      )

      b
    }

    val add = Main.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    Prop.forAll(Gen.choose(-10, 10), Gen.choose(-10, 10)) { (x, y) => add(x, y) == x + y }
      .check()
  }

}
