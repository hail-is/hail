package is.hail.asm4s

import is.hail.HailSuite
import is.hail.asm4s.Code._
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq

import scala.language.postfixOps

import java.io.PrintWriter

import org.scalacheck.Gen.choose
import org.scalacheck.Prop.forAll

trait Z2Z { def apply(z: Boolean): Boolean }

class ASM4SSuite extends HailSuite with munit.ScalaCheckSuite {
  override val theHailClassLoader = new HailClassLoader(getClass().getClassLoader())

  test("not") {
    val notb = FunctionBuilder[Z2Z](
      "is/hail/asm4s/Z2Z",
      ArraySeq(NotGenericTypeInfo[Boolean]),
      NotGenericTypeInfo[Boolean],
    )
    notb.emit(!notb.getArg[Boolean](1))
    val not = notb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assert(!not(true))
    assert(not(false))
  }

  test("mux") {
    val gb = FunctionBuilder[Boolean, Int]("G")
    gb.emit(gb.getArg[Boolean](1).mux(11, -1))
    val g = gb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assertEquals(g(true), 11)
    assertEquals(g(false), -1)
  }

  test("add") {
    val fb = FunctionBuilder[Int, Int]("F")
    fb.emit(fb.getArg[Int](1) + 5)
    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assertEquals(f(-2), 3)
  }

  test("iinc") {
    val fb = FunctionBuilder[Int]("F")
    val l = fb.newLocal[Int]()
    fb.emit(Code(l := 0, l ++, l += 2, l))
    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assertEquals(f(), 3)
  }

  test("array") {
    val hb = FunctionBuilder[Int, Int]("H")
    val arr = hb.newLocal[Array[Int]]()
    hb.emit(Code(
      arr.store(newArray[Int](3)),
      arr(0) = 6,
      arr(1) = 7,
      arr(2) = -6,
      arr(hb.getArg[Int](1)),
    ))
    val h = hb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assertEquals(h(0), 6)
    assertEquals(h(1), 7)
    assertEquals(h(2), -6)
  }

  test("get") {
    val fb = FunctionBuilder[Foo, Int]("F")
    fb.emit(fb.getArg[Foo](1).getField[Int]("i"))
    val i = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)

    val a = new Foo
    assertEquals(i(a), 5)
  }

  test("invoke") {
    val fb = FunctionBuilder[Foo, Int]("F")
    fb.emit(fb.getArg[Foo](1).invoke[Int]("f"))
    val i = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)

    val a = new Foo
    assertEquals(i(a), 6)
  }

  test("invoke2") {
    val fb = FunctionBuilder[Foo, Int]("F")
    fb.emit(fb.getArg[Foo](1).invoke[Int, Int]("g", 6))
    val j = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)

    val a = new Foo
    assertEquals(j(a), 11)
  }

  test("newInstance") {
    val fb = FunctionBuilder[Int]("F")
    fb.emit(Code.newInstance[Foo]().invoke[Int]("f"))
    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assertEquals(f(), 6)
  }

  test("put") {
    val fb = FunctionBuilder[Int]("F")
    val inst = fb.newLocal[Foo]()
    fb.emit(Code(
      inst.store(Code.newInstance[Foo]()),
      inst.put("i", -2),
      inst.getField[Int]("i"),
    ))
    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assertEquals(f(), -2)
  }

  test("staticPut") {
    val fb = FunctionBuilder[Int]("F")
    val inst = fb.newLocal[Foo]()
    fb.emit(Code(
      inst.store(Code.newInstance[Foo]()),
      inst.put("j", -2),
      Code.getStatic[Foo, Int]("j"),
    ))
    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assertEquals(f(), -2)
  }

  test("f2") {
    val fb = FunctionBuilder[Int, Int, Int]("F")
    fb.emit(fb.getArg[Int](1) + fb.getArg[Int](2))
    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assertEquals(f(3, 5), 8)
  }

  test("compare") {
    val fb = FunctionBuilder[Int, Int, Boolean]("F")
    fb.emit(fb.getArg[Int](1) > fb.getArg[Int](2))
    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assert(f(5, 2))
    assert(!f(-1, -1))
    assert(!f(2, 5))
  }

  test("fact") {
    val fb = FunctionBuilder[Int, Int]("Fact")
    val i = fb.getArg[Int](1)
    fb.emitWithBuilder[Int] { cb =>
      val r = cb.newLocal[Int]("r")
      cb.assign(r, 1)
      cb.while_(
        i > 1, {
          cb.assign(r, r * i)
          cb.assign(i, i - 1)
        },
      )
      r
    }

    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assertEquals(f(3), 6)
    assertEquals(f(4), 24)
  }

  test("dcmp") {
    val fb = FunctionBuilder[Double, Double, Boolean]("F")
    fb.emit(fb.getArg[Double](1) > fb.getArg[Double](2))
    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assert(f(5.2, 2.3))

    val d = -2.3
    assert(!f(d, d))
    assert(!f(2.3, 5.2))
  }

  test("anewarray") {
    val fb = FunctionBuilder[Int]("F")
    val arr = fb.newLocal[Array[Foo]]()
    fb.emit(Code(
      arr.store(newArray[Foo](2)),
      arr(0) = Code.newInstance[Foo](),
      arr(1) = Code.newInstance[Foo](),
      arr(0).getField[Int]("i") + arr(1).getField[Int]("i"),
    ))
    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assertEquals(f(), 10)
  }

  def fib(_n: Int): Int = {
    var n = _n

    var a = 1
    var b = 0

    while (n > 0) {
      val tmp = a
      a = b + tmp
      b = tmp
      n = n - 1
    }

    b
  }

  property("fibonacci") {
    val Fib = FunctionBuilder[Int, Int]("Fib")
    Fib.emitWithBuilder[Int] { cb =>
      val n = Fib.getArg[Int](1)

      val a = cb.newLocal[Int]("a", 1)
      val b = cb.newLocal[Int]("b", 0)

      cb.while_(
        n > 0, {
          val tmp = cb.memoize(a, "tmp")
          cb.assign(a, tmp + b)
          cb.assign(b, tmp)
          cb.assign(n, n - 1)
        },
      )

      b
    }

    val f = Fib.result(ctx.shouldWriteIRFiles())(theHailClassLoader)

    forAll(choose(0, 100))(i => fib(i) == f(i))
  }

  object checkNanDoubleComparisons extends TestCases {
    def apply(
      op: (Code[Double], Code[Double]) => Code[Boolean],
      expected: Boolean,
    )(implicit loc: munit.Location
    ): Unit = property(s"nan double always compares false") {
      forAll { (x: Double) =>
        val F = FunctionBuilder[Double, Double, Boolean]("CMP")
        F.emit(op(F.getArg[Double](1), F.getArg[Double](2)))
        val cmp = F.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
        cmp(Double.NaN, x) == expected && cmp(x, Double.NaN) == expected
      }
    }
  }

  checkNanDoubleComparisons(_ < _, false)
  checkNanDoubleComparisons(_ <= _, false)
  checkNanDoubleComparisons(_ >= _, false)
  checkNanDoubleComparisons(_ > _, false)
  checkNanDoubleComparisons(_ ceq _, false)
  checkNanDoubleComparisons(_ cne _, true)

  object checkNanFloatComparisons extends TestCases {
    def apply(
      op: (Code[Float], Code[Float]) => Code[Boolean],
      expected: Boolean,
    )(implicit loc: munit.Location
    ): Unit = property(s"nan float always compares false case") {
      forAll { (x: Float) =>
        val F = FunctionBuilder[Float, Float, Boolean]("CMP")
        F.emit(op(F.getArg[Float](1), F.getArg[Float](2)))
        val cmp = F.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
        cmp(Float.NaN, x) == expected && cmp(x, Float.NaN) == expected
      }
    }
  }

  checkNanFloatComparisons(_ < _, false)
  checkNanFloatComparisons(_ <= _, false)
  checkNanFloatComparisons(_ >= _, false)
  checkNanFloatComparisons(_ > _, false)
  checkNanFloatComparisons(_ ceq _, false)
  checkNanFloatComparisons(_ cne _, true)

  test("defineOpsAsMethods") {
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
      val res = cb.newLocal[Int]("result")
      cb.switch(
        a,
        cb._fatal("invalid choice"),
        FastSeq(
          () => cb.assign(res, cb.invoke(add, cb.this_, b, c)),
          () => cb.assign(res, cb.invoke(sub, cb.this_, b, c)),
          () => cb.assign(res, cb.invoke(mul, cb.this_, b, c)),
        ),
      )
      res
    }

    val f =
      fb.result(ctx.shouldWriteIRFiles(), Some(new PrintWriter(System.out)))(theHailClassLoader)
    assertEquals(f(0, 1, 1), 2)
    assertEquals(f(1, 5, 1), 4)
    assertEquals(f(2, 2, 8), 16)
  }

  test("checkLocalVarsOnMethods") {
    val fb = FunctionBuilder[Int, Int, Int]("F")
    val add = fb.genMethod[Int, Int, Int]("add")

    val v1 = add.newLocal[Int]()
    val v2 = add.newLocal[Int]()

    add.emit(
      Code(
        v1 := add.getArg[Int](1),
        v2 := add.getArg[Int](2),
        v1 + v2,
      )
    )

    fb.emitWithBuilder(cb => cb.invoke(add, cb.this_, fb.getArg[Int](1), fb.getArg[Int](2)))
    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assertEquals(f(1, 1), 2)
  }

  test("checkClassFields") {

    def readField[T: TypeInfo](arg1: Int, arg2: Long, arg3: Boolean): T = {
      val fb = FunctionBuilder[Int, Long, Boolean, T]("F")
      val intField = fb.genFieldThisRef[Int]()
      val longField = fb.genFieldThisRef[Long]()
      val booleanField = fb.genFieldThisRef[Boolean]()
      val c = Code(
        intField.store(fb.getArg[Int](1)),
        longField.store(fb.getArg[Long](2)),
        booleanField.store(fb.getArg[Boolean](3)),
      )

      typeInfo[T] match {
        case IntInfo => fb.emit(Code(c, intField.load()))
        case LongInfo => fb.emit(Code(c, longField.load()))
        case BooleanInfo => fb.emit(Code(c, booleanField.load()))
      }
      val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
      f(arg1, arg2, arg3)
    }

    assertEquals(readField[Int](1, 2L, true), 1)
    assertEquals(readField[Long](1, 2L, true), 2L)
    assert(readField[Boolean](1, 2L, true))
  }

  test("checkClassFieldsFromMethod") {
    def readField[T: TypeInfo](arg1: Int, arg2: Long, arg3: Boolean): T = {
      val fb = FunctionBuilder[Int, Long, Boolean, T]("F")
      val mb = fb.genMethod[Int, Long, Boolean, T]("m")
      val intField = fb.genFieldThisRef[Int]()
      val longField = fb.genFieldThisRef[Long]()
      val booleanField = fb.genFieldThisRef[Boolean]()
      val c = Code(
        intField.store(fb.getArg[Int](1)),
        longField.store(fb.getArg[Long](2)),
        booleanField.store(fb.getArg[Boolean](3)),
      )

      typeInfo[T] match {
        case IntInfo => mb.emit(Code(c, intField.load()))
        case LongInfo => mb.emit(Code(c, longField.load()))
        case BooleanInfo => mb.emit(Code(c, booleanField.load()))
      }
      fb.emitWithBuilder { cb =>
        cb.invoke(mb, cb.this_, fb.getArg[Int](1), fb.getArg[Long](2), fb.getArg[Boolean](3))
      }
      val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
      f(arg1, arg2, arg3)
    }

    assertEquals(readField[Int](1, 2L, true), 1)
    assertEquals(readField[Long](1, 2L, true), 2L)
    assert(readField[Boolean](1, 2L, true))
  }

  test("lazyFieldEvaluatesOnce") {
    val F = FunctionBuilder[Int]("LazyField")
    val a = F.genFieldThisRef[Int]("a")
    val lzy = F.genLazyFieldThisRef(a + 1, "lzy")

    F.emit(Code(
      a := 0,
      a := lzy,
      a := lzy,
      lzy,
    ))

    val f = F.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assertEquals(f(), 1)
  }

  test("initialize") {
    val fb = FunctionBuilder[Boolean, Int]("F")
    fb.emitWithBuilder { cb =>
      val a = cb.newLocal[Int]("a")
      cb.if_(!fb.getArg[Boolean](1), cb.assign(a, 5))
      a
    }
    val f = fb.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assertEquals(f(true), 0)
    assertEquals(f(false), 5)
  }

  test("init") {
    val Main = FunctionBuilder[Int]("Main")
    val a = Main.genFieldThisRef[Int]("a")
    Main.emitInit(a := 1)
    Main.emit(a)

    val test = Main.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assertEquals(test(), 1)
  }

  test("clinit") {
    val Main = FunctionBuilder[Int]("Main")
    val a = Main.newStaticField[Int]("a")
    Main.emitClinit(a.put(1))
    Main.emit(a.get())

    val test = Main.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assertEquals(test(), 1)
  }

  test("classInstances") {
    val Counter = FunctionBuilder[Int]("Counter")
    val x = Counter.genFieldThisRef[Int]("x")
    Counter.emitInit(x := 0)
    Counter.emit {
      Code(
        x := x + 1,
        x,
      )
    }

    val Main = FunctionBuilder[Int]("Main")
    Main.emitWithBuilder[Int] { cb =>
      val a = cb.newLocal("a", Code.newInstance(Counter.cb, Counter.cb.ctor, FastSeq()))
      val b = cb.newLocal("b", Code.newInstance(Counter.cb, Counter.cb.ctor, FastSeq()))
      cb.invoke[Int](Counter.mb, a): Unit
      cb.invoke[Int](Counter.mb, a): Unit
      cb.invoke[Int](Counter.mb, b): Unit
      cb.invoke[Int](Counter.mb, a) * cb.invoke[Int](Counter.mb, b)
    }

    Counter.result(ctx.shouldWriteIRFiles())(theHailClassLoader): Unit
    val test = Main.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    assertEquals(test(), 6)
  }

  property("if") {
    val Main = FunctionBuilder[Int, Int]("If")
    Main.emitWithBuilder[Int] { cb =>
      val a = cb.mb.getArg[Int](1)
      val t = cb.newLocal[Int]("t")
      cb.if_(a > 0, cb.assign(t, a), cb.assign(t, -a))
      t
    }

    val abs = Main.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    forAll((x: Int) => abs(x) == x.abs)
  }

  property("while") {
    val Main = FunctionBuilder[Int, Int, Int]("While")
    Main.emitWithBuilder[Int] { cb =>
      val a = cb.mb.getArg[Int](1)
      val b = cb.mb.getArg[Int](2)

      val acc = cb.newLocal[Int]("signum")
      cb.if_(a > 0, cb.assign(acc, 1), cb.assign(acc, -1))

      cb.while_(
        a cne 0, {
          cb.assign(a, a - acc)
          cb.assign(b, b + acc)
        },
      )

      b
    }

    val add = Main.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    forAll(choose(-10, 10), choose(-10, 10))((x, y) => add(x, y) == x + y)
  }

  property("for") {
    val Main = FunctionBuilder[Int, Int, Int]("For")
    Main.emitWithBuilder[Int] { cb =>
      val a = cb.mb.getArg[Int](1)
      val b = cb.mb.getArg[Int](2)

      val acc = cb.newLocal[Int]("signum")

      cb.for_(
        setup = cb.if_(a > 0, cb.assign(acc, 1), cb.assign(acc, -1)),
        cond = a cne 0,
        incr = cb.assign(a, a - acc),
        body = cb.assign(b, b + acc),
      )

      b
    }

    val add = Main.result(ctx.shouldWriteIRFiles())(theHailClassLoader)
    forAll(choose(-10, 10), choose(-10, 10))((x, y) => add(x, y) == x + y)
  }

}
