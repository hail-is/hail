package is.hail.asm4s

import is.hail.asm4s.Code._
import is.hail.check.{Gen, Prop}
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test
import is.hail.asm4s.FunctionBuilder._

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
      fb.getStatic[A, Int]("j"))))
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

}
