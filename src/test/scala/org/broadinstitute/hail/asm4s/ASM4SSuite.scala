package org.broadinstitute.hail.asm4s

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class ASM4SSuite extends TestNGSuite {
  @Test def not(): Unit = {
    val notb = new Function1Builder[Boolean, Boolean]
    val not = notb.result(!notb.arg1)
    assert(!not(true))
    assert(not(false))
  }

  @Test def mux(): Unit = {
    val gb = new Function1Builder[Boolean, Int]
    val g = gb.result(gb.arg1.mux(11, -1))
    assert(g(true) == 11)
    assert(g(false) == -1)
  }

  @Test def add(): Unit = {
    val fb = new Function1Builder[Int, Int]
    val f = fb.result(fb.arg1 + 5)
    assert(f(-2) == 3)
  }

  @Test def array(): Unit = {
    val hb = new Function1Builder[Int, Int]
    val arr = hb.newLocal[Array[Int]]()
    val h = hb.result(Code(
      arr.store(hb.newArray[Int](3)),
      arr(0) = 6,
      arr(1) = 7,
      arr(2) = -6,
      arr(hb.arg1)
    ))
    assert(h(0) == 6)
    assert(h(1) == 7)
    assert(h(2) == -6)
  }

  @Test def get(): Unit = {
    val ib = new Function1Builder[A, Int]
    val i = ib.result(ib.arg1.get[Int]("i"))

    val a = new A
    assert(i(a) == 5)
  }

  @Test def invoke(): Unit = {
    val ib = new Function1Builder[A, Int]
    val i = ib.result(ib.arg1.invoke[Int]("f"))

    val a = new A
    assert(i(a) == 6)
  }

  @Test def invoke2(): Unit = {
    val jb = new Function1Builder[A, Int]
    val j = jb.result(jb.arg1.invoke[Int, Int]("g", 6))

    val a = new A
    assert(j(a) == 11)
  }

  @Test def newInstance(): Unit = {
    val fb = new Function0Builder[Int]
    val f = fb.result(
      fb.newInstance[A]().invoke[Int]("f"))
    assert(f() == 6)
  }

  @Test def put(): Unit = {
    val fb = new Function0Builder[Int]
    val inst = fb.newLocal[A]()
    val f = fb.result(Code(
      inst.store(fb.newInstance[A]()),
      inst.put("i", -2),
      inst.get("i")))
    assert(f() == -2)
  }

  @Test def staticPut(): Unit = {
    val fb = new Function0Builder[Int]
    val inst = fb.newLocal[A]()
    val f = fb.result(Code(
      inst.store(fb.newInstance[A]()),
      inst.put("j", -2),
      fb.getStatic[A, Int]("j")))
    assert(f() == -2)
  }

  @Test def f2(): Unit = {
    val fb = new Function2Builder[Int, Int, Int]
    val f = fb.result(fb.arg1 + fb.arg2)
    assert(f(3, 5) == 8)
  }

  @Test def compare(): Unit = {
    val fb = new Function2Builder[Int, Int, Boolean]
    val f = fb.result(fb.arg1 > fb.arg2)
    assert(f(5, 2))
    assert(!f(-1, -1))
    assert(!f(2, 5))
  }

  @Test def fact(): Unit = {
    val fb = new Function1Builder[Int, Int]
    val i = fb.arg1
    val r = fb.newLocal[Int]()
    val f = fb.result(Code(
      r.store(1),
      fb.whileLoop(
        fb.arg1 > 1,
        Code(
          r.store(r * i),
          i.store(i - 1))),
      r))

    assert(f(3) == 6)
    assert(f(4) == 24)
  }

  @Test def dcmp(): Unit = {
    val fb = new Function2Builder[Double, Double, Boolean]
    val f = fb.result(fb.arg1 > fb.arg2)
    assert(f(5.2, 2.3))

    val d = -2.3
    assert(!f(d, d))
    assert(!f(2.3, 5.2))
  }

  @Test def anewarray(): Unit = {
    val fb = new Function0Builder[Int]
    val arr = fb.newLocal[Array[A]]()
    val f = fb.result(Code(
      arr.store(fb.newArray[A](2)),
      arr(0) = fb.newInstance[A](),
      arr(1) = fb.newInstance[A](),
      arr(0).get[Int]("i") + arr(1).get[Int]("i")
    ))
    assert(f() == 10)
  }  
}
