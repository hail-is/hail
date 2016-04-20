package org.broadinstitute.hail.check

import scala.collection.mutable.ArrayBuffer

abstract class Prop {
  def apply(p: Parameters, name: Option[String] = None): Unit

  def check() {
    apply(Parameters.default)
  }

  def check(size: Int = 100, count: Int = 100) {
    apply(Parameters.default.copy(size = size, count = count))
  }
}

class GenProp1[T1](g1: Gen[T1], f: (T1) => Boolean) extends Prop {
  override def apply(p: Parameters, name: Option[String]) {
    val prefix = name.map(_ + ": ").getOrElse("")
    for (i <- 0 until p.count) {
      val v1 = g1(p)
      val r = f(v1)
      if (!r) {
        println(s"""! ${prefix}Falsified after $i passed tests.""")
        println("> ARG_0: $v1")
        assert(r)
      }
    }
    println(s" + ${prefix}OK, passed ${p.count} tests.")
  }
}

class GenProp2[T1, T2](g1: Gen[T1], g2: Gen[T2], f: (T1, T2) => Boolean) extends Prop {
  override def apply(p: Parameters, name: Option[String]) {
    val prefix = name.map(_ + ": ").getOrElse("")
    for (i <- 0 until p.count) {
      val v1 = g1(p)
      val v2 = g2(p)
      val r = f(v1, v2)
      if (!r) {
        println(s"! ${prefix}Falsified after $i passed tests.")
        println("> ARG_0: $v1")
        println("> ARG_1: $v2")
        assert(r)
      }
    }
    println(s" + ${prefix}OK, passed ${p.count} tests.")
  }
}

class GenProp3[T1, T2, T3](g1: Gen[T1], g2: Gen[T2], g3: Gen[T3], f: (T1, T2, T3) => Boolean) extends Prop {
  override def apply(p: Parameters, name: Option[String]) {
    val prefix = name.map(_ + ": ").getOrElse("")
    for (i <- 0 until p.count) {
      val v1 = g1(p)
      val v2 = g2(p)
      val v3 = g3(p)
      val r = f(v1, v2, v3)
      if (!r) {
        println(s"! ${prefix}Falsified after $i passed tests.")
        println("> ARG_0: $v1")
        println("> ARG_1: $v2")
        println("> ARG_2: $v3")
        assert(r)
      }
    }
    println(s" + ${prefix}OK, passed ${p.count} tests.")
  }
}

class Properties(val name: String) extends Prop {
  val properties = ArrayBuffer.empty[(String, Prop)]

  class PropertySpecifier {
    def update(propName: String, prop: Prop) {
      properties += (name + "." + propName) -> prop
    }
  }

  lazy val property = new PropertySpecifier

  override def apply(p: Parameters, prefix: Option[String]) {
    for ((propName, prop) <- properties)
      prop(p, prefix.map(_ + "." + propName).orElse(Some(propName)))
  }
}

object Prop {
  def check(p: Prop) { p.check() }

  def forAll[T1](g1: Gen[T1])(p: (T1) => Boolean): Prop =
    new GenProp1(g1, p)

  def forAll[T1, T2](g1: Gen[T1], g2: Gen[T2])(p: (T1, T2) => Boolean): Prop =
    new GenProp2(g1, g2, p)

  def forAll[T1, T2, T3](g1: Gen[T1], g2: Gen[T2], g3: Gen[T3])(p: (T1, T2, T3) => Boolean): Prop =
    new GenProp3(g1, g2, g3, p)

  def forAll[T1](p: (T1) => Boolean)(implicit a1: Arbitrary[T1]): Prop =
    new GenProp1(a1.arbitrary, p)

  def forAll[T1, T2](p: (T1, T2) => Boolean)(implicit a1: Arbitrary[T1], a2: Arbitrary[T2]): Prop =
    new GenProp2(a1.arbitrary, a2.arbitrary, p)

  def forAll[T1, T2, T3](p: (T1, T2, T3) => Boolean)(implicit a1: Arbitrary[T1], a2: Arbitrary[T2], a3: Arbitrary[T3]): Prop =
    new GenProp3(a1.arbitrary, a2.arbitrary, a3.arbitrary, p)

}
