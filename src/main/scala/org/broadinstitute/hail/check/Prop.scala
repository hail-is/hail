package org.broadinstitute.hail.check

import org.apache.commons.math3.random.RandomDataGenerator
import scala.util.Random

abstract class Prop {
  def apply(p: Parameters, name: Option[String] = None): Unit
}

class GenProp1[T1](g1: Gen[T1], f: (T1) => Boolean) extends Prop {
  override def apply(p: Parameters, name: Option[String]) {
    val prefix = name.map(_ + ": ").getOrElse("")
    for (i <- 0 until p.count) {
      val v1 = g1(p)
      val r = f(v1)
      if (!r) {
        println(s"""! ${ prefix }Falsified after $i passed tests.""")
        println(s"> ARG_0: $v1")
        assert(r)
      }
    }
    println(s" + ${ prefix }OK, passed ${ p.count } tests.")
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
        println(s"! ${ prefix }Falsified after $i passed tests.")
        println(s"> ARG_0: $v1")
        println(s"> ARG_1: $v2")
        assert(r)
      }
    }
    println(s" + ${ prefix }OK, passed ${ p.count } tests.")
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
        println(s"! ${ prefix }Falsified after $i passed tests.")
        println(s"> ARG_0: $v1")
        println(s"> ARG_1: $v2")
        println(s"> ARG_2: $v3")
        assert(r)
      }
    }
    println(s" + ${ prefix }OK, passed ${ p.count } tests.")
  }
}

object Prop {
  lazy val _seed: Int = {
    val seedStr = System.getProperty("check.seed")
    if (seedStr == null)
      1
    else if (seedStr == "random")
      Random.nextInt()
    else
      seedStr.toInt
  }

  def seed: Int = {
    println(s"check: seed = ${ _seed }")
    _seed
  }

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
