package is.hail.check

import scala.collection.mutable.ArrayBuffer
import scala.util.{Failure, Random, Success, Try}

import org.apache.commons.math3.random.RandomDataGenerator

abstract class Prop {
  def apply(p: Parameters, name: Option[String] = None): Unit

  def check(): Unit = {
    val size = System.getProperty("check.size", "1000").toInt
    val count = System.getProperty("check.count", "10").toInt

    println(s"check: size = $size, count = $count")

    val rng = new RandomDataGenerator()
    rng.reSeed(Prop.seed)
    apply(Parameters(rng, size, count))
  }
}

class GenProp1[T1](g1: Gen[T1], f: (T1) => Boolean) extends Prop {
  override def apply(p: Parameters, name: Option[String]): Unit = {
    val prefix = name.map(_ + ": ").getOrElse("")
    for (i <- 0 until p.count) {
      val v1 = g1(p)
      val r = Try(f(v1))
      r match {
        case Success(true) =>
        case Success(false) =>
          println(s"""! ${prefix}Falsified after $i passed tests.""")
          println(s"> ARG_0: $v1")
          throw new AssertionError(null)
        case Failure(e) =>
          println(s"""! ${prefix}Error after $i passed tests.""")
          println(s"> ARG_0: $v1")
          throw new AssertionError(e)
      }
    }
    println(s" + ${prefix}OK, passed ${p.count} tests.")
  }
}

class GenProp2[T1, T2](g1: Gen[T1], g2: Gen[T2], f: (T1, T2) => Boolean) extends Prop {
  override def apply(p: Parameters, name: Option[String]): Unit = {
    val prefix = name.map(_ + ": ").getOrElse("")
    for (i <- 0 until p.count) {
      val v1 = g1(p)
      val v2 = g2(p)
      val r = Try(f(v1, v2))
      r match {
        case Success(true) =>
        case Success(false) =>
          println(s"""! ${prefix}Falsified after $i passed tests.""")
          println(s"> ARG_0: $v1")
          throw new AssertionError(null)
        case Failure(e) =>
          println(s"""! ${prefix}Error after $i passed tests.""")
          println(s"> ARG_0: $v1")
          throw new AssertionError(e)
      }
    }
    println(s" + ${prefix}OK, passed ${p.count} tests.")
  }
}

class GenProp3[T1, T2, T3](g1: Gen[T1], g2: Gen[T2], g3: Gen[T3], f: (T1, T2, T3) => Boolean)
    extends Prop {
  override def apply(p: Parameters, name: Option[String]): Unit = {
    val prefix = name.map(_ + ": ").getOrElse("")
    for (i <- 0 until p.count) {
      val v1 = g1(p)
      val v2 = g2(p)
      val v3 = g3(p)
      val r = Try(f(v1, v2, v3))
      r match {
        case Success(true) =>
        case Success(false) =>
          println(s"""! ${prefix}Falsified after $i passed tests.""")
          println(s"> ARG_0: $v1")
          throw new AssertionError(null)
        case Failure(e) =>
          println(s"""! ${prefix}Error after $i passed tests.""")
          println(s"> ARG_0: $v1")
          throw new AssertionError(e)
      }
    }
    println(s" + ${prefix}OK, passed ${p.count} tests.")
  }
}

class Properties(val name: String) extends Prop {
  val properties = ArrayBuffer.empty[(String, Prop)]

  class PropertySpecifier {
    def update(propName: String, prop: Prop): Unit =
      properties += (name + "." + propName) -> prop
  }

  lazy val property = new PropertySpecifier

  override def apply(p: Parameters, prefix: Option[String]): Unit =
    for ((propName, prop) <- properties)
      prop.apply(p, prefix.map(_ + "." + propName).orElse(Some(propName)))

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
    println(s"check: seed = ${_seed}")
    _seed
  }

  def check(prop: Prop): Unit =
    prop.check()

  def forAll[T1](g1: Gen[Boolean]): Prop =
    new GenProp1(g1, identity[Boolean])

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

  def forAll[T1, T2, T3](
    p: (T1, T2, T3) => Boolean
  )(implicit
    a1: Arbitrary[T1],
    a2: Arbitrary[T2],
    a3: Arbitrary[T3],
  ): Prop =
    new GenProp3(a1.arbitrary, a2.arbitrary, a3.arbitrary, p)

}
