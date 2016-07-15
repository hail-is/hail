package org.broadinstitute.hail.check

import org.apache.commons.math3.random._

import scala.collection.generic.CanBuildFrom
import scala.collection.mutable
import scala.math.Numeric.Implicits._

object Parameters {
  val default = Parameters(new RandomDataGenerator(), 100, 100)
}

case class Parameters(rng: RandomDataGenerator, size: Int, count: Int) {

  def frequency(pass: Int, outOf: Int): Boolean = {
    assert(outOf > 0)
    rng.getRandomGenerator.nextInt(outOf) < pass
  }
}

object Gen {
  // utility
  def partition(rng: RandomDataGenerator, size: Int, parts: Int): Array[Int] = {
    if (parts == 0)
      return Array()

    val a = new Array[Int](parts)
    for (_ <- 0 until size) {
      val i = rng.getRandomGenerator.nextInt(parts)
      a(i) += 1
    }
    assert(a.sum == size)
    a
  }

  val printableChars = (0 to 127).map(_.toChar).filter(!_.isControl).toArray
  val identifierLeadingChars = (0 to 127).map(_.toChar)
    .filter(c => c == '_' || c.isLetter)
  val identifierChars = (0 to 127).map(_.toChar)
    .filter(c => c == '_' || c.isLetterOrDigit)

  def apply[T](gen: (Parameters) => T): Gen[T] = new Gen[T](gen)

  def const[T](x: T): Gen[T] = Gen { (p: Parameters) => x }

  def oneOfSeq[T](xs: Seq[T]): Gen[T] = {
    assert(xs.nonEmpty)
    Gen { (p: Parameters) =>
      xs(p.rng.getRandomGenerator.nextInt(xs.length))
    }
  }

  def oneOfGen[T](gs: Gen[T]*): Gen[T] = {
    assert(gs.nonEmpty)
    Gen { (p: Parameters) =>
      gs(p.rng.getRandomGenerator.nextInt(gs.length))(p)
    }
  }

  def oneOf[T](xs: T*): Gen[T] = oneOfSeq(xs)

  def choose(min: Int, max: Int): Gen[Int] = {
    assert(max >= min)
    Gen { (p: Parameters) => p.rng.nextInt(min, max) }
  }

  def choose(min: Long, max: Long): Gen[Long] = {
    assert(max >= min)
    Gen { (p: Parameters) => p.rng.nextLong(min, max) }
  }

  def choose(min: Double, max: Double): Gen[Double] = Gen { (p: Parameters) =>
    p.rng.nextUniform(min, max, true)
  }

  def shuffle[T](is: IndexedSeq[T]): Gen[IndexedSeq[T]] = {
    Gen { (p: Parameters) =>
      if (is.isEmpty)
        is
      else
        p.rng.nextPermutation(is.size, is.size).map(is)
    }
  }

  def chooseWithWeights(weights: Array[Double]): Gen[Int] =
    frequency(weights.zipWithIndex.map { case (w, i) => (w, Gen.const(i)) }: _*)

  def frequency[T, U](wxs: (T, Gen[U])*)(implicit ev: T => scala.math.Numeric[T]#Ops): Gen[U] = {
    assert(wxs.nonEmpty)

    val running = Array.fill[Double](wxs.length)(0d)
    for (i <- 1 until wxs.length) {
      val w = wxs(i - 1)._1.toDouble
      assert(w >= 0d)
      running(i) = running(i - 1) + w
    }

    val outOf = running.last + wxs.last._1.toDouble

    Gen { (p: Parameters) =>
      val v = p.rng.getRandomGenerator.nextDouble * outOf.toDouble
      val t = running.indexWhere(x => x >= v) - 1
      val j = if (t < 0) running.length - 1 else t
      assert(j >= 0 && j < wxs.length)
      assert(v >= running(j)
        && (j == wxs.length - 1 || v < running(j + 1)))
      wxs(j)._2(p)
    }
  }

  def subset[T](s: Set[T]): Gen[Set[T]] = Gen.parameterized { p =>
    Gen.choose(0.0, 1.0).map(cutoff =>
      s.filter(_ => p.rng.getRandomGenerator.nextDouble <= cutoff))
  }

  def sequence[C, T](gs: Traversable[Gen[T]])(implicit cbf: CanBuildFrom[Nothing, T, C]): Gen[C] =
    Gen { (p: Parameters) =>
      val b = cbf()
      gs.foreach { g => b += g(p) }
      b.result()
    }

  def buildableOf[C, T](g: Gen[T])(implicit cbf: CanBuildFrom[Nothing, T, C]): Gen[C] =
    Gen { (p: Parameters) =>
      val b = cbf()
      if (p.size == 0)
        b.result()
      else {
        val s = p.rng.getRandomGenerator.nextInt(p.size)
        val part = partition(p.rng, p.size, s)
        for (i <- 0 until s)
          b += g(p.copy(size = part(i)))
        b.result()
      }
    }

  def distinctBuildableOf[C, T](g: Gen[T])(implicit cbf: CanBuildFrom[Nothing, T, C]): Gen[C] =
    Gen { (p: Parameters) =>
      val b = cbf()
      if (p.size == 0)
        b.result()
      else {
        val s = p.rng.getRandomGenerator.nextInt(p.size)
        val part = partition(p.rng, p.size, s)
        val t = mutable.Set.empty[T]
        for (i <- 0 until s)
          t += g(p.copy(size = part(i)))
        b ++= t
        b.result()
      }
    }

  def buildableOfN[C, T](n: Int, g: Gen[T])(implicit cbf: CanBuildFrom[Nothing, T, C]): Gen[C] =
    Gen { (p: Parameters) =>
      val part = partition(p.rng, p.size, n)
      val b = cbf()
      for (i <- 0 until n)
        b += g(p.copy(size = part(i)))
      b.result()
    }

  def distinctBuildableOfN[C, T](n: Int, g: Gen[T])(implicit cbf: CanBuildFrom[Nothing, T, C]): Gen[C] =
    Gen { (p: Parameters) =>
      val part = partition(p.rng, p.size, n)
      val t: mutable.Set[T] = mutable.Set.empty[T]
      var i = 0
      while (i < n) {
        t += g(p.copy(size = part(i)))
        i = t.size
      }
      val b = cbf()
      b ++= t
      b.result()
    }

  def randomOneOf[T](rng: RandomDataGenerator, is: IndexedSeq[T]): T = {
    assert(is.nonEmpty)
    is(rng.getRandomGenerator.nextInt(is.length))
  }

  def identifier: Gen[String] = Gen { (p: Parameters) =>
    val s = 1 + p.rng.getRandomGenerator.nextInt(11)
    val b = new StringBuilder()
    b += randomOneOf(p.rng, identifierLeadingChars)
    for (_ <- 1 until s)
      b += randomOneOf(p.rng, identifierChars)
    b.result()
  }

  def arbString: Gen[String] = Gen { (p: Parameters) =>
    val s = p.rng.getRandomGenerator.nextInt(12)
    val b = new StringBuilder()
    for (i <- 0 until s)
      b += randomOneOf(p.rng, printableChars)
    b.result()
  }

  def option[T](g: Gen[T], someFraction: Double = 0.8): Gen[Option[T]] = Gen { (p: Parameters) =>
    if (p.rng.getRandomGenerator.nextDouble < someFraction)
      Some(g(p))
    else
      None
  }

  def nonnegInt: Gen[Int] = Gen { p =>
    p.rng.getRandomGenerator.nextInt() & Int.MaxValue
  }

  def posInt: Gen[Int] = Gen { (p: Parameters) =>
    p.rng.getRandomGenerator.nextInt(Int.MaxValue - 1) + 1
  }

  def arbBoolean: Gen[Boolean] = Gen { p =>
    p.rng.getRandomGenerator.nextBoolean()
  }

  def arbByte: Gen[Byte] = Gen { p => p.rng.getRandomGenerator.nextInt().toByte }

  def arbInt: Gen[Int] = Gen { p => p.rng.getRandomGenerator.nextInt() }

  def arbLong: Gen[Long] = Gen { p => p.rng.getRandomGenerator.nextLong() }

  def arbDouble: Gen[Double] = Gen { p =>
    p.rng.nextUniform(Double.MinValue, Double.MaxValue, true)
  }

  def zip[T1](g1: Gen[T1]): Gen[T1] = g1

  def zip[T1, T2](g1: Gen[T1], g2: Gen[T2]): Gen[(T1, T2)] = Gen { (p: Parameters) =>
    (g1(p), g2(p))
  }

  def zip[T1, T2, T3](g1: Gen[T1], g2: Gen[T2], g3: Gen[T3]): Gen[(T1, T2, T3)] = Gen { (p: Parameters) =>
    (g1(p), g2(p), g3(p))
  }

  def parameterized[T](f: (Parameters => Gen[T])) = Gen { p => f(p)(p) }

  def sized[T](f: (Int) => Gen[T]): Gen[T] = Gen { (p: Parameters) => f(p.size)(p) }

}

class Gen[+T](val gen: (Parameters) => T) extends AnyVal {

  def apply(p: Parameters): T = gen(p)

  def sample(): T = apply(Parameters.default)

  def map[U](f: (T) => U): Gen[U] = Gen { p => f(apply(p)) }

  def flatMap[U](f: (T) => Gen[U]): Gen[U] = Gen { p =>
    f(apply(p))(p)
  }

  def resize(newSize: Int): Gen[T] = Gen { (p: Parameters) =>
    apply(p.copy(size = newSize))
  }

  // FIXME should be non-strict
  def withFilter(f: (T) => Boolean): Gen[T] = Gen { (p: Parameters) =>
    var x = apply(p)
    while (!f(x))
      x = apply(p)
    x
  }

  def filter(f: (T) => Boolean): Gen[T] = withFilter(f)

}
