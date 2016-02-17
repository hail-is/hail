package org.broadinstitute.hail.check

import scala.collection.generic.CanBuildFrom
import scala.collection.mutable
import org.apache.commons.math3.random._

object Parameters {
  val default = new Parameters(new RandomDataGenerator(), 100)
}

class Parameters(val rng: RandomDataGenerator, val size: Int) {
  def frequency(pass: Int, outOf: Int): Boolean =
    rng.getRandomGenerator.nextInt(outOf) < pass
}

object Gen {
  val printableChars = (0 to 127).map(_.toChar).filter(!_.isControl).toArray
  val identifierLeadingChars = (0 to 127).map(_.toChar)
    .filter(c => c == '_' || c.isLetter)
  val identifierChars = (0 to 127).map(_.toChar)
    .filter(c => c == '_' || c.isLetterOrDigit)

  def apply[T](gen: (Parameters) => T): Gen[T] = new Gen[T](gen)

  def const[T](x: T): Gen[T] = Gen { (p: Parameters) => x }

  def oneOfSeq[T](xs: Seq[T]): Gen[T] = Gen { (p: Parameters) =>
    xs(p.rng.getRandomGenerator.nextInt(xs.length))
  }

  def oneOfGen[T](gs: Gen[T]*): Gen[T] = Gen { (p: Parameters) =>
    gs(p.rng.getRandomGenerator.nextInt(gs.length))(p)
  }

  def oneOf[T](xs: T*): Gen[T] = oneOfSeq(xs)

  def choose(min: Int, max: Int): Gen[Int] = Gen { (p: Parameters) => p.rng.nextInt(min, max) }

  def choose(min: Long, max: Long): Gen[Long] = Gen { (p: Parameters) => p.rng.nextLong(min, max) }

  def choose(min: Double, max: Double): Gen[Double] = Gen { (p: Parameters) =>
    p.rng.nextUniform(min, max, true)
  }

  def frequency[T](wxs: (Int, Gen[T])*): Gen[T] = {
    val running = new Array[Int](wxs.length)
    running(0) = 0
    for (i <- 1 until wxs.length)
      running(i) = running(i - 1) + wxs(i - 1)._1

    val outOf = running.last + wxs.last._1

    Gen { (p: Parameters) =>
      val v = p.rng.getRandomGenerator.nextInt(outOf)
      val t = java.util.Arrays.binarySearch(running, v)
      val j = if (t < 0) -t - 2 else t
      assert(j >= 0 && j < wxs.length)
      assert(v >= running(j)
        && (j == wxs.length - 1 || v < running(j + 1)))
      wxs(j)._2(p)
    }
  }

  def sequence[C, T](gs: Traversable[Gen[T]])(implicit cbf: CanBuildFrom[Nothing, T, C]): Gen[C] =
    Gen { (p: Parameters) =>
      val b = cbf()
      gs.foreach { g => b += g(p) }
      b.result()
    }

  def buildableOf[C, T](g: Gen[T])(implicit cbf: CanBuildFrom[Nothing, T, C]): Gen[C] =
    Gen { (p: Parameters) =>
      val s: Int = p.rng.getRandomGenerator.nextInt(p.size)
      val b = cbf()
      for (_ <- 0 until s)
        b += g(p)
      b.result()
    }

  def distinctBuildableOf[C, T](g: Gen[T])(implicit cbf: CanBuildFrom[Nothing, T, C]): Gen[C] =
    Gen { (p: Parameters) =>
      val s: Int = p.rng.getRandomGenerator.nextInt(p.size)
      val t: mutable.Set[T] = mutable.Set.empty[T]
      for (_ <- 0 until s)
        t += g(p)
      val b = cbf()
      b ++= t
      b.result()
    }

  def buildableOfN[C, T](n: Int, g: Gen[T])(implicit cbf: CanBuildFrom[Nothing, T, C]): Gen[C] =
    Gen { (p: Parameters) =>
      val b = cbf()
      for (_ <- 0 until n)
        b += g(p)
      b.result()
    }

  def distinctBuildableOfN[C, T](n: Int, g: Gen[T])(implicit cbf: CanBuildFrom[Nothing, T, C]): Gen[C] =
    Gen { (p: Parameters) =>
      val t: mutable.Set[T] = mutable.Set.empty[T]
      while (t.size < n)
        t += g(p)
      val b = cbf()
      b ++= t
      b.result()
    }

  def randomOneOf[T](rng: RandomDataGenerator, is: IndexedSeq[T]): T =
    is(rng.getRandomGenerator.nextInt(is.length))

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
}

class Gen[T](val gen: (Parameters) => T) extends AnyVal {

  def apply(p: Parameters): T = gen(p)

  def sample(): T = apply(Parameters.default)

  def map[U](f: (T) => U): Gen[U] = Gen { p => f(apply(p)) }

  def flatMap[U](f: (T) => Gen[U]): Gen[U] = Gen { p =>
    f(apply(p))(p)
  }

  // FIXME should be non-strict
  def withFilter(f: (T) => Boolean): Gen[T] = Gen { (p: Parameters) =>
    var x = apply(p)
    while (!f(x))
      x = apply(p)
    x
  }

  def filter(f: (T) => Boolean): Gen[T] = withFilter(f)

  def parameterized(f: (Parameters => Gen[T])) = Gen { p => f(p)(p) }

  def sized(f: (Int) => Gen[T]): Gen[T] = Gen { (p: Parameters) => f(p.size)(p) }
}
