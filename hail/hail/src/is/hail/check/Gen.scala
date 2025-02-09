package is.hail.check

import is.hail.check.Arbitrary.arbitrary
import is.hail.utils.roundWithConstantSum

import scala.collection.generic.CanBuildFrom
import scala.collection.mutable
import scala.language.higherKinds
import scala.math.Numeric.Implicits._
import scala.reflect.ClassTag

import breeze.linalg.DenseMatrix
import breeze.storage.Zero
import org.apache.commons.math3.random._

object Parameters {
  val default = Parameters(new RandomDataGenerator(), 1000, 10)
}

case class Parameters(rng: RandomDataGenerator, size: Int, count: Int) {

  def frequency(pass: Int, outOf: Int): Boolean = {
    assert(outOf > 0)
    rng.getRandomGenerator.nextInt(outOf) < pass
  }
}

object Gen {

  val nonExtremeDouble: Gen[Double] = oneOfGen(
    oneOf(1e30, -1.0, -1e-30, 0.0, 1e-30, 1.0, 1e30),
    choose(-100.0, 100.0),
    choose(-1e150, 1e150),
  )

  def squareOfAreaAtMostSize: Gen[(Int, Int)] =
    nCubeOfVolumeAtMostSize(2).map(x => (x(0), x(1)))

  def nonEmptySquareOfAreaAtMostSize: Gen[(Int, Int)] =
    nonEmptyNCubeOfVolumeAtMostSize(2).map(x => (x(0), x(1)))

  def nCubeOfVolumeAtMostSize(n: Int): Gen[Array[Int]] =
    Gen((p: Parameters) => nCubeOfVolumeAtMost(p.rng, n, p.size))

  def nonEmptyNCubeOfVolumeAtMostSize(n: Int): Gen[Array[Int]] =
    Gen { (p: Parameters) =>
      nCubeOfVolumeAtMost(p.rng, n, p.size).map(x => if (x == 0) 1 else x).toArray
    }

  def partition[T](
    rng: RandomDataGenerator,
    size: T,
    parts: Int,
    f: (RandomDataGenerator, T) => T,
  )(implicit
    tn: Numeric[T],
    tct: ClassTag[T],
  ): Array[T] = {
    import tn.mkOrderingOps
    assert(
      size >= tn.zero,
      s"size must be greater than or equal to 0. Found $size. tn.zero=${tn.zero}.",
    )

    if (parts == 0)
      return Array()

    val a = Array.fill[T](parts)(tn.zero)
    var sizeAvail = size
    val nSuccesses = rng.getRandomGenerator.nextInt(parts) + 1

    for (i <- 0 until nSuccesses - 1) {
      val s = if (sizeAvail != tn.zero) f(rng, sizeAvail) else tn.zero
      a(i) = s
      sizeAvail -= s
    }

    a(nSuccesses - 1) = sizeAvail

    assert(a.sum == size)

    rng.nextPermutation(a.length, a.length).map(a)
  }

  def partition(rng: RandomDataGenerator, size: Int, parts: Int): Array[Int] =
    partition(rng, size, parts, (rng: RandomDataGenerator, avail: Int) => rng.nextInt(0, avail))

  /** Picks a number of bins, n, from a BetaBinomial(alpha, beta), then takes {@code size} balls and
    * places them into n bins according to a dirichlet-multinomial distribution with all alpha_i
    * equal to n.
    */
  def partitionBetaDirichlet(rng: RandomDataGenerator, size: Int, alpha: Double, beta: Double)
    : Array[Int] =
    partitionDirichlet(rng, size, sampleBetaBinomial(rng, size, alpha, beta))

  /** Takes {@code size} balls and places them into {@code parts} bins according to a
    * dirichlet-multinomial distribution with alpha_n equal to {@code parts} for all n. The outputs
    * of this function tend towards uniformly distributed balls, i.e. vectors close to the center of
    * the simplex in {@code parts} dimensions.
    */
  def partitionDirichlet(rng: RandomDataGenerator, size: Int, parts: Int): Array[Int] = {
    val simplexVector = sampleDirichlet(rng, Array.fill(parts)(parts.toDouble))
    roundWithConstantSum(simplexVector.map((x: Double) => x * size).toArray)
  }

  def nCubeOfVolumeAtMost(rng: RandomDataGenerator, n: Int, size: Int, alpha: Int = 1)
    : Array[Int] = {
    val sizeOfSum = math.log(size)
    val simplexVector = sampleDirichlet(rng, Array.fill(n)(alpha.toDouble))
    roundWithConstantSum(simplexVector.map((x: Double) => x * sizeOfSum).toArray)
      .map(x => math.exp(x).toInt).toArray
  }

  private def sampleDirichlet(rng: RandomDataGenerator, alpha: Array[Double]): Array[Double] = {
    val draws = alpha.map(rng.nextGamma(_, 1))
    val sum = draws.sum
    draws.map((x: Double) => x / sum).toArray
  }

  def partition(parts: Int, sum: Int): Gen[Array[Int]] =
    Gen { p =>
      partition(p.rng, sum, parts, (rng: RandomDataGenerator, avail: Int) => rng.nextInt(0, avail))
    }

  def partition(parts: Int, sum: Long): Gen[Array[Long]] =
    Gen { p =>
      partition(
        p.rng,
        sum,
        parts,
        (rng: RandomDataGenerator, avail: Long) => rng.nextLong(0, avail),
      )
    }

  def partition(parts: Int, sum: Double): Gen[Array[Double]] =
    Gen { p =>
      partition(
        p.rng,
        sum,
        parts,
        (rng: RandomDataGenerator, avail: Double) => rng.nextUniform(0, avail),
      )
    }

  def partitionSize(parts: Int): Gen[Array[Int]] =
    Gen(p => partitionDirichlet(p.rng, p.size, parts))

  def size: Gen[Int] = Gen(p => p.size)

  val printableChars = (0 to 127).map(_.toChar).filter(!_.isControl).toArray

  val identifierLeadingChars = (0 to 127).map(_.toChar)
    .filter(c => c == '_' || c.isLetter)

  val identifierChars = (0 to 127).map(_.toChar)
    .filter(c => c == '_' || c.isLetterOrDigit)

  val plinkSafeStartOfIdentifierChars = (0 to 127).map(_.toChar)
    .filter(c => c.isLetter)

  val plinkSafeChars = (0 to 127).map(_.toChar)
    .filter(c => c.isLetterOrDigit)

  def apply[T](gen: (Parameters) => T): Gen[T] = new Gen[T](gen)

  def const[T](x: T): Gen[T] = Gen((p: Parameters) => x)

  def coin(p: Double = 0.5): Gen[Boolean] = {
    require(0.0 < p)
    require(p < 1.0)
    choose(0.0, 1.0).map(_ <= p)
  }

  def oneOfSeq[T](xs: Seq[T]): Gen[T] = {
    assert(xs.nonEmpty)
    Gen((p: Parameters) => xs(p.rng.getRandomGenerator.nextInt(xs.length)))
  }

  def oneOfGen[T](gs: Gen[T]*): Gen[T] = {
    assert(gs.nonEmpty)
    Gen((p: Parameters) => gs(p.rng.getRandomGenerator.nextInt(gs.length))(p))
  }

  def oneOf[T](xs: T*): Gen[T] = oneOfSeq(xs)

  def choose(min: Int, max: Int): Gen[Int] = {
    assert(max >= min)
    Gen((p: Parameters) => p.rng.nextInt(min, max))
  }

  def choose(min: Long, max: Long): Gen[Long] = {
    assert(max >= min)
    Gen((p: Parameters) => p.rng.nextLong(min, max))
  }

  def choose(min: Float, max: Float): Gen[Float] = Gen { (p: Parameters) =>
    p.rng.nextUniform(min, max, true).toFloat
  }

  def choose(min: Double, max: Double): Gen[Double] = Gen { (p: Parameters) =>
    p.rng.nextUniform(min, max, true)
  }

  def gaussian(mu: Double, sigma: Double): Gen[Double] = Gen { (p: Parameters) =>
    p.rng.nextGaussian(mu, sigma)
  }

  def nextBeta(alpha: Double, beta: Double): Gen[Double] = Gen { (p: Parameters) =>
    p.rng.nextBeta(alpha, beta)
  }

  def nextCoin(p: Double) =
    choose(0.0, 1.0).map(_ < p)

  private def sampleBetaBinomial(rng: RandomDataGenerator, n: Int, alpha: Double, beta: Double)
    : Int =
    rng.nextBinomial(n, rng.nextBeta(alpha, beta))

  def nextBetaBinomial(n: Int, alpha: Double, beta: Double): Gen[Int] = Gen { p =>
    sampleBetaBinomial(p.rng, n, alpha, beta)
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

  def frequency[T, U](wxs: (T, Gen[U])*)(implicit ev: scala.math.Numeric[T]): Gen[U] = {
    import Numeric.Implicits._

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
      s.filter(_ => p.rng.getRandomGenerator.nextDouble <= cutoff)
    )
  }

  def sequence[C[_], T](gs: Traversable[Gen[T]])(implicit cbf: CanBuildFrom[Nothing, T, C[T]])
    : Gen[C[T]] =
    Gen { (p: Parameters) =>
      val b = cbf()
      gs.foreach(g => b += g(p))
      b.result()
    }

  def denseMatrix[T: ClassTag: Zero: Arbitrary](): Gen[DenseMatrix[T]] = for {
    (l, w) <- Gen.nonEmptySquareOfAreaAtMostSize
    m <- denseMatrix(l, w)
  } yield m

  def denseMatrix[T: ClassTag: Zero: Arbitrary](n: Int, m: Int): Gen[DenseMatrix[T]] =
    denseMatrix[T](n, m, arbitrary[T])

  def denseMatrix[T: ClassTag: Zero](n: Int, m: Int, g: Gen[T]): Gen[DenseMatrix[T]] =
    Gen((p: Parameters) => DenseMatrix.fill[T](n, m)(g.resize(p.size / (n * m))(p)))

  def twoMultipliableDenseMatrices[T: ClassTag: Zero: Arbitrary]()
    : Gen[(DenseMatrix[T], DenseMatrix[T])] =
    twoMultipliableDenseMatrices(arbitrary[T])

  def twoMultipliableDenseMatrices[T: ClassTag: Zero](g: Gen[T])
    : Gen[(DenseMatrix[T], DenseMatrix[T])] = for {
    Array(rows, inner, columns) <- Gen.nonEmptyNCubeOfVolumeAtMostSize(3)
    l <- denseMatrix(rows, inner, g)
    r <- denseMatrix(inner, columns, g)
  } yield (l, r)

  /** In general, for any Traversable type T and any Monad M, we may convert an {@code F[M[T]]} to
    * an {@code M[F[T]]} by choosing to perform the actions in the order defined by the traversable.
    * With {@code Gen} we must also consider the distribution of size. {@code uniformSequence}
    * distributes the size uniformly across all elements of the traversable.
    */
  def uniformSequence[C[_], T](
    gs: Traversable[Gen[T]]
  )(implicit cbf: CanBuildFrom[Nothing, T, C[T]]
  ): Gen[C[T]] =
    partitionSize(gs.size).map(resizeMany(gs, _)).flatMap(sequence[C, T])

  private def resizeMany[T](gs: Traversable[Gen[T]], partition: Array[Int]): Iterable[Gen[T]] =
    (gs.toIterable, partition).zipped.map((gen, size) => gen.resize(size))

  def stringOf[T](g: Gen[T])(implicit cbf: CanBuildFrom[Nothing, T, String]): Gen[String] =
    unsafeBuildableOf(g)

  sealed trait BuildableOf[C[_]] {
    def apply[T](g: Gen[T])(implicit cbf: CanBuildFrom[Nothing, T, C[T]]): Gen[C[T]] =
      unsafeBuildableOf(g)
  }

  private object buildableOfInstance extends BuildableOf[Nothing]

  def buildableOf[C[_]] = buildableOfInstance.asInstanceOf[BuildableOf[C]]

  implicit def buildableOfFromElements[C[_], T](
    implicit g: Gen[T],
    cbf: CanBuildFrom[Nothing, T, C[T]],
  ): Gen[C[T]] =
    buildableOf[C](g)

  sealed trait BuildableOf2[C[_, _]] {
    def apply[T, U](g: Gen[(T, U)])(implicit cbf: CanBuildFrom[Nothing, (T, U), C[T, U]])
      : Gen[C[T, U]] =
      unsafeBuildableOf(g)
  }

  private object buildableOf2Instance extends BuildableOf2[Nothing]

  def buildableOf2[C[_, _]] = buildableOf2Instance.asInstanceOf[BuildableOf2[C]]

  private val buildableOfAlpha = 3
  private val buildableOfBeta = 6

  private def unsafeBuildableOf[C, T](g: Gen[T])(implicit cbf: CanBuildFrom[Nothing, T, C])
    : Gen[C] =
    Gen { (p: Parameters) =>
      val b = cbf()
      if (p.size == 0)
        b.result()
      else {
        // scale up a bit by log, so that we can spread out a bit more with
        // higher sizes
        val part = partitionBetaDirichlet(
          p.rng,
          p.size,
          buildableOfAlpha,
          buildableOfBeta * math.log(p.size + 0.01),
        )
        val s = part.length
        for (i <- 0 until s)
          b += g(p.copy(size = part(i)))
        b.result()
      }
    }

  sealed trait DistinctBuildableOf[C[_]] {
    def apply[T](g: Gen[T])(implicit cbf: CanBuildFrom[Nothing, T, C[T]]): Gen[C[T]] =
      Gen { (p: Parameters) =>
        val b = cbf()
        if (p.size == 0)
          b.result()
        else {
          // scale up a bit by log, so that we can spread out a bit more with
          // higher sizes
          val part = partitionBetaDirichlet(
            p.rng,
            p.size,
            buildableOfAlpha,
            buildableOfBeta * math.log(p.size + 0.01),
          )
          val s = part.length
          val t = mutable.Set.empty[T]
          for (i <- 0 until s)
            t += g(p.copy(size = part(i)))
          b ++= t
          b.result()
        }
      }
  }

  private object distinctBuildableOfInstance extends DistinctBuildableOf[Nothing]

  def distinctBuildableOf[C[_]] = distinctBuildableOfInstance.asInstanceOf[DistinctBuildableOf[C]]

  /** This function terminates with probability equal to the probability of {@code g} generating
    * {@code min} distinct elements in finite time.
    */
  sealed trait DistinctBuildableOfAtLeast[C[_]] {
    def apply[T](min: Int, g: Gen[T])(implicit cbf: CanBuildFrom[Nothing, T, C[T]]): Gen[C[T]] = {
      Gen { (p: Parameters) =>
        val b = cbf()
        if (p.size < min) {
          throw new RuntimeException(
            s"Size (${p.size}) is too small for buildable of size at least $min"
          )
        } else if (p.size == 0)
          b.result()
        else {
          // scale up a bit by log, so that we can spread out a bit more with
          // higher sizes
          val s = min + sampleBetaBinomial(
            p.rng,
            p.size - min,
            buildableOfAlpha,
            buildableOfBeta * math.log((p.size - min) + 0.01),
          )
          val part = partitionDirichlet(p.rng, p.size, s)
          val t = mutable.Set.empty[T]
          for (i <- 0 until s) {
            var element = g.resize(part(i))(p)
            while (t.contains(element))
              element = g.resize(part(i))(p)
            t += element
          }
          b ++= t
          b.result()
        }
      }
    }
  }

  private object distinctBuildableOfAtLeastInstance extends DistinctBuildableOfAtLeast[Nothing]

  def distinctBuildableOfAtLeast[C[_]] =
    distinctBuildableOfAtLeastInstance.asInstanceOf[DistinctBuildableOfAtLeast[C]]

  sealed trait BuildableOfN[C[_]] {
    def apply[T](n: Int, g: Gen[T])(implicit cbf: CanBuildFrom[Nothing, T, C[T]]): Gen[C[T]] =
      Gen { (p: Parameters) =>
        val part = partitionDirichlet(p.rng, p.size, n)
        val b = cbf()
        for (i <- 0 until n)
          b += g(p.copy(size = part(i)))
        b.result()
      }
  }

  private object buildableOfNInstance extends BuildableOfN[Nothing]

  def buildableOfN[C[_]] = buildableOfNInstance.asInstanceOf[BuildableOfN[C]]

  sealed trait DistinctBuildableOfN[C[_]] {
    def apply[T](n: Int, g: Gen[T])(implicit cbf: CanBuildFrom[Nothing, T, C[T]]): Gen[C[T]] =
      Gen { (p: Parameters) =>
        val part = partitionDirichlet(p.rng, p.size, n)
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
  }

  private object distinctBuildableOfNInstance extends DistinctBuildableOfN[Nothing]

  def distinctBuildableOfN[C[_]] =
    distinctBuildableOfNInstance.asInstanceOf[DistinctBuildableOfN[C]]

  def randomOneOf[T](rng: RandomDataGenerator, is: IndexedSeq[T]): T = {
    assert(is.nonEmpty)
    is(rng.getRandomGenerator.nextInt(is.length))
  }

  def identifier: Gen[String] =
    identifierGen(identifierLeadingChars, identifierChars)

  def plinkSafeIdentifier: Gen[String] =
    identifierGen(plinkSafeStartOfIdentifierChars, plinkSafeChars)

  private def identifierGen(
    leadingCharacter: IndexedSeq[Char],
    trailingCharacters: IndexedSeq[Char],
  ): Gen[String] = Gen { p =>
    val s = 1 + p.rng.getRandomGenerator.nextInt(11)
    val b = new StringBuilder()
    b += randomOneOf(p.rng, leadingCharacter)
    for (_ <- 1 until s)
      b += randomOneOf(p.rng, trailingCharacters)
    b.result()
  }

  def option[T](g: Gen[T], someFraction: Double = 0.8): Gen[Option[T]] = Gen { (p: Parameters) =>
    if (p.rng.getRandomGenerator.nextDouble < someFraction)
      Some(g(p))
    else
      None
  }

  def nonnegInt: Gen[Int] = Gen(p => p.rng.getRandomGenerator.nextInt() & Int.MaxValue)

  def posInt: Gen[Int] = Gen { (p: Parameters) =>
    p.rng.getRandomGenerator.nextInt(Int.MaxValue - 1) + 1
  }

  def interestingPosInt: Gen[Int] = oneOfGen(
    oneOf(1, 2, Int.MaxValue - 1, Int.MaxValue),
    choose(1, 100),
    posInt,
  )

  def zip[T1](g1: Gen[T1]): Gen[T1] = g1

  def zip[T1, T2](g1: Gen[T1], g2: Gen[T2]): Gen[(T1, T2)] = for {
    Array(s1, s2) <- partitionSize(2)
    x <- g1.resize(s1)
    y <- g2.resize(s2)
  } yield (x, y)

  def zip[T1, T2, T3](g1: Gen[T1], g2: Gen[T2], g3: Gen[T3]): Gen[(T1, T2, T3)] = for {
    Array(s1, s2, s3) <- partitionSize(3)
    x <- g1.resize(s1)
    y <- g2.resize(s2)
    z <- g3.resize(s3)
  } yield (x, y, z)

  def zip[T1, T2, T3, T4](g1: Gen[T1], g2: Gen[T2], g3: Gen[T3], g4: Gen[T4])
    : Gen[(T1, T2, T3, T4)] = for {
    Array(s1, s2, s3, s4) <- partitionSize(4)
    x <- g1.resize(s1)
    y <- g2.resize(s2)
    z <- g3.resize(s3)
    w <- g4.resize(s4)
  } yield (x, y, z, w)

  def parameterized[T](f: (Parameters => Gen[T])) = Gen(p => f(p)(p))

  def sized[T](f: (Int) => Gen[T]): Gen[T] = Gen((p: Parameters) => f(p.size)(p))

  def applyGen[T, S](gf: Gen[(T) => S], gx: Gen[T]): Gen[S] = Gen { p =>
    val f = gf(p)
    val x = gx(p)
    f(x)
  }
}

class Gen[+T](val gen: (Parameters) => T) extends AnyVal {

  def apply(p: Parameters): T = gen(p)

  def sample(): T = apply(Parameters.default)

  def map[U](f: (T) => U): Gen[U] = Gen(p => f(apply(p)))

  def flatMap[U](f: (T) => Gen[U]): Gen[U] = Gen(p => f(apply(p))(p))

  def resize(newSize: Int): Gen[T] = Gen((p: Parameters) => apply(p.copy(size = newSize)))

  // FIXME should be non-strict
  def withFilter(f: (T) => Boolean): Gen[T] = Gen { (p: Parameters) =>
    var x = apply(p)
    var i = 0
    while (!f(x)) {
      assert(i < 100)
      x = apply(p)
      i += 1
    }
    x
  }

  def filter(f: (T) => Boolean): Gen[T] = withFilter(f)
}
