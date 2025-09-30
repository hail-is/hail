package is.hail.scalacheck

import is.hail.utils.roundWithConstantSum

import breeze.linalg.DenseMatrix
import org.scalacheck.{Arbitrary, Gen}
import org.scalacheck.Arbitrary._
import org.scalacheck.Gen._

private[scalacheck] trait ArbitraryDenseMatrixInstances {

  def genDenseMatrix(n: Int, m: Int): Gen[DenseMatrix[Double]] =
    for {
      lim <- sized(_ / (m * n))
      data <- resize(lim, containerOfN[Array, Double](m * n, arbitrary[Double]))
    } yield DenseMatrix.create(n, m, data)

  lazy val genSquareOfAreaAtMostSize: Gen[(Int, Int)] =
    genNCubeOfVolumeAtMostSize(2).map(x => (x(0), x(1)))

  lazy val genNonEmptySquareOfAreaAtMostSize: Gen[(Int, Int)] =
    genNonEmptyNCubeOfVolumeAtMostSize(2).map(x => (x(0), x(1)))

  def genNCubeOfVolumeAtMostSize(n: Int): Gen[Array[Int]] =
    sized(s => genNCubeOfVolumeAtMost(n, s))

  def genNonEmptyNCubeOfVolumeAtMostSize(n: Int): Gen[Array[Int]] =
    sized(s => genNCubeOfVolumeAtMost(n, s).map(_.map(x => if (x == 0) 1 else x)))

  def genNCubeOfVolumeAtMost(n: Int, size: Int, alpha: Int = 1): Gen[Array[Int]] =
    for {
      simplexVector <- dirichlet(Array.fill(n)(alpha.toDouble))
      sizeOfSum = math.log(size.toDouble)
    } yield roundWithConstantSum(simplexVector.map(_ * sizeOfSum)).map(i =>
      math.exp(i.toDouble).toInt
    )

  implicit lazy val arbDenseMatrix: Arbitrary[DenseMatrix[Double]] =
    genNonEmptySquareOfAreaAtMostSize flatMap { case (m, n) => genDenseMatrix(m, n) }

  lazy val genMultipliableDenseMatrices: Gen[(DenseMatrix[Double], DenseMatrix[Double])] =
    for {
      Array(rows, inner, columns) <- genNonEmptyNCubeOfVolumeAtMostSize(3)
      l <- genDenseMatrix(rows, inner)
      r <- genDenseMatrix(inner, columns)
    } yield (l, r)

}
