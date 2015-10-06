package org.broadinstitute.k3.methods
import breeze.linalg._
import breeze.stats._
import breeze.numerics._

object LinRegBreeze {
  def main(args: Array[String]): Unit = {

    val X = DenseMatrix(
      (0.0, 1.0, 0.0, 0.0, 0.0),
      (1.0, 2.0, 1.0, 0.0, 1.0),
      (0.0, 1.0, 1.0, 0.0, 1.0))

    val y = DenseVector(0.0, 1.0, 1.0)

    val C = DenseMatrix(
      (0.0, 1.0),
      (2.0, 3.0),
      (4.0, 5.0))

    val mv = meanAndVariance(X(::, *))
    val mu = mv.map(_.mean).toDenseVector
    val sigma = mv.map(_.stdDev).toDenseVector
    val Xstd = ((X(*, ::) - mu): DenseMatrix[Double])(*,::) :/ sigma

    println("X")
    println(X)
    println("Xstd")
    println(Xstd)
    println("y")
    println(y)

    val b = X \ y
    println("b")
    println(b)

    println("C")
    println(C)

    var i = 0
    for (i <- 0 until X.cols) {
      println(X(::, i))
    }
    val Q = qr.reduced.justQ(C)

    println("Q")
    println(Q)


  }
}

//  Breeze bug
//  val X = DenseMatrix((1.0, 0.0), (0.0, 1.0))
//  val y = DenseVector(0.0, 1.0)
//  val X0 = X(::, 0).toDenseMatrix.t
//  val b = X \ y
//  println(X0.majorStride)
