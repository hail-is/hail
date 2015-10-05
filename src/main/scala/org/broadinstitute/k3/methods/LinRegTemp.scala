package org.broadinstitute.k3.methods
import breeze.linalg._
import breeze.stats._
import breeze.numerics._

object LinRegTemp {
  def main(args: Array[String]): Unit = {

    val X = DenseMatrix((0.0, 1.0, 0.0, 0.0, 0.0), (1.0, 2.0, 1.0, 0.0, 1.0), (0.0, 1.0, 1.0, 0.0, 1.0))

    val y = DenseVector(0.0, 1.0, 1.0)

    val C = DenseMatrix((0.0, 1.0), (2.0, 3.0), (4.0, 5.0))

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
    println("C")
    println(C)

    var i = 0
    for (i <- 0 until X.cols) {
      println(X(::, i))
    }

    val Q = qr.reduced.justQ(C)

    println("Q")
    println(Q)

    val X00 = new DenseMatrix(3, 1, Array(0.0, 1.0, 0.0))
    println(X00)
    val X0 = X(::, 0).toDenseMatrix.t
    println(X0)
    println(X00.majorStride)
    println(X0 == X00)
    println(y)
    val beta00 = X00 \ y
    println(beta00)
    println(X0)
    println(y)
    val beta0 = X0 \ y
    println(y)
    println(beta0)


  }
}
