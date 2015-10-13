package org.broadinstitute.k3.methods
import breeze.linalg._
import breeze.stats.mean

object LinRegBreeze {
  def main(args: Array[String]): Unit = {

    val X = DenseMatrix(
      (0.0, 1.0, 0.0, 0.0, 0.0),
      (1.0, 2.0, 1.0, 0.0, 1.0),
      (0.0, 1.0, 1.0, 0.0, 1.0),
      (0.0, 2.0, 1.0, 1.0, 2.0),
      (0.0, 0.0, 1.0, 1.0, 0.0),
      (1.0, 0.0, 2.0, 1.0, 0.0))

    val y = DenseVector(1.0, 1.0, 2.0, 2.0, 2.0, 2.0)

    val mu = mean(y)
    y :-= mu

    val C = DenseMatrix(
      ( 0.0, -1.0),
      ( 2.0,  3.0),
      ( 1.0,  5.0),
      (-2.0,  0.0),
      (-2.0, -4.0),
      ( 4.0,  3.0))

    val allOnes = DenseMatrix.ones[Double](6, 1)

    val C1 = DenseMatrix.horzcat(allOnes, C)

    def standardizeInPlace(m: DenseMatrix[Double]) = {
      val mu = mean(m(::, *))
      for (j <- 0 until m.cols)
        m(::, j) :-= mu(0, j)
      val nm = norm(m(::, *))
      for (j <- 0 until m.cols)
        m(::, j) :/= nm(0, j)
    }

    standardizeInPlace(X)

    println(X)

    var i = 0
    for (i <- 0 until X.cols) {
      val Xi = X(::, i to i)
      val XiC = DenseMatrix.horzcat(allOnes, Xi, C)
      println(XiC \ y)
    }

    val Q = qr.reduced.justQ(C1)

    val yp = y - Q * (Q.t * y)
    //yp :/= norm(yp)

    val Xp = X - Q * (Q.t * X)

    for (i <- 0 until Xp.cols) {
      val Xpi = Xp(::, i to i).copy
      val b = Xpi \ yp
      println(b(0))
      val Xpi2 = Xp(::, i)
      val b2 = (yp dot Xpi2) / (Xpi2 dot Xpi2)
      println(b2)
    }

  }
}

/*
  Breeze bug
  val X = DenseMatrix((1.0, 0.0), (0.0, 1.0))
  val y = DenseVector(0.0, 1.0)
  val X0 = X(::, 0).toDenseMatrix.t
  val b = X \ y
  println(X0.majorStride)

val X0 = X(::, 0 to 0)
val b0 = X(::, 0 to 0) \ y
println(X0.offset)
println("b0")
println(b0)

val X1 = X(::, 1 to 1).copy
println(X1)
println(X1.offset)
val b1 = X1 \ y
println("b1")
println(b1)
*/