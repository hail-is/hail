package org.broadinstitute.hail.methods

import org.apache.spark.mllib.linalg.{Matrix, Matrices}

/* Implements Ward's method, an agglomerative clustering algorithm that, at 
 * each step, merges clusters so as to minimize the increase in within-cluster 
 * sum-of-squares error.
 * 
 * Implements it as a recursive nearest neighbor (RNN) algorithm with Lance-
 * Williams updating, as described here:
 * https://arxiv.org/pdf/1105.0121.pdf
 *
 * Clustering n points on R^m takes roughly O(n^2 * m) to calculate the dist
 * matrix, plus O(n^2) for the clustering.
 */

/** It is assumed that rowcount(M) >= k.  The rows of M are treated as points
  * to be clustered
  */
class Ward() {
  def name = "Ward"
  
  // perform's Ward's clustering on distance Seq[Seq[Double]]
  // like the one returned by Ward.distMat
  def apply(D : Seq[Seq[Double]],k : Int) : Seq[Set[Int]] = { 
    val nPoints = D.size + 1
    val H = hier(D,Vector.fill(nPoints)(1),(0 until nPoints).toSet)
    cut(H,k,nPoints).values.toVector
  }

  def pointDist (M : Matrix, x : Int, y : Int) : Double = {
    def square (x : Double) : Double = x * x
    ( (0 until M.numCols) map ((i : Int) => square(M(x,i) - M(y,i))) ).sum
  }
    

  // calculates distance seq seq D such that
  // S[i][j] = dist(M[i],M[i + 1 + j])^2
  // [[d(0,1),d(0,2),d(0,3)],
  //  [d(1,2),d(1,3)],
  //  [d(2,3) ]
  def distMat(M : Matrix) : Seq[Seq[Double]] = { 
    val applyTo = (0 until (M.numRows - 1)) map ((i : Int) => (((i+1) until M.numRows) map ((j : Int) => (i,j))))
    applyTo map ((S : Seq[(Int,Int)]) => S map { case (i,j) => pointDist(M,i,j) } )
  }

  // Finds NN of cluster c
  def NN (D : Seq[Seq[Double]], c : Int, remain : Set[Int]) : (Double, Int) = {
    val neighbors = remain map { (i : Int) => (
      if (i < c) D(i)(c - (i+1)) else 
        (if (i == c) -1 else D(c)(i - (c + 1))),i)
    }
    val (min,minidx) = neighbors reduceLeft { (l : (Double,Int),r : (Double,Int)) => (l,r) match {
      case ((-1,_),_) => r
      case (_,(-1,_)) => l
      case ((i,_),(j,_)) => if (i > j) r else l
      }}
    (min,minidx)
  }

  // Finds an arbitrary RNN starting on cluster c
  def RNN (D : Seq[Seq[Double]], c : Int, remain : Set[Int]) : (Int, Int, Double) = { 
    val (min,minidx) = NN(D,c,remain)
    val (min2,minidx2) = NN(D,minidx,remain)
    if (minidx2 == c | min == min2) (Math.min(c,minidx),Math.max(c,minidx),min)
    else RNN(D,minidx,remain)
  }
 
  // Calculates LW formula w/ Ward coefficients for dist between a union b and i
  def LW(D : Seq[Seq[Double]],a : Int,b : Int,i : Int,n : Seq[Int]) = {
    val d_ab = if (a > b) D(b)(a - (b + 1)) else D(a)(b - (a + 1))
    val d_ai = if (a > i) D(i)(a - (i + 1)) else D(a)(i - (a + 1))
    val d_bi = if (i > b) D(b)(i - (b + 1)) else D(i)(b - (i + 1))
    val ntot = (n(a) + n(b) + n(i)).toDouble
    ( (n(a) + n(i)) * d_ai / ntot ) + ( (n(b) + n(i)) * d_bi / ntot ) - ( n(i) * d_ab / ntot )
  }

  // Uses the Lance-Williams formula to update distances adding clust b to
  // clust a
  def LWupdate(D : Seq[Seq[Double]], a : Int, b : Int, n : Seq[Int],remain : Set[Int]) : Seq[Seq[Double]] = { 
  // Warning: mutable data structure!
  val A = D.toArray
  val _ = remain.foreach( (i : Int) => if (i == a) Unit else 
    (if (i > a) A(a) = A(a) updated (i - (a + 1),LW(D,a,b,i,n)) else A(i) = A(i) updated (a - (i + 1),LW(D,a,b,i,n)) ) )
  A.toVector
  }
 
  // Uses RNN algorithm to build a hierarchy
  def hier(D : Seq[Seq[Double]], n : Seq[Int], remain : Set[Int]) : Set[(Int,Int,Double)] = {
    if (remain.size < 2) Set() : Set[(Int,Int,Double)] else {
      val (a,b,d) = RNN(D,0,remain) // Start RNN finding with clust 0, as it's always extant
      val remain_new = remain - b
      val D_new = LWupdate(D,a,b,n,remain_new)
      val n_new = n updated (a,n(a) + n(b))
      val merge = (a,b,d)
      (hier(D_new,n_new,remain_new)) + merge }
  }

  def makeClusters(joins : Set[(Int,Int,Double)], n : Int) : Map[Int,Set[Int]] = { 
    val mappedJoins = (joins map { case(a,b,_) => (b,a) }).toMap
    // Warning: code develops side effects here!
    val A = (0 until n).toArray
    val _ = (0 until n).foreach( (i : Int) => A(i) = A(mappedJoins getOrElse (i,i)) )
    // Ok, functional again
    A.zipWithIndex.foldLeft (Map(0 -> Set(0))) ((M : Map[Int,Set[Int]],p : (Int,Int)) =>
      p match { case (c,i) => if (M contains c) M + (c -> (M(c) + i)) else M + (c -> Set(i)) } )
  }
  
  def maxJoin(joins : Set[(Int,Int,Double)]) = { 
    joins reduceLeft ((l : (Int,Int,Double),r : (Int,Int,Double)) =>
                      (l,r) match { case ((_,_,a),(_,_,b)) => if (a > b) l else r })
  }
  
  // Cuts a hierarchy H on n points into k clusters
  def cut(H : Set[(Int,Int,Double)],k : Int,n : Int) : Map[Int,Set[Int]] = { 
    if (k <= 1) makeClusters(H,n) else (
      cut(H - maxJoin(H),k-1,n)
    )  
  } 
  
}
