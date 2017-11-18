package is.hail.methods

import breeze.linalg.{DenseMatrix => BDM, _}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._
import is.hail.variant.{Genotype, Locus, Variant, VariantDataset}

object PCRelateReferenceImplementation {
  def apply(vds: VariantDataset, pcs: DenseMatrix, maf: Double = 0.0): (Map[(String, String), (Double, Double, Double, Double)], BDM[Double], BDM[Double], BDM[Double]) = {
    val indexToId: Map[Int, String] = vds.stringSampleIds.zipWithIndex.map { case (id, index) => (index, id) }.toMap

    val gts = vds.typedRDD[Locus, Variant, Genotype].zipWithIndex.map { case ((v, (va, gs)), i) =>
      val a = gs.map(Genotype.gt _).toArray
      val mean = a.flatten.sum.toDouble / a.flatten.length
      (i, a.map { case Some(v) => v.toDouble ; case None => mean }.toArray)
    }.collect().sortBy(_._1).map(_._2).flatten

    val mat = new BDM[Double](vds.nSamples, vds.countVariants().toInt, gts)

    val (foo, ibs0, mu_si) = forMatrices(mat, new BDM[Double](pcs.numRows, pcs.numCols, pcs.toArray), maf=0.01)

    val PCRelate.Result(phi, k0, k1, k2) = foo.map(symmetricMatrixToMap(indexToId,_))

    (phi.keys.map(k => (k, (phi(k), k0(k), k1(k), k2(k)))).toMap, mat, ibs0, mu_si)
  }

  // keys.length == mat.rows == mat.cols
  private def symmetricMatrixToMap[T,U](keys: Int => T, mat: BDM[U]): Map[(T, T), U] = (for {
    i <- 0 until mat.rows
    j <- (i+1) until mat.cols
  } yield ((keys(i), keys(j)), mat(i, j))).toMap

  // g : N x M
  // pcs : N x K
  def forMatrices(g: BDM[Double], pcs: BDM[Double], maf: Double = 0.0): (PCRelate.Result[BDM[Double]], BDM[Double], BDM[Double]) = {
    val n = g.rows
    val m = g.cols
    require(n == pcs.rows)
    val k = pcs.cols

    // N x (K + 1)
    val pcsWithIntercept = BDM.horzcat(new BDM(n, 1, Array.fill[Double](n)(1.0)), pcs)

    // beta : M x K
    val beta = (inv(pcsWithIntercept.t * pcsWithIntercept) * pcsWithIntercept.t * g).t

    // mu_si : M x N
    val mu_si: BDM[Double] = (beta * pcsWithIntercept.t) / 2.0

    def goodMu(mu: Double): Boolean =
      (mu > 0.0 && mu > maf &&
       mu < 1.0 && mu < (1.0 - maf))
    def goodGT(gt: Double): Boolean =
      gt == 0.0 || gt == 1.0 || gt == 2.0

    // g2mu : M x N
    // the if eliminates contribution from bad mu and bad genotype entries
    val g2mua = new Array[Double](m*n)
    var i = 0
    while (i < n) {
      var j = 0
      while (j < m) {
        val gt = g(i,j)
        val mu = mu_si(j,i)

        if (goodMu(mu) && goodGT(gt))
          g2mua(i*m + j) = gt - (2.0 * mu)

        j += 1
      }
      i += 1
    }
    val g2mu = new BDM[Double](m, n, g2mua)

    val numer = g2mu.t * g2mu

    val stddeva = new Array[Double](m*n)
    i = 0
    while (i < n) {
      var j = 0
      while (j < m) {
        val gt = g(i,j)
        val mu = mu_si(j,i)

        if (goodMu(mu) && goodGT(gt))
          stddeva(i*m + j) = math.sqrt(mu * (1.0 - mu))

        j += 1
      }
      i += 1
    }
    val stddev = new BDM[Double](m, n, stddeva)

    val denom = (stddev.t * stddev)
    val phi = (numer :/ denom) / 4.0

    def toDom(gt: Double, mu: Double): Double = gt match {
      case 0.0 => mu
      case 1.0 => 0.0
      case 2.0 => 1.0 - mu
      case _   => Double.NaN
    }

    val k2a = new Array[Double](n*n)
    i = 0
    while (i < n) {
      var j = 0
      while (j < n) {
        var k = 0
        var numer = 0.0
        var denom = 0.0
        while (k < m) {
          val gt_ki = g(i,k)
          val gt_kj = g(j,k)
          val mu_ki = mu_si(k,i)
          val mu_kj = mu_si(k,j)
          val gD_ki = toDom(gt_ki, mu_ki)
          val gD_kj = toDom(gt_kj, mu_kj)

          if (goodMu(mu_ki) && goodMu(mu_kj) && goodGT(gt_ki) && goodGT(gt_kj)) {
            numer += (gD_ki - mu_ki * (1.0 - mu_ki) * 2.0 * phi(i,i)) *
            (gD_kj - mu_kj * (1.0 - mu_kj) * 2.0 * phi(j,j))

            denom += mu_ki * (1.0 - mu_ki) * mu_kj * (1.0 - mu_kj)
          }

          k += 1
        }

        k2a(j*n + i) = numer / denom

        j += 1
      }
      i += 1
    }
    val k2 = new BDM[Double](n, n, k2a)

    val ibs0a = new Array[Double](n*n)
    i = 0
    while (i < n) {
      var j = 0
      while (j < n) {
        var k = 0
        var count = 0.0
        while (k < m) {
          val g_ki = g(i,k)
          val g_kj = g(j,k)
          val mu_ki = mu_si(k,i)
          val mu_kj = mu_si(k,j)

          if (goodMu(mu_ki) && goodMu(mu_kj) && goodGT(g_ki) && goodGT(g_kj))
            if (math.abs(g_ki - g_kj) == 2.0)
              count += 1.0

          k += 1
        }

        ibs0a(j*n + i) = count

        j += 1
      }
      i += 1
    }
    val ibs0 = new BDM[Double](n,n,ibs0a)

    val k0a = new Array[Double](n*n)
    i = 0
    val k0cutoff = math.pow(2.0, (-5.0/2.0))
    while (i < n) {
      var j = 0
      while (j < n) {
        if (phi(i,j) > k0cutoff) {
          var k = 0
          var denom = 0.0
          while (k < m) {
            val g_ki = g(i,k)
            val g_kj = g(j,k)
            val mu_ki = mu_si(k,i)
            val mu_kj = mu_si(k,j)

            if (goodMu(mu_ki) && goodMu(mu_kj) && goodGT(g_ki) && goodGT(g_kj)) {
              denom += mu_ki*mu_ki*(1.0-mu_kj)*(1.0-mu_kj) + mu_kj*mu_kj*(1.0-mu_ki)*(1.0-mu_ki)
            }

            k += 1
          }

          k0a(j*n + i) = ibs0(i,j) / denom
        } else {
          k0a(j*n + i) = 1.0 - 4.0 * phi(i,j) + k2(i,j)
        }

        j += 1
      }
      i += 1
    }

    val k0 = new BDM[Double](n, n, k0a)

    val k1 = 1.0 - (k0 :+ k2)

    (PCRelate.Result(phi, k0, k1, k2), ibs0, mu_si)
  }
}
