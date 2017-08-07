package is.hail.methods

import breeze.linalg._
import breeze.numerics.sqrt
import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.stats._
import is.hail.utils._
import is.hail.variant._
import net.sourceforge.jdistlib.T

object LinearRegression {
  def schema = TStruct(
    ("nCompleteSamples", TInt32),
    ("AC", TFloat64),
    ("ytx", TArray(TFloat64)),
    ("beta", TArray(TFloat64)),
    ("se", TArray(TFloat64)),
    ("tstat", TArray(TFloat64)),
    ("pval", TArray(TFloat64)))

  def apply(vds: VariantDataset, ysExpr: Array[String], covExpr: Array[String], root: String, useDosages: Boolean, variantBlockSize: Int): VariantDataset = {
    require(vds.wasSplit)

    val (y, cov, completeSamples) = RegressionUtils.getPhenosCovCompleteSamples(vds, ysExpr, covExpr)
    val completeSamplesSet = completeSamples.toSet
    val sampleMask = vds.sampleIds.map(completeSamplesSet).toArray
    val completeSampleIndex = (0 until vds.nSamples)
      .filter(i => completeSamplesSet(vds.sampleIds(i)))
      .toArray
    
    val n = y.rows // nCompleteSamples
    val k = cov.cols // nCovariates
    val d = n - k - 1
    val dRec = 1d / d

    if (d < 1)
      fatal(s"$n samples and $k ${ plural(k, "covariate") } including intercept implies $d degrees of freedom.")

    info(s"Running linear regression for ${ y.cols } ${ plural(y.cols, "phenotype") } on $n samples with $k ${ plural(k, "covariate") } including intercept...")

    val Qt = qr.reduced.justQ(cov).t
    val Qty = Qt * y

    val sc = vds.sparkContext
    val sampleMaskBc = sc.broadcast(sampleMask)
    val completeSampleIndexBc = sc.broadcast(completeSampleIndex)
    val yBc = sc.broadcast(y)
    val QtBc = sc.broadcast(Qt)
    val QtyBc = sc.broadcast(Qty)
    val yypBc = sc.broadcast(y.t(*, ::).map(r => r dot r) - Qty.t(*, ::).map(r => r dot r))

    val pathVA = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)
    val (newVAS, inserter) = vds.insertVA(LinearRegression.schema, pathVA)

    val newRDD = vds.rdd.mapPartitions({ it =>
      val missingSamples = new ArrayBuilder[Int]

      // columns are genotype vectors
      var X: DenseMatrix[Double] = null

      it.grouped(variantBlockSize)
        .flatMap { git =>
          val block = git.toArray
          val blockLength = block.length

          if (X == null || X.cols != blockLength)
            X = new DenseMatrix[Double](n, blockLength)

          var i = 0
          while (i < blockLength) {
            val (_, (_, gs)) = block(i)

            if (useDosages)
              RegressionUtils.dosages(X(::, i), gs, completeSampleIndexBc.value, missingSamples)
            else
              X(::, i) := RegressionUtils.hardCalls(gs, n, sampleMaskBc.value)

            i += 1
          }

          // val AC: DenseMatrix[Double] = sum(X(::, *))
          val AC: DenseVector[Double] = X.t(*, ::).map(r => sum(r))
          assert(AC.length == blockLength)

          val qtx: DenseMatrix[Double] = QtBc.value * X
          val qty: DenseMatrix[Double] = QtyBc.value
          val xxpRec: DenseVector[Double] = 1.0 / (X.t(*, ::).map(r => r dot r) - qtx.t(*, ::).map(r => r dot r))
          val ytx: DenseMatrix[Double] = yBc.value.t * X
          assert(ytx.rows == yBc.value.cols && ytx.cols == blockLength)

          val xyp: DenseMatrix[Double] = ytx - (qty.t * qtx)
          val yyp: DenseVector[Double] = yypBc.value

          // resuse xyp
          val b = xyp
          i = 0
          while (i < blockLength) {
            xyp(::, i) :*= xxpRec(i)
            i += 1
          }

          val se = sqrt(dRec * (yyp * xxpRec.t - (b :* b)))

          val t = b :/ se
          val p = t.map(s => 2 * T.cumulative(-math.abs(s), d, true, false))

          block.zipWithIndex.map { case ((v, (va, gs)), i) =>
            val result = Annotation(
              n,
              AC(i),
              ytx(::, i).toArray: IndexedSeq[Double],
              b(::, i).toArray: IndexedSeq[Double],
              se(::, i).toArray: IndexedSeq[Double],
              t(::, i).toArray: IndexedSeq[Double],
              p(::, i).toArray: IndexedSeq[Double])
            val newVA = inserter(va, result)
            (v, (newVA, gs))
          }
        }
    }, preservesPartitioning = true)

    vds.copy(
      rdd = newRDD.asOrderedRDD,
      vaSignature = newVAS)
  }
}
