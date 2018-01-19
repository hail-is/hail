package is.hail.methods

import breeze.linalg._
import breeze.numerics.sqrt
import is.hail.annotations._
import is.hail.expr._
import is.hail.expr.types._
import is.hail.stats._
import is.hail.utils._
import is.hail.variant._
import net.sourceforge.jdistlib.T

object LinearRegression {
  def schema = TStruct(
    ("nCompleteSamples", TInt32()),
    ("AC", TFloat64()),
    ("ytx", TArray(TFloat64())),
    ("beta", TArray(TFloat64())),
    ("se", TArray(TFloat64())),
    ("tstat", TArray(TFloat64())),
    ("pval", TArray(TFloat64())))

  def apply(vsm: MatrixTable,
    ysExpr: Array[String], xExpr: String, covExpr: Array[String], root: String, variantBlockSize: Int
  ): MatrixTable = {
    val ec = vsm.matrixType.genotypeEC
    val xf = RegressionUtils.parseExprAsDouble(xExpr, ec)

    val (y, cov, completeSampleIndex) = RegressionUtils.getPhenosCovCompleteSamples(vsm, ysExpr, covExpr)

    val n = y.rows // nCompleteSamples
    val k = cov.cols // nCovariates
    val d = n - k - 1
    val dRec = 1d / d

    if (d < 1)
      fatal(s"$n samples and $k ${ plural(k, "covariate") } including intercept implies $d degrees of freedom.")

    info(s"Running linear regression for ${ y.cols } ${ plural(y.cols, "phenotype") } on $n samples with $k ${ plural(k, "covariate") } including intercept...")

    val Qt = qr.reduced.justQ(cov).t
    val Qty = Qt * y

    val sc = vsm.sparkContext

    val localGlobalAnnotationBc = sc.broadcast(vsm.globalAnnotation)
    val sampleIdsBc = vsm.sampleIdsBc
    val sampleAnnotationsBc = vsm.sampleAnnotationsBc

    val completeSampleIndexBc = sc.broadcast(completeSampleIndex)
    val yBc = sc.broadcast(y)
    val QtBc = sc.broadcast(Qt)
    val QtyBc = sc.broadcast(Qty)
    val yypBc = sc.broadcast(y.t(*, ::).map(r => r dot r) - Qty.t(*, ::).map(r => r dot r))

    val pathVA = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)
    val (newRDD2Type, inserter) = vsm.rdd2.typ.insert(LinearRegression.schema, "va" :: pathVA)
    val newVAType = newRDD2Type.rowType.fieldTypes(2)

    val localRowType = vsm.rowType
    val newRDD2 = vsm.rdd2.copy(
      typ = newRDD2Type,
      rdd = vsm.rdd2.mapPartitions { it =>
        val region2 = Region()
        val rvb = new RegionValueBuilder(region2)
        val rv2 = RegionValue(region2)

        val missingSamples = new ArrayBuilder[Int]

        // columns are genotype vectors
        var X: DenseMatrix[Double] = new DenseMatrix[Double](n, variantBlockSize)

        val blockWRVs = new Array[WritableRegionValue](variantBlockSize)
        var i = 0
        while (i < variantBlockSize) {
          blockWRVs(i) = WritableRegionValue(localRowType)
          i += 1
        }

        it.trueGroupedIterator(variantBlockSize)
          .flatMap { git =>
            var i = 0
            while (git.hasNext) {
              val rv = git.next()

              val ur = new UnsafeRow(localRowType, rv)
              val v = ur.get(1)
              val va = ur.get(2)
              val gs = ur.getAs[IndexedSeq[Annotation]](3)

              RegressionUtils.inputVector(X(::, i),
                localGlobalAnnotationBc.value, sampleIdsBc.value, sampleAnnotationsBc.value, (v, (va, gs)),
                ec, xf,
                completeSampleIndexBc.value, missingSamples)

              blockWRVs(i).set(rv)
              i += 1
            }
            val blockLength = i

            // val AC: DenseMatrix[Double] = sum(X(::, *))
            val AC: DenseVector[Double] = X.t(*, ::).map(r => sum(r))
            assert(AC.length == variantBlockSize)

            val qtx: DenseMatrix[Double] = QtBc.value * X
            val qty: DenseMatrix[Double] = QtyBc.value
            val xxpRec: DenseVector[Double] = 1.0 / (X.t(*, ::).map(r => r dot r) - qtx.t(*, ::).map(r => r dot r))
            val ytx: DenseMatrix[Double] = yBc.value.t * X
            assert(ytx.rows == yBc.value.cols && ytx.cols == variantBlockSize)

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

            (0 until blockLength).iterator.map { i =>
              val result = Annotation(
                n,
                AC(i),
                ytx(::, i).toArray: IndexedSeq[Double],
                b(::, i).toArray: IndexedSeq[Double],
                se(::, i).toArray: IndexedSeq[Double],
                t(::, i).toArray: IndexedSeq[Double],
                p(::, i).toArray: IndexedSeq[Double])

              val wrv = blockWRVs(i)
              region2.setFrom(wrv.region)
              val offset2 = wrv.offset

              rvb.start(newRDD2Type.rowType)
              inserter(region2, offset2, rvb, () =>
                rvb.addAnnotation(LinearRegression.schema, result))

              rv2.setOffset(rvb.end())
              rv2
            }
          }
      })

    vsm.copy2(
      rdd2 = newRDD2,
      vaSignature = newVAType)
  }
}
