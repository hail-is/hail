package is.hail.methods

import breeze.linalg._
import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.stats._
import is.hail.utils._
import is.hail.variant._

object LinearRegression {
  def apply(vds: VariantDataset, ysExpr: Array[String], xsExpr: Array[String], covsExpr: Array[String], root: String, variantBlockSize: Int): VariantDataset = {
    require(vds.wasSplit)

    if (xsExpr.isEmpty)
      fatal("linreg: must have at least one per-key covariate expression in `xs'.")

    if (ysExpr.isEmpty)
      fatal("linreg: must have at least one response expression in `ys'.")

    val (ys, covs, completeSampleIndex) = RegressionUtils.getPhenosCovCompleteSamples(vds, ysExpr, covsExpr)
    val sampleMask = Array.fill[Boolean](vds.nSamples)(false)
    completeSampleIndex.foreach(sampleMask(_) = true)
    
    val n = ys.rows // nCompleteSamples
    val nys = ys.cols
    val nxs = xsExpr.length
    val ncovs = covs.cols
    val d = n - nxs - ncovs
    
    if (d < 1)
      fatal(s"$n samples with $nxs ${ plural(nxs, "field") } and $ncovs ${ plural(ncovs, "covariate") } implies $d degrees of freedom.")

    info(s"Running linear regression for ${ nys } ${ plural(nys, "phenotype") } on $n samples with $nxs ${ plural(nxs, "field") } and $ncovs ${ plural(ncovs, "covariate") }...")

    val Qt = qr.reduced.justQ(covs).t
    val Qty = Qt * ys
    val yyp = ys.t(*, ::).map(r => r dot r) - Qty.t(*, ::).map(r => r dot r)
    
    val sc = vds.sparkContext
    val sampleMaskBc = sc.broadcast(sampleMask)
    val yBc = sc.broadcast(ys)
    val QtBc = sc.broadcast(Qt)
    val QtyBc = sc.broadcast(Qty)
    val yypBc = sc.broadcast(yyp)

    val vas = vds.vaSignature
    val sas = vds.saSignature
    
    val symTab: SymbolTable = Map(
      "v"      -> (0, TVariant(GenomeReference.GRCh37)),
      "va"     -> (1, vas),
      "s"      -> (2, TString),
      "sa"     -> (3, sas),
      "g"      -> (4, TGenotype),
      "global" -> (5, vds.globalSignature))

    val ec = EvalContext(symTab)
    ec.set(5, vds.globalAnnotation)

    val samplesIds = vds.sampleIds // filter before broadcast?
    val sampleAnnotations = vds.sampleAnnotations

    val sampleIdsBc = sc.broadcast(samplesIds)
    val sampleAnnotationsBc = sc.broadcast(sampleAnnotations)
   
    val pathVA = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)
    val (newVAS, inserter) = vds.insertVA(LinearRegressionModel.schemaNew, pathVA)
    
    val newRDD = 
      if (nxs == 1) {
        val xExpr = xsExpr(0)
        val (xType, xThunk) = Parser.parseExpr(xExpr, ec)
        val toFloat64 = RegressionUtils.toFloat64(xType, xExpr)
        
        vds.rdd.mapPartitions( { it =>            
          val sampleMask = sampleMaskBc.value
          var data: Array[Double] = null
          var means: Array[Double] = null
          val missingRows = new ArrayBuilder[Int]()
          val missingCols = new ArrayBuilder[Int]()
          
          // columns are single fields for block of variants
          var X: DenseMatrix[Double] = null
    
          it.grouped(variantBlockSize)
            .flatMap { git =>
              val block = git.toArray
              val blockSize = block.length
              
              if (X == null || X.cols != blockSize) {
                data = Array.ofDim[Double](n * blockSize)
                means = Array.fill[Double](blockSize)(0.0)
                X = new DenseMatrix[Double](n, blockSize, data)
              }

              missingRows.clear()
              missingCols.clear()
              
              var c = 0
              block.foreach { case (v, (va, gs)) =>
                ec.set(0, v)
                ec.set(1, va)
                
                val gsIter = gs.iterator

                var sum = 0.0
                var nMissing = 0
                var r = 0
                var i = 0
                while (i < sampleMask.length) {
                  val g = gsIter.next()
                  if (sampleMask(i)) {
                    ec.set(2, sampleIdsBc.value(i))
                    ec.set(3, sampleAnnotationsBc.value(i))
                    ec.set(4, g)

                    val x0 = xThunk()
                    if (x0 != null) {
                      val x = toFloat64(x0)
                      sum += x
                      data(c * n + r) = x
                    } else {
                      nMissing += 1
                      missingRows += r
                      missingCols += c
                    }
                    r += 1
                  }
                  i += 1
                }
                means(c) = sum / (n - nMissing)
                c += 1
              }
              
              var j = 0
              while (j < missingRows.length) {
                val c = missingCols(j)
                data(c * n + missingRows(j)) = means(c)
                j += 1
              }
              
              val stats = LinearRegressionModel.fitBlock(X, yBc.value, yypBc.value, QtBc.value, QtyBc.value, d: Int, blockSize: Int)
                                      
              (block, stats).zipped.map { case ((v, (va, gs)), stat) => (v, (inserter(va, stat.toAnnotation), gs)) }
            }
        }, preservesPartitioning = true) 
      } else {
        val (xsType, xsThunk) = Parser.parseExprs(xsExpr.mkString(","), ec)
        val toFloat64Array = (xsType, xsExpr).zipped.map(RegressionUtils.toFloat64)

        vds.rdd.mapPartitions({ it =>
          val sampleMask = sampleMaskBc.value
          val data = Array.ofDim[Double](n * nxs)
          val colIndices = (0 until nxs).toArray
          val colSums = Array.fill[Double](nxs)(0.0)
          val nMissings = Array.fill[Int](nxs)(0)
          val missingRows = new ArrayBuilder[Int]()
          val missingCols = new ArrayBuilder[Int]()

          // columns are all fields of single variant
          val X = new DenseMatrix[Double](n, nxs, data)

          it.map { case (v, (va, gs)) =>
            ec.set(0, v)
            ec.set(1, va)

            var c = 0
            while (c < nxs) {
              colSums(c) = 0
              nMissings(c) = 0
              c += 1
            }
            missingRows.clear()
            missingCols.clear()

            val gsIter = gs.iterator

            var r = 0
            var i = 0
            while (i < sampleMask.length) {
              val g = gsIter.next()
              if (sampleMask(i)) {
                ec.set(2, sampleIdsBc.value(i))
                ec.set(3, sampleAnnotationsBc.value(i))
                ec.set(4, g)

                val xs0 = xsThunk()
                
                var c = 0
                while (c < nxs) {
                  val x0 = xs0(c)
                  if (x0 != null) {
                    val x = toFloat64Array(c)(x0)
                    colSums(c) += x
                    data(c * n + r) = x
                  } else {
                    nMissings(c) += 1
                    missingRows += r
                    missingCols += c
                  }
                  c += 1
                }
                r += 1
              }
              i += 1
            }

            var j = 0
            while (j < missingRows.length) {
              c = missingCols(j)
              data(c * n + missingRows(j)) = colSums(c) / (n - nMissings(c))
              j += 1
            }

            val stat = LinearRegressionModel.fit(X, yBc.value, yypBc.value, QtBc.value, QtyBc.value, d)

            (v, (inserter(va, stat.map(_.toAnnotation).orNull), gs))
          }
        }, preservesPartitioning = true)
      }
    
    vds.copy(
      rdd = newRDD.asOrderedRDD,
      vaSignature = newVAS)
  }
}
