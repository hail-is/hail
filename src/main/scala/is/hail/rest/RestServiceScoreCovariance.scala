package is.hail.rest

import breeze.linalg._
import is.hail.distributedmatrix.DistributedMatrix
import is.hail.stats.RegressionUtils
import is.hail.variant._
import is.hail.utils._
import is.hail.variant.VariantDataset
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow, IndexedRowMatrix}
import org.json4s.jackson.Serialization.read

import scala.collection.mutable

case class GetRequestScoreCovariance(passback: Option[String],
  md_version: Option[String],
  api_version: Int,
  phenotype: Option[String],
  covariates: Option[Array[Covariate]],
  variant_filters: Option[Array[VariantFilter]],
  variant_list: Option[Array[SingleVariant]],
  compute_cov: Option[Boolean],
  limit: Option[Int],
  count: Option[Boolean]) extends GetRequest

case class GetResultScoreCovariance(is_error: Boolean,
  error_message: Option[String],
  passback: Option[String],
  active_variants: Option[Array[SingleVariant]],
  scores: Option[Array[Double]],
  covariance: Option[Array[Double]],
  sigma_sq: Option[Double],
  nsamples: Option[Int],
  count: Option[Int]) extends GetResult

class RestServiceScoreCovariance(vds: VariantDataset, covariates: Array[String], useDosages: Boolean, maxWidth: Int, hardLimit: Int) extends RestService { 
  private val nSamples: Int = vds.nSamples
  private val sampleMask: Array[Boolean] = Array.ofDim[Boolean](nSamples)
  private val availableCovariates: Set[String] = covariates.toSet
  private val availableCovariateToIndex: Map[String, Int] = covariates.zipWithIndex.toMap
  private val (sampleIndexToPresentness: Array[Array[Boolean]], 
                 covariateIndexToValues: Array[Array[Double]]) = RegressionUtils.getSampleAndCovMaps(vds, covariates)
  
  def readText(text: String): GetRequestScoreCovariance = read[GetRequestScoreCovariance](text)
  
  def getError(message: String, passback: Option[String]) = GetResultScoreCovariance(is_error = true, Some(message), passback, None, None, None, None, None, None)
  
  def getStats(req0: GetRequest): GetResultScoreCovariance = {
    val req = req0.asInstanceOf[GetRequestScoreCovariance]
    req.md_version.foreach { md_version =>
      if (md_version != "mdv1")
        throw new RestFailure(s"Unknown md_version `$md_version'. Available md_versions: mdv1")
    }

    if (req.api_version != 1)
      throw new RestFailure(s"Unsupported API version `${ req.api_version }'. Supported API versions: 1")

    // construct yName, covNames, and covVariants
    val covNamesSet = mutable.Set[String]()
    val covVariantsSet = mutable.Set[Variant]()

    req.covariates.foreach { covariates =>
      for (c <- covariates)
        c.`type` match {
          case "phenotype" =>
            c.name match {
              case Some(name) =>
                if (availableCovariates(name))
                  if (covNamesSet(name))
                    throw new RestFailure(s"Covariate $name is included as a covariate more than once")
                  else
                    covNamesSet += name
                else
                  throw new RestFailure(s"$name is not a valid covariate name")
              case None =>
                throw new RestFailure("Covariate of type 'phenotype' must include 'name' field in request")
            }
          case "variant" =>
            (c.chrom, c.pos, c.ref, c.alt) match {
              case (Some(chr), Some(pos), Some(ref), Some(alt)) =>
                val v = Variant(chr, pos, ref, alt)
                if (covVariantsSet(v))
                  throw new RestFailure(s"$v is included as a covariate more than once")
                else
                  covVariantsSet += v
              case missingFields =>
                throw new RestFailure("Covariate of type 'variant' must include 'chrom', 'pos', 'ref', and 'alt' fields in request")
            }
          case other =>
            throw new RestFailure(s"Covariate type must be 'phenotype' or 'variant': got $other")
        }
    }

    val covNames = covNamesSet.toArray
    val covVariants = covVariantsSet.toArray

    val yNameOpt = req.phenotype

    yNameOpt.foreach { yName =>
      if (!availableCovariates(yName))
        throw new RestFailure(s"$yName is not a valid phenotype name")
      if (covNamesSet(yName))
        throw new RestFailure(s"$yName appears as both response phenotype and covariate")
    }

    // construct variant filters
    var chrom = ""

    var minPos = 1
    var maxPos = Int.MaxValue // 2,147,483,647 is greater than length of longest chromosome

    var minMAC = 1
    var maxMAC = Int.MaxValue

    val nonNegIntRegEx = """\d+""".r

    req.variant_filters.foreach(_.foreach { f =>
      f.operand match {
        case "chrom" =>
          if (!(f.operator == "eq" && f.operand_type == "string"))
            throw new RestFailure(s"chrom filter operator must be 'eq' and operand_type must be 'string': got '${ f.operator }' and '${ f.operand_type }'")
          else if (f.value.isEmpty)
            throw new RestFailure("chrom filter value cannot be the empty string")
          else if (chrom.isEmpty)
            chrom = f.value
          else if (chrom != f.value)
            throw new RestFailure(s"Got incompatible chrom filters: '$chrom' and '${ f.value }'")
        case "pos" =>
          if (f.operand_type != "integer")
            throw new RestFailure(s"pos filter operand_type must be 'integer': got '${ f.operand_type }'")
          else if (!nonNegIntRegEx.matches(f.value))
            throw new RestFailure(s"Value of position in variant_filter must be a valid non-negative integer: got '${ f.value }'")
          else {
            val pos = f.value.toInt
            f.operator match {
              case "gte" => minPos = minPos max pos
              case "gt" => minPos = minPos max (pos + 1)
              case "lte" => maxPos = maxPos min pos
              case "lt" => maxPos = maxPos min (pos - 1)
              case "eq" => minPos = minPos max pos; maxPos = maxPos min pos
              case other =>
                throw new RestFailure(s"pos filter operator must be 'gte', 'gt', 'lte', 'lt', or 'eq': got '$other'")
            }
          }
        case "mac" =>
          if (f.operand_type != "integer")
            throw new RestFailure(s"mac filter operand_type must be 'integer': got '${ f.operand_type }'")
          else if (!nonNegIntRegEx.matches(f.value))
            throw new RestFailure(s"mac filter value must be a valid non-negative integer: got '${ f.value }'")
          val mac = f.value.toInt
          f.operator match {
            case "gte" => minMAC = minMAC max mac
            case "gt" => minMAC = minMAC max (mac + 1)
            case "lte" => maxMAC = maxMAC min mac
            case "lt" => maxMAC = maxMAC min (mac - 1)
            case other =>
              throw new RestFailure(s"mac filter operator must be 'gte', 'gt', 'lte', 'lt': got '$other'")
          }
        case other => throw new RestFailure(s"Filter operand must be 'chrom' or 'pos': got '$other'")
      }
    })

    // construct window
    if (chrom.isEmpty)
      throw new RestFailure("No chromosome specified in variant_filter")

    val width = maxPos - minPos
    if (width > maxWidth)
      throw new RestFailure(s"Interval length cannot exceed $maxWidth: got $width")

    if (width < 0)
      throw new RestFailure(s"Window is empty: got start $minPos and end $maxPos")

    val window = Interval(Locus(chrom, minPos), Locus(chrom, maxPos + 1))

    info(s"Using window ${ RestService.windowToString(window) } of size ${ width + 1 }")

    val variantsSet = mutable.Set[Variant]()

    req.variant_list.foreach {
      _.foreach { sv =>
        val v =
          if (sv.chrom.isDefined && sv.pos.isDefined && sv.ref.isDefined && sv.alt.isDefined) {
            val v = Variant(sv.chrom.get, sv.pos.get, sv.ref.get, sv.alt.get)
            if (v.contig != chrom || v.start < minPos || v.start > maxPos)
              throw new RestFailure(s"Variant ${ v.toString } from 'variant_list' is not in the window ${ RestService.windowToString(window) }")
            variantsSet += v
          }
          else
            throw new RestFailure(s"All variants in 'variant_list' must include 'chrom', 'pos', 'ref', and 'alt' fields: got ${ (sv.chrom.getOrElse("NA"), sv.pos.getOrElse("NA"), sv.ref.getOrElse("NA"), sv.alt.getOrElse("NA")) }")
      }
    }

    // pull up to server?
    import is.hail.distributedmatrix.DistributedMatrix.implicits._
    val dm = DistributedMatrix[BlockMatrix]
    import dm.ops._
    
    // filter and compute
    val filteredVds =
      if (variantsSet.isEmpty)
        vds.filterIntervals(IntervalTree(Array(window)), keep = true)
      else
        vds.filterVariantsList(variantsSet.toSet, keep = true)

    if (req.count.getOrElse(false)) {
      val count = filteredVds.countVariants().toInt // upper bound as does not account for MAC filtering
      GetResultScoreCovariance(is_error = false, None, req.passback, None, None, None, None, None, Some(count))
    }
    else {
      val (yOpt, cov) = RestService.getYCovAndSetMask(sampleMask, vds, window, yNameOpt, covNames, covVariants,
        useDosages, availableCovariates, availableCovariateToIndex, sampleIndexToPresentness, covariateIndexToValues)

      val (yResOpt, sigmaSqOpt) =
        if (yOpt.isDefined) {
          val y = yOpt.get
          val d = cov.rows - cov.cols
          if (d < 1)
            fatal(s"${cov.rows} samples and ${cov.cols} ${ plural(cov.cols, "covariate") } including intercept implies $d degrees of freedom.")
          
          val qrDecomp = qr.reduced(cov)
          val beta = qrDecomp.r \ (qrDecomp.q.t * y)
          val yRes = y - cov * beta
          val sigmaSq = (yRes dot yRes) / d
          
          (Some(yRes), Some(sigmaSq))
        } else
          (None, None)

      val (xOpt, activeVariants) = ToFilteredCenteredIndexedRowMatrix(filteredVds, cov.rows, sampleMask, minMAC, maxMAC)

      val noActiveVariants = xOpt.isEmpty

      val nCompleteSamples = cov.rows
      
      if (noActiveVariants) {
        GetResultScoreCovariance(is_error = false, None, req.passback,
          Some(Array.empty[SingleVariant]),
          Some(Array.empty[Double]),
          Some(Array.empty[Double]),
          sigmaSqOpt,
          Some(nCompleteSamples),
          Some(0))
      } else {
        val limit = req.limit.getOrElse(hardLimit)
        if (activeVariants.length > limit)
          throw new RestFailure(s"Number of active variants $activeVariants exceeds limit $limit")

        val X = xOpt.get        
        X.rows.persist()
        
        // consider block matrix route
        val scoresOpt = yResOpt.map { yRes =>
          val yResMat = new org.apache.spark.mllib.linalg.DenseMatrix(yRes.length, 1, yRes.toArray, true)
          X.multiply(yResMat).toBlockMatrixDense().toLocalMatrix().toArray
        }
        
        val covarianceOpt =
          if (req.compute_cov.isEmpty || (req.compute_cov.isDefined && req.compute_cov.get)) {
            val Xb = X.toBlockMatrixDense()
            val covarianceSquare = (Xb * Xb.transpose).toLocalMatrix().toArray
            val covarianceTri = RestService.lowerTriangle(covarianceSquare, activeVariants.length)
            Some(covarianceTri)
          } else
            None
        
        X.rows.unpersist()

        val activeSingleVariants = activeVariants.map(v =>
          SingleVariant(Some(v.contig), Some(v.start), Some(v.ref), Some(v.alt)))
        
        GetResultScoreCovariance(is_error = false, None, req.passback,
          Some(activeSingleVariants),
          scoresOpt,
          covarianceOpt,
          sigmaSqOpt,
          Some(nCompleteSamples),
          Some(activeVariants.length))
      }
    }
  }
}

// n is nCompleteSamples
object ToFilteredCenteredIndexedRowMatrix {  
  def apply(vds: VariantDataset, n: Int, mask: Array[Boolean], minMAC: Int, maxMAC: Int): (Option[IndexedRowMatrix], Array[Variant]) = {
    require(vds.wasSplit)
    
    val inRange = RestService.inRangeFunc(n, minMAC, maxMAC)

    val variantVectors = vds.rdd.flatMap { case (v, (_, gs)) =>  // when all missing, gets filtered by ac
      val (x, ac) = RegressionUtils.hardCallsWithAC(gs, n, mask)
      if (inRange(ac)) {
        val mu = sum(x) / n // inefficient
        Some(v, Vectors.dense((x - mu).toArray))
      } else
        None
    }

    // order is preserved from original OrderedRDD
    val activeVariants = variantVectors.map(_._1).collect()
    
    val irmOpt = 
      if (!activeVariants.isEmpty) {
        val reducedIndicesBc = vds.sparkContext.broadcast(activeVariants.index)    
        val indexedRows = variantVectors.map { case (v, x) => IndexedRow(reducedIndicesBc.value(v), x) }
        Some(new IndexedRowMatrix(indexedRows, nRows = activeVariants.length, nCols = n))
      } else
        None

    (irmOpt, activeVariants)
  }
}