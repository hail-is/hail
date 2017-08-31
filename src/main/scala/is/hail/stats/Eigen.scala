package is.hail.stats

import breeze.linalg.{*, DenseMatrix, DenseVector, sum}
import is.hail.HailContext
import is.hail.annotations.Annotation
import is.hail.distributedmatrix.{BlockMatrixIsDistributedMatrix, DistributedMatrix}
import is.hail.distributedmatrix.DistributedMatrix.implicits._
import is.hail.expr.{JSONAnnotationImpex, Parser, TString, TVariant, Type}
import is.hail.utils._
import is.hail.variant.{Variant, VariantDataset}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{DenseMatrix => SparkDenseMatrix}
import org.apache.spark.mllib.linalg.distributed.BlockMatrix
import org.json4s.jackson.{JsonMethods, Serialization}
import org.json4s.{JArray, JInt, JObject, JString, JValue}

import scala.reflect.ClassTag

object Eigen {
  def apply(rowSignature: Type, rowIds: Array[Annotation], X: DenseMatrix[Double], optRankBound: Option[Int]): Eigen = {
    val n = rowIds.length
    require(n == X.rows && n == X.cols)
    
    info(s"Computing eigenvectors...")
    val eig = printTime(eigSymD(X))
    
    val nEigs = optRankBound.map(_ min n).getOrElse(n)
    
    info(s"Eigendecomposition complete, returning $nEigs eigenvectors.")
        
    val (evects, evals) =
      if (nEigs == n)
        (eig.eigenvectors, eig.eigenvalues)
      else
        (eig.eigenvectors(::, (n - nEigs) until n).copy, eig.eigenvalues((n - nEigs) until n).copy)
    
    Eigen(rowSignature, rowIds, evects, evals)
  }
  
  private val metadataRelativePath = "/metadata.json.gz"
  private val evectsRelativePath = "/evects"
  private val evalsRelativePath = "/evals"
  
  def read(hc: HailContext, uri: String): Eigen = {
    if (!uri.endsWith(".eig") && !uri.endsWith(".eig/"))
      fatal(s"input path ending in `.eig' required, found `$uri'")

    val hadoop = hc.hadoopConf

    val json = try {
      hadoop.readFile(uri + metadataRelativePath)(
        in => JsonMethods.parse(in))
    } catch {
      case e: Throwable => fatal(
        s"""
           |invalid eig metadata file.
           |  Recreate with current version of Hail.
           |  caught exception: ${ expandException(e, logMessage = true) }
         """.stripMargin)
    }

    val fields = json match {
      case jo: JObject => jo.obj.toMap
      case _ =>
        fatal(
          s"""eig: invalid metadata value
             |  Recreate with current version of Hail.""".stripMargin)
    }

    def getAndCastJSON[T <: JValue](fname: String)(implicit tct: ClassTag[T]): T =
      fields.get(fname) match {
        case Some(t: T) => t
        case Some(other) =>
          fatal(
            s"""corrupt eig: invalid metadata
               |  Expected `${ tct.runtimeClass.getName }' in field `$fname', but got `${ other.getClass.getName }'
               |  Recreate with current version of Hail.""".stripMargin)
        case None =>
          fatal(
            s"""corrupt eig: invalid metadata
               |  Missing field `$fname'
               |  Recreate with current version of Hail.""".stripMargin)
      }
   
    val rowSignature = fields.get("row_schema") match {
      case Some(t: JString) => Parser.parseType(t.s)
      case Some(other) => fatal(
        s"""corrupt eig: invalid metadata
           |  Expected `JString' in field `row_schema', but got `${ other.getClass.getName }'
           |  Recreate with current version of Hail.""".stripMargin)
      case _ => TString
    }
    
    val rowIds = getAndCastJSON[JArray]("row_ids")
      .arr
      .map {
        case JObject(List(("id", jRowId))) =>
          JSONAnnotationImpex.importAnnotation(jRowId, rowSignature, "row_ids")
        case _ => fatal(
          s"""corrupt eig: invalid metadata
             |  Invalid sample annotation metadata
             |  Recreate with current version of Hail.""".stripMargin)
      }
      .toArray
    
    val nRows = getAndCastJSON[JInt]("num_rows").num.toInt
    
    val nEigs = getAndCastJSON[JInt]("num_eigs").num.toInt
    
    assert(nRows == rowIds.length)
    
    val nEntries = nRows * nEigs
    val evectsData = Array.ofDim[Double](nEntries)
    val evalsData = Array.ofDim[Double](nEigs)
    
    hadoop.readDataFile(uri + evectsRelativePath) { is =>
      var i = 0
      while (i < nEntries) {
        evectsData(i) = is.readDouble()
        i += 1
      }
    }

    hadoop.readDataFile(uri + evalsRelativePath) { is =>
      var i = 0
      while (i < nEigs) {
        evalsData(i) = is.readDouble()
        i += 1
      }
    }
    
    new Eigen(rowSignature, rowIds, new DenseMatrix[Double](nRows, nEigs, evectsData), DenseVector(evalsData))
  }
}

case class Eigen(rowSignature: Type, rowIds: Array[Annotation], evects: DenseMatrix[Double], evals: DenseVector[Double]) {
  require(evects.rows == rowIds.length)
  require(evects.cols == evals.length)
    
  def nEvects: Int = evals.length
  
  def filterRows(signature: Type, pred: (Annotation => Boolean)): Eigen = {
    require(signature == rowSignature)
        
    val (newRowIds, newRows) = rowIds.zipWithIndex.filter{ case (id, _) => pred(id) }.unzip
    val newEvects = evects.filterRows(newRows.toSet).getOrElse(fatal("Filtering would remove all rows from eigenvectors"))
    
    Eigen(rowSignature, newRowIds, newEvects, evals)
  }
  
  def takeTop(k: Int): Eigen = {
    if (k < 1)
      fatal(s"k must be a positive integer, got $k")
    
    if (k >= nEvects)
      this
    else
      Eigen(rowSignature, rowIds,
        evects(::, (nEvects - k) until nEvects).copy, evals((nEvects - k) until nEvects).copy)
  }

  def dropProportion(proportion: Double = 1e-6): Eigen = {
    if (proportion < 0 || proportion >= 1)
      fatal(s"Relative threshold must be in range [0,1), got $proportion")

    val threshold = proportion * sum(evals)
    var acc = 0.0
    var i = -1
    while (acc <= threshold) {
      i += 1
      acc += evals(i)
    }
    
    info(s"Dropping $i eigenvectors, leaving ${nEvects - i} of the original $nEvects")
    
    Eigen(rowSignature, rowIds, evects(::, i until nEvects).copy, evals(i until nEvects).copy)
  }
  
  def dropThreshold(threshold: Double = 1e-6): Eigen = {
    if (threshold < 0)
      fatal(s"Threshold must be non-negative, got $threshold")

    val newEvals = DenseVector(evals.toArray.dropWhile(_ <= threshold))
    val k = newEvals.length

    info(s"Dropping ${nEvects - k} eigenvectors with eigenvalues below $threshold, leaving $k of the original $nEvects")
    
    Eigen(rowSignature, rowIds, evects(::, (nEvects - k) until nEvects).copy, newEvals)
  }
  
  def evectsSpark(): SparkDenseMatrix = evects.asSpark()
  
  def evalsArray(): Array[Double] = evals.toArray
  
  def distribute(sc: SparkContext): EigenDistributed = {
    val U = BlockMatrixIsDistributedMatrix.from(sc, evects.asSpark(), 1024, 1024)
    EigenDistributed(rowSignature, rowIds, U, evals)
  }
  
  def toEigenDistributedRRM(vds: VariantDataset, nSamplesInLDMatrix: Option[Int]): EigenDistributed = {
    val dm = DistributedMatrix[BlockMatrix]
    import dm.ops._

    // G = (n x m) matrix, consisting of rows of n=nUsedToNormalize x m matrix with columns of mean 0 and variance 1
    // G = U * sqrt(S) * V.t is SVD
    // L = 1 / n0 * G.t * G = V * S_L * V.t, S_L = S / n
    // K = 1 / m * G * G.t = U * S_K * U.t, S_K = S / m
    // U = G * V * inv(sqrt(S_L * n))
    // S_K = S_L * n / m
    // Here G0 is an (n0 x m) matrix consisting of a subset of rows of G and U0 consists of the corresponding subset of rows of U.
    // If G0 has fewer rows than G, then set nUsedToNormalize to Some(n).
    def normalizedRightEigenToLeftEigen(G0: BlockMatrix, V: DenseMatrix[Double], S_L: DenseVector[Double], nUsedToNormalize: Option[Int] = None): (BlockMatrix, DenseVector[Double]) = { 
      val n0 = nUsedToNormalize.getOrElse(G0.numRows().toInt)
      val m = V.rows
      val sqrtSInv = S_L.map(e => 1 / math.sqrt(e * n0))
      
      val U0 = G0 * (V(* , ::) :* sqrtSInv).asSpark()
      val S_K = S_L :* n0.toDouble / m

      (U0, S_K)
    }

    if (rowSignature != TVariant)
      fatal(s"Rows must have type TVariant, got $rowSignature")

   // if (evals(0) < 1e-6)
   //   fatal(s"Use drop_threshold() to drop eigenvectors with eigenvalues below 1e-6, found eigenvalue ${evals(0)}")

    val variants = rowIds.map(_.asInstanceOf[Variant])    
    val variantSet = variants.toSet
    val nEigs = evals.length
    
    info(s"Transforming $nEigs variant eigenvectors to sample eigenvectors...")
        
    var filteredVDS = vds.filterVariants((v, _, _) => variantSet(v))
    filteredVDS = filteredVDS.persist()
    require(filteredVDS.variants.count() == variantSet.size, "Some variants in LD matrix eigendecomposition are missing from VDS")
    
    val G = ToNormalizedIndexedRowMatrix(filteredVDS).toBlockMatrixDense().t
    
    val (evects_K, evals_K) = normalizedRightEigenToLeftEigen(G, evects, evals, nSamplesInLDMatrix)

    filteredVDS.unpersist()

    EigenDistributed(vds.sSignature, vds.sampleIds.toArray, evects_K, evals_K)
  }
  
  import Eigen._
  
  def write(hc: HailContext, uri: String) {   
    if (!uri.endsWith(".eig") && !uri.endsWith(".eig/"))
      fatal(s"output path ending in `.eig' required, found `$uri'")
    
    val hadoop = hc.sc.hadoopConfiguration
    hadoop.mkDir(uri)

    hadoop.writeDataFile(uri + evectsRelativePath) { os =>
      val evectsData = evects.toArrayShallow
      var i = 0
      while (i < evectsData.length) {
        os.writeDouble(evectsData(i))
        i += 1
      }
    }

    hadoop.writeDataFile(uri + evalsRelativePath) { os =>
      val evalsData = evals.toArray
      var i = 0
      while (i < evalsData.length) {
        os.writeDouble(evalsData(i))
        i += 1
      }
    }
    
    val sb = new StringBuilder
    rowSignature.pretty(sb, printAttrs = true, compact = true)
    val rowSignatureString = sb.result()
    
    val rowIdsJson = JArray(
      rowIds.map(rowId => JObject(("id", JSONAnnotationImpex.exportAnnotation(rowId, rowSignature)))).toList)

    val json = JObject(
      ("row_schema", JString(rowSignatureString)),
      ("row_ids", rowIdsJson),
      ("num_rows", JInt(rowIds.length)),
      ("num_eigs", JInt(nEvects)))
        
    hadoop.writeTextFile(uri + metadataRelativePath)(Serialization.writePretty(json, _))
  }
}

case class EigenFileMetadata(rowSignature: Type, rowIds: Array[String], nRows: Int, nEigs: Int)

object EigenDistributed {
  private val metadataRelativePath = "/metadata.json.gz"
  private val evectsRelativePath = "/evects"
  private val evalsRelativePath = "/evals"
  
  def read(hc: HailContext, uri: String): EigenDistributed = {
    if (!uri.endsWith(".eigd") && !uri.endsWith(".eigd/"))
      fatal(s"input path ending in `.eigd' required, found `$uri'")

    val hadoop = hc.hadoopConf

    val json = try {
      hadoop.readFile(uri + metadataRelativePath)(
        in => JsonMethods.parse(in))
    } catch {
      case e: Throwable => fatal(
        s"""
           |invalid eigd metadata file.
           |  Recreate with current version of Hail.
           |  caught exception: ${ expandException(e, logMessage = true) }
         """.stripMargin)
    }

    val fields = json match {
      case jo: JObject => jo.obj.toMap
      case _ =>
        fatal(
          s"""eigd: invalid metadata value
             |  Recreate with current version of Hail.""".stripMargin)
    }

    def getAndCastJSON[T <: JValue](fname: String)(implicit tct: ClassTag[T]): T =
      fields.get(fname) match {
        case Some(t: T) => t
        case Some(other) =>
          fatal(
            s"""corrupt eigd: invalid metadata
               |  Expected `${ tct.runtimeClass.getName }' in field `$fname', but got `${ other.getClass.getName }'
               |  Recreate with current version of Hail.""".stripMargin)
        case None =>
          fatal(
            s"""corrupt eigd: invalid metadata
               |  Missing field `$fname'
               |  Recreate with current version of Hail.""".stripMargin)
      }
   
    val rowSignature = fields.get("row_schema") match {
      case Some(t: JString) => Parser.parseType(t.s)
      case Some(other) => fatal(
        s"""corrupt eigd: invalid metadata
           |  Expected `JString' in field `row_schema', but got `${ other.getClass.getName }'
           |  Recreate with current version of Hail.""".stripMargin)
      case _ => TString
    }
    
    val rowIds = getAndCastJSON[JArray]("row_ids")
      .arr
      .map {
        case JObject(List(("id", jRowId))) =>
          JSONAnnotationImpex.importAnnotation(jRowId, rowSignature, "row_ids")
        case _ => fatal(
          s"""corrupt eig: invalid metadata
             |  Invalid sample annotation metadata
             |  Recreate with current version of Hail.""".stripMargin)
      }
      .toArray
    
    val nRows = getAndCastJSON[JInt]("num_rows").num.toInt
    
    val nEigs = getAndCastJSON[JInt]("num_eigs").num.toInt
    
    assert(nRows == rowIds.length)
    
    val evalsData = Array.ofDim[Double](nEigs)
        
    val dm = DistributedMatrix[BlockMatrix]
    val evects = dm.read(hc, uri + evectsRelativePath)

    hadoop.readDataFile(uri + evalsRelativePath) { is =>
      var i = 0
      while (i < nEigs) {
        evalsData(i) = is.readDouble()
        i += 1
      }
    }
    
    new EigenDistributed(rowSignature, rowIds, evects, DenseVector(evalsData))
  }
}

case class EigenDistributed(rowSignature: Type, rowIds: Array[Annotation], evects: BlockMatrix, evals: DenseVector[Double]) {
  require(evects.numRows() == rowIds.length)
  require(evects.numCols() == evals.length)
  
  def nEvects: Int = evals.length
  
  def evalsArray(): Array[Double] = evals.toArray
    
  def localize(): Eigen = {
    val U = evects.toLocalMatrix().asBreeze().asInstanceOf[DenseMatrix[Double]]
    Eigen(rowSignature, rowIds, U, evals)
  }
  
  import EigenDistributed._
  
  def write(uri: String) {   
    if (!uri.endsWith(".eigd") && !uri.endsWith(".eigd/"))
      fatal(s"output path ending in `.eigd' required, found `$uri'")
    
    val hadoop = evects.blocks.sparkContext.hadoopConfiguration
    hadoop.mkDir(uri)
    
    val dm = DistributedMatrix[BlockMatrix]
    dm.write(evects, uri + evectsRelativePath)
    
    hadoop.writeDataFile(uri + evalsRelativePath) { os =>
      val evalsData = evals.toArray
      var i = 0
      while (i < evalsData.length) {
        os.writeDouble(evalsData(i))
        i += 1
      }
    }
    
    val sb = new StringBuilder
    rowSignature.pretty(sb, printAttrs = true, compact = true)
    val rowSignatureString = sb.result()
    
    val rowIdsJson = JArray(
      rowIds.map(rowId => JObject(("id", JSONAnnotationImpex.exportAnnotation(rowId, rowSignature)))).toList)

    val json = JObject(
      ("row_schema", JString(rowSignatureString)),
      ("row_ids", rowIdsJson),
      ("num_rows", JInt(rowIds.length)),
      ("num_eigs", JInt(nEvects)))
        
    hadoop.writeTextFile(uri + metadataRelativePath)(Serialization.writePretty(json, _))
  }
}