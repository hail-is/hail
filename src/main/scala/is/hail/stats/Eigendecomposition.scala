package is.hail.stats

import breeze.linalg.{DenseMatrix, DenseVector}
import is.hail.annotations.Annotation
import is.hail.expr.Type
import is.hail.utils._

case class Eigendecomposition(rowSignature: Type, rowIds: Array[Annotation], evects: DenseMatrix[Double], evals: DenseVector[Double]) {
  require(evects.rows == rowIds.length)
  require(evects.cols == evals.length)
    
  def filterRows(signature: Type, pred: (Annotation => Boolean)): Eigendecomposition = {
    require(signature == rowSignature)
        
    val (newRowIds, newRows) = rowIds.zipWithIndex.filter{ case (id, row) => pred(id) }.unzip
    val newEvects = evects.filterRows(newRows.toSet).getOrElse(fatal("No rows left")) // FIXME: improve message
    
    Eigendecomposition(rowSignature, newRowIds, newEvects, evals)
  }
  
  def take(k: Int): Eigendecomposition = {
    if (k < 1)
      fatal(s"k must be a positive integer, got $k")
    else if (k >= rowIds.length)
      this
    else
      Eigendecomposition(rowSignature, rowIds, evects(::, 0 until k), evals(0 until k))
  }
}