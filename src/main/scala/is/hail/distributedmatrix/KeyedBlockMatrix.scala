package is.hail.distributedmatrix

import is.hail.HailContext
import is.hail.annotations.Annotation
import is.hail.utils._
import org.apache.spark.storage.StorageLevel

object KeyedBlockMatrix {
  def from(bm: BlockMatrix): KeyedBlockMatrix =
    new KeyedBlockMatrix(bm, None, None)
  
  def read(hc: HailContext, uri: String): KeyedBlockMatrix = {
    val hadoop = hc.hadoopConf
    
    val rowFile = uri + "/rowkeys"
    val rowKeys =
      if (hadoop.exists(rowFile))
        Some(Keys.read(hc.sc, rowFile))
      else
        None
    
    val colFile = uri + "/colkeys"
    val colKeys =
      if (hadoop.exists(colFile))
        Some(Keys.read(hc.sc, colFile))
      else
        None

    val bm = BlockMatrix.read(hc, uri + "/blockmatrix")

    new KeyedBlockMatrix(bm, rowKeys, colKeys)
  }
}

class KeyedBlockMatrix(val bm: BlockMatrix, val rowKeys: Option[Keys], val colKeys: Option[Keys]) extends Serializable {
  require(rowKeys.forall(_.length == bm.nRows))
  require(colKeys.forall(_.length == bm.nCols))
  
  def copy(bm: BlockMatrix = bm,
    rowKeys: Option[Keys] = rowKeys,
    colKeys: Option[Keys] = colKeys): KeyedBlockMatrix = new KeyedBlockMatrix(bm, rowKeys, colKeys)

  def cache(): KeyedBlockMatrix = copy(bm.cache())
  
  def persist(storageLevel: StorageLevel): KeyedBlockMatrix = copy(bm.persist(storageLevel))
  
  def setRowKeys(keys: Keys): KeyedBlockMatrix = {
    if (keys.length != bm.nRows)
      fatal(s"Differing number of keys and rows: $keys.length, ${bm.nRows}")    

    copy(rowKeys = Some(keys))
  }
  
  def setColKeys(keys: Keys): KeyedBlockMatrix = {
    if (keys.length != bm.nCols)
      fatal(s"Differing number of keys and cols: ${keys.length}, ${bm.nCols}")    

    copy(colKeys = Some(keys))
  }
  
  def dropRowKeys(): KeyedBlockMatrix = copy(rowKeys = None)

  def dropColKeys(): KeyedBlockMatrix = copy(colKeys = None)

  def dropKeys(): KeyedBlockMatrix = copy(rowKeys = None, colKeys = None)
  
  def unifyRowKeys(that: KeyedBlockMatrix): Option[Keys] = Keys.unify(rowKeys, that.rowKeys, "Inconsistent row keys. ")
  
  def unifyColKeys(that: KeyedBlockMatrix): Option[Keys] = Keys.unify(colKeys, that.colKeys, "Inconsistent col keys. ")
  
  def add(that: KeyedBlockMatrix): KeyedBlockMatrix =
    new KeyedBlockMatrix(bm.add(that.bm), unifyRowKeys(that), unifyColKeys(that))
  
  def subtract(that: KeyedBlockMatrix): KeyedBlockMatrix =
      new KeyedBlockMatrix(bm.subtract(that.bm), unifyRowKeys(that), unifyColKeys(that))


  def multiply(that: KeyedBlockMatrix): KeyedBlockMatrix = {
    Keys.unify(colKeys, that.rowKeys, "Column keys on left do not match row keys on right. ")
    new KeyedBlockMatrix(bm.multiply(that.bm), rowKeys, that.colKeys)
  }

  def multiply(bm: BlockMatrix): KeyedBlockMatrix = multiply(KeyedBlockMatrix.from(bm))

  def scalarAdd(i: Double): KeyedBlockMatrix = copy(bm.scalarAdd(i))
  
  def scalarSubtract(i: Double): KeyedBlockMatrix = copy(bm.scalarSubtract(i))
  
  def scalarMultiply(i: Double): KeyedBlockMatrix = copy(bm.scalarMultiply(i))

  def scalarDivide(i: Double): KeyedBlockMatrix = copy(bm.scalarDivide(i))

  def pointwiseMultiply(that: KeyedBlockMatrix): KeyedBlockMatrix =
    new KeyedBlockMatrix(bm.pointwiseMultiply(that.bm), unifyRowKeys(that), unifyColKeys(that))

  def pointwiseDivide(that: KeyedBlockMatrix): KeyedBlockMatrix =
    new KeyedBlockMatrix(bm.pointwiseDivide(that.bm), unifyRowKeys(that), unifyColKeys(that))
    
  def vectorAddToEveryRow(v: Array[Double]): KeyedBlockMatrix = copy(bm.vectorAddToEveryRow(v))

  def vectorAddToEveryColumn(v: Array[Double]): KeyedBlockMatrix = copy(bm.vectorAddToEveryColumn(v))

  def vectorPointwiseMultiplyEveryRow(v: Array[Double]): KeyedBlockMatrix = copy(bm.vectorPointwiseMultiplyEveryRow(v))

  def vectorPointwiseMultiplyEveryColumn(v: Array[Double]): KeyedBlockMatrix = copy(bm.vectorPointwiseMultiplyEveryColumn(v))
  
  def transpose(): KeyedBlockMatrix = new KeyedBlockMatrix(bm.transpose(), colKeys, rowKeys)

  def filterRows(pred: Annotation => Boolean): KeyedBlockMatrix =
    rowKeys match {
      case Some(rk) =>
        val (newRK, keep) = rk.filterAndIndex(pred)
        new KeyedBlockMatrix(bm.filterRows(keep.map(_.toLong)), Some(newRK), colKeys)
      case None =>
        fatal("Cannot filter rows using predicate: no row keys")
    }
  
  def filterRows(keep: Array[Int]): KeyedBlockMatrix =
    new KeyedBlockMatrix(
      bm.filterRows(keep.map(_.toLong)),
      rowKeys.map(_.filter(keep, check = false)),
      colKeys)

  def filterCols(pred: Annotation => Boolean): KeyedBlockMatrix =
    colKeys match {
      case Some(ck) =>
        val (newCK, keep) = ck.filterAndIndex(pred)
        new KeyedBlockMatrix(bm.filterCols(keep.map(_.toLong)), rowKeys, Some(newCK))
      case None =>
        fatal("Cannot filter cols using predicate: no col keys")
    }
  
  def filterCols(keep: Array[Int]): KeyedBlockMatrix =
      new KeyedBlockMatrix(
        bm.filterCols(keep.map(_.toLong)),
        rowKeys,
        colKeys.map(_.filter(keep, check = false)))

  def filter(rowPred: Annotation => Boolean, colPred: Annotation => Boolean): KeyedBlockMatrix =
    (rowKeys, colKeys) match {
      case (Some(rk), Some(ck)) =>
        val (newRK, keepRows) = rk.filterAndIndex(rowPred)
        val (newCK, keepCols) = ck.filterAndIndex(colPred)
        new KeyedBlockMatrix(
          bm.filter(keepRows.map(_.toLong), keepCols.map(_.toLong)),
          Some(newRK),
          Some(newCK))
      case (None, Some(_)) => fatal("Cannot filter using predicate: no row keys")
      case (Some(_), None) => fatal("Cannot filter using predicate: no col keys")
      case _ => fatal("Cannot filter using predicates: no row keys, no col keys")
    }
  
  def filter(keepRows: Array[Int], keepCols: Array[Int]): KeyedBlockMatrix =
        new KeyedBlockMatrix(
          bm.filter(keepRows.map(_.toLong), keepCols.map(_.toLong)),
          rowKeys.map(_.filter(keepRows, check = false)),
          colKeys.map(_.filter(keepCols, check = false)))
  
  def write(uri: String, forceRowMajor: Boolean = false) {
    val sc = bm.blocks.sparkContext
    sc.hadoopConfiguration.mkDir(uri)

    rowKeys.foreach(_.write(sc, uri + "/rowkeys"))
    colKeys.foreach(_.write(sc, uri + "/colkeys"))
    bm.write(uri + "/blockmatrix", forceRowMajor)
  }
}
