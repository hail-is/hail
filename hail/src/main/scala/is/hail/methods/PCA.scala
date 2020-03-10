package is.hail.methods

import breeze.linalg.{*, DenseMatrix, DenseVector}
import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.ir.functions.MatrixToTableFunction
import is.hail.expr.ir.{ExecuteContext, MatrixValue, TableValue}
import is.hail.expr.types._
import is.hail.expr.types.physical.{PStruct, PType}
import is.hail.expr.types.virtual._
import is.hail.rvd.{RVD, RVDContext, RVDType}
import is.hail.sparkextras.ContextRDD
import is.hail.utils._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.sql.Row

case class PCA(entryField: String, k: Int, computeLoadings: Boolean) extends MatrixToTableFunction {
  override def typ(childType: MatrixType): TableType = {
    TableType(
      childType.rowKeyStruct ++ TStruct("loadings" -> TArray(TFloat64)),
      childType.rowKey,
      TStruct("eigenvalues" -> TArray(TFloat64), "scores" -> TArray(childType.colKeyStruct ++ TStruct("scores" -> TArray(TFloat64)))))
  }

  def preservesPartitionCounts: Boolean = false

  def execute(ctx: ExecuteContext, mv: MatrixValue): TableValue = {
    val hc = HailContext.get
    val sc = hc.sc

    if (k < 1)
      fatal(s"""requested invalid number of components: $k
               |  Expect componenents >= 1""".stripMargin)

    val rowMatrix = mv.toRowMatrix(entryField)
    val indexedRows = rowMatrix.rows.map { case (i, a) => IndexedRow(i, Vectors.dense(a)) }
      .cache()
    val irm = new IndexedRowMatrix(indexedRows, rowMatrix.nRows, rowMatrix.nCols)

    info(s"pca: running PCA with $k components...")

    val svd = irm.computeSVD(k, computeLoadings)
    if (svd.s.size < k)
      fatal(
        s"Found only ${ svd.s.size } non-zero (or nearly zero) eigenvalues, " +
          s"but user requested ${ k } principal components.")

    def collectRowKeys(): Array[Annotation] = {
      val rowKeyIdx = mv.typ.rowKeyFieldIdx
      val rowKeyTypes = mv.typ.rowKeyStruct.types

      mv.rvd.toUnsafeRows.map[Any] { r =>
        Row.fromSeq(rowKeyIdx.map(i => Annotation.copy(rowKeyTypes(i), r(i))))
      }
        .collect()
    }

    val rowType = PStruct.canonical(TStruct(mv.typ.rowKey.zip(mv.typ.rowKeyStruct.types): _*) ++ TStruct("loadings" -> TArray(TFloat64)))
    val rowKeysBc = HailContext.backend.broadcast(collectRowKeys())
    val localRowKeySignature = mv.typ.rowKeyStruct.types

    val crdd: ContextRDD[RegionValue] = if (computeLoadings) {
      ContextRDD.weaken(svd.U.rows).cmapPartitions { (ctx, it) =>
        val region = ctx.region
        val rv = RegionValue(region)
        val rvb = new RegionValueBuilder(region)
        it.map { ir =>
          rvb.start(rowType)
          rvb.startStruct()

          val rowKeys = rowKeysBc.value(ir.index.toInt).asInstanceOf[Row]
          var j = 0
          while (j < localRowKeySignature.length) {
            rvb.addAnnotation(localRowKeySignature(j), rowKeys.get(j))
            j += 1
          }

          rvb.startArray(k)
          var i = 0
          while (i < k) {
            rvb.addDouble(ir.vector(i))
            i += 1
          }
          rvb.endArray()
          rvb.endStruct()
          rv.setOffset(rvb.end())
          rv
        }
      }
    } else
      ContextRDD.empty(sc)
    val rvd = RVD.coerce(RVDType(rowType, mv.typ.rowKey), crdd, ctx)

    val (t1, f1) = mv.typ.globalType.insert(TArray(TFloat64), "eigenvalues")
    val (globalScoreType, f3) = mv.typ.colKeyStruct.insert(TArray(TFloat64), "scores")
    val (newGlobalType, f2) = t1.insert(TArray(globalScoreType), "scores")

    val data =
      if (!svd.V.isTransposed)
        svd.V.asInstanceOf[org.apache.spark.mllib.linalg.DenseMatrix].values
      else
        svd.V.toArray
    
    val V = new DenseMatrix[Double](svd.V.numRows, svd.V.numCols, data)
    val S = DenseVector(svd.s.toArray)

    val eigenvalues = svd.s.toArray.map(math.pow(_, 2))
    val scaledEigenvectors = V(*, ::) *:* S

    val scores = (0 until mv.nCols).iterator.map { i =>
      (0 until k).iterator.map { j => scaledEigenvectors(i, j) }.toFastIndexedSeq
    }.toFastIndexedSeq

    val g1 = f1(mv.globals.value, eigenvalues.toFastIndexedSeq)
    val globalScores = mv.colValues.safeJavaValue.zipWithIndex.map { case (cv, i) =>
      f3(mv.typ.extractColKey(cv.asInstanceOf[Row]), scores(i))
    }
    val newGlobal = f2(g1, globalScores)
    
    TableValue(TableType(rowType.virtualType, mv.typ.rowKey, newGlobalType.asInstanceOf[TStruct]),
      BroadcastRow(ctx, newGlobal.asInstanceOf[Row], newGlobalType.asInstanceOf[TStruct]), rvd)
  }
}
