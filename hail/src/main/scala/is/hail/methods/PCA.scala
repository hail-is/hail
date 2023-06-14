package is.hail.methods

import breeze.linalg.{*, DenseMatrix, DenseVector}
import cats.implicits.toFlatMapOps
import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.ir.functions.MatrixToTableFunction
import is.hail.expr.ir.lowering.MonadLower
import is.hail.expr.ir.{MatrixValue, TableValue}
import is.hail.rvd.{RVD, RVDType}
import is.hail.sparkextras.ContextRDD
import is.hail.types._
import is.hail.types.physical.{PCanonicalStruct, PStruct}
import is.hail.types.virtual._
import is.hail.utils._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.sql.Row

import scala.language.higherKinds

case class PCA(entryField: String, k: Int, computeLoadings: Boolean) extends MatrixToTableFunction {
  override def typ(childType: MatrixType): TableType = {
    TableType(
      childType.rowKeyStruct ++ TStruct("loadings" -> TArray(TFloat64)),
      childType.rowKey,
      TStruct("eigenvalues" -> TArray(TFloat64), "scores" -> TArray(childType.colKeyStruct ++ TStruct("scores" -> TArray(TFloat64)))))
  }

  def preservesPartitionCounts: Boolean = false

  def execute[M[_]](mv: MatrixValue)(implicit M: MonadLower[M]): M[TableValue] =
    M.ask.flatMap { ctx =>
      if (k < 1)
        fatal(
          s"""requested invalid number of components: $k
             |  Expect componenents >= 1""".stripMargin)

      val rowMatrix = mv.toRowMatrix(entryField)
      val indexedRows = rowMatrix.rows.map { case (i, a) => IndexedRow(i, Vectors.dense(a)) }
        .cache()
      val irm = new IndexedRowMatrix(indexedRows, rowMatrix.nRows, rowMatrix.nCols)

      info(s"pca: running PCA with $k components...")

      val svd = irm.computeSVD(k, computeLoadings)
      if (svd.s.size < k)
        fatal(
          s"Found only ${svd.s.size} non-zero (or nearly zero) eigenvalues, " +
            s"but user requested ${k} principal components.")

      def collectRowKeys(): Array[Annotation] = {
        val rowKeyIdx = mv.typ.rowKeyFieldIdx
        val rowKeyTypes = mv.typ.rowKeyStruct.types

        mv.rvd.toUnsafeRows.map[Any] { r =>
          Row.fromSeq(rowKeyIdx.map(i => Annotation.copy(rowKeyTypes(i), r(i))))
        }
          .collect()
      }

      val rowType = PCanonicalStruct.canonical(TStruct(mv.typ.rowKey.zip(mv.typ.rowKeyStruct.types): _*) ++ TStruct("loadings" -> TArray(TFloat64)))
        .setRequired(true)
        .asInstanceOf[PStruct]
      val rowKeysBc = HailContext.backend.broadcast(collectRowKeys())
      val localRowKeySignature = mv.typ.rowKeyStruct.types

      val crdd: ContextRDD[Long] = if (computeLoadings) {
        ContextRDD.weaken(svd.U.rows).cmapPartitions { (ctx, it) =>
          val rvb = ctx.rvb
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

            rvb.end()
          }
        }
      } else
        ContextRDD.empty()
      val rvd = RVD.coerce(ctx, RVDType(rowType, mv.typ.rowKey), crdd)

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


      M.map(BroadcastRow(newGlobal.asInstanceOf[Row], newGlobalType.asInstanceOf[TStruct])) { br =>
        TableValue(
          TableType(rowType.virtualType, mv.typ.rowKey, newGlobalType.asInstanceOf[TStruct]),
          br,
          rvd
        )
      }
    }
}
