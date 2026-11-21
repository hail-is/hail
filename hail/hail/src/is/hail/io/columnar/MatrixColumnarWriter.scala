package is.hail.io.columnar

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{IR, MatrixWriter}
import is.hail.expr.ir.lowering.TableStage
import is.hail.types.RTable
import is.hail.types.physical.PStruct
import is.hail.types.virtual.MatrixType
import is.hail.utils.fatal

/** Writes a matrix table in the columnar format (`dev-docs/hail-query/columnar-format-spec.md`),
  * with entry files in the dense layout and one column block.
  *
  * `leafCap` is the most values a leaf field of an entry file may hold in one row group. It is at
  * most the spec's cap, 2^30^, and tests lower it.
  */
case class MatrixColumnarWriter(
  path: String,
  overwrite: Boolean = false,
  leafCap: Long = ColumnarPartitionWriter.MaxLeafCap,
) extends MatrixWriter {

  // `r.key` is the matrix table's row key (see WrappedMatrixWriter.apply).
  override def lower(
    colsFieldName: String,
    entriesFieldName: String,
    colKey: IndexedSeq[String],
    ctx: ExecuteContext,
    ts: TableStage,
    r: RTable,
  ): IR = {
    if (leafCap < 1 || leafCap > ColumnarPartitionWriter.MaxLeafCap)
      fatal(
        s"columnar writer: the leaf cap must be between 1 and " +
          s"${ColumnarPartitionWriter.MaxLeafCap}, found $leafCap"
      )
    val tm = MatrixType.fromTableType(ts.tableType, colsFieldName, entriesFieldName, colKey)
    val rm = r.asMatrixType(colsFieldName, entriesFieldName)
    RelationalColumnarWriter.lower(
      ctx,
      ts,
      r.key.toIndexedSeq,
      rm.rowType.canonicalPType(tm.rowType).asInstanceOf[PStruct],
      rm.globalType.canonicalPType(tm.globalType).asInstanceOf[PStruct],
      path,
      overwrite,
      Some(RelationalColumnarWriter.MatrixParts(
        colsField = colsFieldName,
        colKey = colKey,
        colPType = rm.colType.canonicalPType(tm.colType).asInstanceOf[PStruct],
        entriesField = entriesFieldName,
        entryPType = rm.entryType.canonicalPType(tm.entryType).asInstanceOf[PStruct],
        leafCap = leafCap,
      )),
    )
  }
}
