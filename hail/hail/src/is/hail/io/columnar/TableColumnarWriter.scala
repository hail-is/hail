package is.hail.io.columnar

import is.hail.backend.ExecuteContext
import is.hail.expr.ir._
import is.hail.expr.ir.lowering.TableStage
import is.hail.types.RTable
import is.hail.types.physical.PStruct

/** Writes a table in the columnar format (`dev-docs/hail-query/columnar-format-spec.md`). */
case class TableColumnarWriter(path: String, overwrite: Boolean = false) extends TableWriter {

  // TableWriter.apply builds the RTable from the stage, so its key would be the partitioner key,
  // which may extend the table key. Give it the table key, as lowering a TableWrite does.
  override def apply(ctx: ExecuteContext, tv: TableValue): Unit = {
    val stage = TableValueIntermediate(tv).asTableStage(ctx)
    val r = RTable.fromTableStage(ctx, stage)
    CompileAndEvaluate(ctx, lower(ctx, stage, RTable(r.rowFields, r.globalFields, tv.typ.key)))
  }

  // `r.key` is the table key: when lowering a TableWrite, `r` is the requiredness of the child
  // TableIR.
  override def lower(ctx: ExecuteContext, stage: TableStage, r: RTable): IR =
    RelationalColumnarWriter.lower(
      ctx,
      stage,
      r.key.toIndexedSeq,
      r.rowType.canonicalPType(stage.rowType).asInstanceOf[PStruct],
      r.globalType.canonicalPType(stage.globalType).asInstanceOf[PStruct],
      path,
      overwrite,
      None,
    )
}
