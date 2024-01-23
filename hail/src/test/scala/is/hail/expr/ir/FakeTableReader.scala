package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.lowering.TableStage
import is.hail.types.{TableType, VirtualTypeWithReq}
import is.hail.types.virtual.TStruct

class FakeTableReader extends TableReader {
  override def pathsUsed: Seq[String] = ???
  override def partitionCounts: Option[IndexedSeq[Long]] = ???
  override def fullType: TableType = ???

  override def rowRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq =
    ???

  override def globalRequiredness(ctx: ExecuteContext, requestedType: TableType)
    : VirtualTypeWithReq = ???

  override def renderShort(): String = ???
  override def lowerGlobals(ctx: ExecuteContext, requestedGlobalsType: TStruct): IR = ???
  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage = ???
}
