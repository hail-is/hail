package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.lowering.TableStage
import is.hail.types.VirtualTypeWithReq
import is.hail.types.virtual.{TStruct, TableType}
import org.json4s.{Formats, JValue}

class FakeTableReader extends TableReader {
  override def pathsUsed: Seq[String] = ???
  override def partitionCounts: Option[IndexedSeq[Long]] = ???
  override def fullType: TableType = ???

  override def rowRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq =
    ???

  override def globalRequiredness(ctx: ExecuteContext, requestedType: TableType)
    : VirtualTypeWithReq = ???

  override def pretty: PrettyOps =
    new PrettyOps {
      override def toJValue(implicit fmts: Formats): JValue = ???
      override def renderShort: String = "FakeTableReader"
    }

  override def lowerGlobals(ctx: ExecuteContext, requestedGlobalsType: TStruct): IR = ???
  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage = ???
}
