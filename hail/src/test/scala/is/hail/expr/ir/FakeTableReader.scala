package is.hail.expr.ir
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.analyses.SemanticHash.NextHash
import is.hail.types.{TableType, VirtualTypeWithReq}

class FakeTableReader extends TableReader {
  override def pathsUsed: Seq[String] = ???
  override def apply(ctx: ExecuteContext, requestedType: TableType, dropRows: Boolean): TableValue = ???
  override def partitionCounts: Option[IndexedSeq[Long]] = ???
  override def fullType: TableType = ???
  override def rowRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq = ???
  override def globalRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq = ???
  override def renderShort(): String = ???
}
