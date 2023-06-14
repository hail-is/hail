package is.hail.expr.ir
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.lowering.MonadLower
import is.hail.types.{TableType, VirtualTypeWithReq}

import scala.language.higherKinds

class FakeTableReader extends TableReader {
  override def pathsUsed: Seq[String] = ???
  override def apply[M[_] : MonadLower](requestedType: TableType, dropRows: Boolean): M[TableValue] = ???
  override def partitionCounts: Option[IndexedSeq[Long]] = ???
  override def fullType: TableType = ???
  override def rowRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq = ???
  override def globalRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq = ???
  override def renderShort(): String = ???
}
