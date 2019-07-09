package is.hail.expr.ir.functions

import is.hail.expr.ir.{ExecuteContext, TableValue}
import is.hail.expr.types.virtual._
import is.hail.expr.types.TableType
import is.hail.rvd.RVD
import is.hail.utils._

case class TableCalculateNewPartitions(
  nPartitions: Int
) extends TableToValueFunction {
  def typ(childType: TableType): Type = TArray(TInterval(childType.keyType))

  def execute(ctx: ExecuteContext, tv: TableValue): Any = {
    val rvd = tv.rvd
    if (rvd.typ.key.isEmpty)
      FastIndexedSeq()
    else {
      val ki = RVD.getKeyInfo(rvd.typ, rvd.typ.key.length, RVD.getKeys(rvd.typ, rvd.crdd))
      if (ki.isEmpty)
        FastIndexedSeq()
      else
        RVD.calculateKeyRanges(rvd.typ, ki, nPartitions, rvd.typ.key.length).rangeBounds.toIndexedSeq
    }
  }
}
