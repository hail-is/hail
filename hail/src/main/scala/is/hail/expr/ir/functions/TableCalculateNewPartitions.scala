package is.hail.expr.ir.functions

import is.hail.expr.ir.{ExecuteContext, TableValue}
import is.hail.types
import is.hail.types.virtual._
import is.hail.rvd.RVD
import is.hail.utils._

case class TableCalculateNewPartitions(
  nPartitions: Int
) extends TableToValueFunction {
  def typ(childType: types.TableType): Type = TArray(TInterval(childType.keyType))

  def unionRequiredness(childType: types.RTable, resultType: types.TypeWithRequiredness): Unit = {
    val rinterval = types.coerce[types.RInterval](
      types.coerce[types.RIterable](resultType).elementType)
    val rstart = types.coerce[types.RStruct](rinterval.startType)
    val rend = types.coerce[types.RStruct](rinterval.endType)
    childType.keyFields.foreach { k =>
      rstart.field(k).unionFrom(childType.field(k))
      rend.field(k).unionFrom(childType.field(k))
    }
  }

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
