package is.hail.expr.ir.functions

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.TableValue
import is.hail.rvd.RVD
import is.hail.types
import is.hail.types.virtual._
import is.hail.utils._

case class TableCalculateNewPartitions(
  nPartitions: Int
) extends TableToValueFunction {
  def typ(childType: types.TableType): Type = TArray(TInterval(childType.keyType))

  def unionRequiredness(childType: types.RTable, resultType: types.TypeWithRequiredness): Unit = {
    val rinterval = types.tcoerce[types.RInterval](
      types.tcoerce[types.RIterable](resultType).elementType
    )
    val rstart = types.tcoerce[types.RStruct](rinterval.startType)
    val rend = types.tcoerce[types.RStruct](rinterval.endType)
    childType.keyFields.foreach { k =>
      rstart.field(k).unionFrom(childType.field(k))
      rend.field(k).unionFrom(childType.field(k))
    }
  }

  def execute(ctx: ExecuteContext, tv: TableValue): Any = {
    val rvd = tv.rvd
    if (rvd.typ.key.isEmpty)
      FastSeq()
    else {
      val ki = RVD.getKeyInfo(ctx, rvd.typ, rvd.typ.key.length, RVD.getKeys(ctx, rvd.typ, rvd.crdd))
      if (ki.isEmpty)
        FastSeq()
      else
        RVD.calculateKeyRanges(
          ctx,
          rvd.typ,
          ki,
          nPartitions,
          rvd.typ.key.length,
        ).rangeBounds.toIndexedSeq
    }
  }
}
