package is.hail.expr.ir.functions

import is.hail.annotations.Region
import is.hail.asm4s.{Code, Value, valueToCodeObject}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.expr.ir.agg.{ApproxCDFStateManager, QuantilesAggregator}
import is.hail.types.physical._
import is.hail.types.physical.stypes.concrete.SBaseStructPointer
import is.hail.types.physical.stypes.interfaces.SBaseStructValue
import is.hail.types.physical.stypes.primitives.SInt32Value
import is.hail.types.virtual.TInt32
import is.hail.utils.toRichIterable
import org.apache.spark.sql.Row

object ApproxCDFFunctions extends RegistryFunctions {
  val statePType = QuantilesAggregator.resultPType
  val stateType = statePType.virtualType

  def rowToStateManager(k: Int, row: Row): ApproxCDFStateManager = {
    val levels = row.getAs[IndexedSeq[Int]](0).toArray
    val items = row.getAs[IndexedSeq[Double]](1).toArray
    val counts = row.getAs[IndexedSeq[Int]](2).toArray
    ApproxCDFStateManager.fromData(k, levels, items, counts)
  }

  def makeStateManager(cb: EmitCodeBuilder, r: Value[Region], k: Value[Int], state: SBaseStructValue): Value[ApproxCDFStateManager] = {
    val row = svalueToJavaValue(cb, r, state)
    cb.memoize(Code.invokeScalaObject2[Int, Row, ApproxCDFStateManager](
      ApproxCDFFunctions.getClass, "rowToStateManager",
      k, Code.checkcast[Row](row)))
  }

  def stateManagerToRow(state: ApproxCDFStateManager): Row = {
    Row(state.levels.toFastSeq, state.items.toFastSeq, state.compactionCounts.toFastSeq)
  }

  def fromStateManager(cb: EmitCodeBuilder, r: Value[Region], state: Value[ApproxCDFStateManager]): SBaseStructValue = {
    val row = cb.memoize(Code.invokeScalaObject1[ApproxCDFStateManager, Row](
      ApproxCDFFunctions.getClass, "stateManagerToRow",
      state))
    unwrapReturn(cb, r, SBaseStructPointer(statePType), row).asBaseStruct
  }

  def registerAll(): Unit = {
    registerSCode3("approxCDFCombine", TInt32, stateType, stateType, stateType, (_, _, _, _) => SBaseStructPointer(statePType)) {
      case (r, cb, rt, k: SInt32Value, left: SBaseStructValue, right: SBaseStructValue, errorID) =>
        val leftState = makeStateManager(cb, r.region, k.value, left)
        val rightState = makeStateManager(cb, r.region, k.value, right)

        cb += leftState.invoke[ApproxCDFStateManager, Unit]("combOp", rightState)

        fromStateManager(cb, r.region, leftState)
    }
  }
}
