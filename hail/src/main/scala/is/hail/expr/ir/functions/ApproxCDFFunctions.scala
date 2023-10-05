package is.hail.expr.ir.functions

import is.hail.annotations.Region
import is.hail.asm4s.{Code, TypeInfo, Value, toCodeObject, valueToCodeObject}
import is.hail.expr.ir.agg.{ApproxCDFCombiner, QuantilesAggregator}
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.stypes.concrete.SBaseStructPointer
import is.hail.types.physical.stypes.interfaces.{SBaseStructValue, SIndexableValue, primitive}
import is.hail.types.physical._
import is.hail.utils.FastSeq

import scala.reflect.ClassTag

object ApproxCDFFunctions extends RegistryFunctions {
  val stateType = QuantilesAggregator.resultPType.virtualType
  val resultPType: PCanonicalStruct =
    PCanonicalStruct(required = false,
      "values" -> PCanonicalArray(PFloat64(true), required = true),
      "ranks" -> PCanonicalArray(PInt64(true), required = true),
      "_compaction_counts" -> PCanonicalArray(PInt32(true), required = true))
  val resultSType = SBaseStructPointer(resultPType)

  def sIndexableFromArray[A: ClassTag: TypeInfo](cb: EmitCodeBuilder, r: Value[Region], array: Value[Array[A]], pt: PCanonicalArray): SIndexableValue = {
    pt.constructFromElements(cb, r, cb.memoize(array.length()), true) { (cb, i) =>
      IEmitCode.present(cb, primitive(pt.elementType.virtualType, cb.memoize(array(i))))
    }
  }

  def indexedSeqToArrayInt(seq: IndexedSeq[Int]): Array[Int] = seq.toArray
  def indexedSeqToArrayDouble(seq: IndexedSeq[Double]): Array[Double] = seq.toArray

  def registerAll(): Unit = {
    registerSCode1("approxCDFResult", stateType, resultPType.virtualType, (_, _) => resultSType) {
      case (r, cb, rt, state: SBaseStructValue, errorID) =>
        val levels = state.loadField(cb, "levels").get(cb).asIndexable
        val items = state.loadField(cb, "items").get(cb).asIndexable
        val counts = state.loadField(cb, "_compaction_counts").get(cb).asIndexable
        val javaLevels = cb.memoize(Code.invokeScalaObject1[IndexedSeq[Int], Array[Int]](ApproxCDFFunctions.getClass, "indexedSeqToArrayInt",
          Code.checkcast[IndexedSeq[Int]](svalueToJavaValue(cb, r.region, levels))))
        val javaItems = cb.memoize(Code.invokeScalaObject1[IndexedSeq[Double], Array[Double]](ApproxCDFFunctions.getClass, "indexedSeqToArrayDouble",
          Code.checkcast[IndexedSeq[Double]](svalueToJavaValue(cb, r.region, items))))
        val javaCounts = cb.memoize(Code.invokeScalaObject1[IndexedSeq[Int], Array[Int]](ApproxCDFFunctions.getClass, "indexedSeqToArrayInt",
          Code.checkcast[IndexedSeq[Int]](svalueToJavaValue(cb, r.region, counts))))
        val combiner = cb.memoize(Code.newInstance[ApproxCDFCombiner, Array[Int], Array[Double], Array[Int], Int, java.util.Random](
          javaLevels,
          javaItems,
          javaCounts,
          levels.loadLength() - 1,
          Code.newInstance[java.util.Random]()))
        val javaCDF = cb.memoize(combiner.invoke[(Array[Double], Array[Long])]("computeCDF"))
        val javaValues = cb.memoize(javaCDF.invoke[Array[Double]]("_1"))
        val javaRanks = cb.memoize(javaCDF.invoke[Array[Long]]("_2"))
        val valuesPType = resultPType.field("values").typ.asInstanceOf[PCanonicalArray]
        val ranksPType = resultPType.field("ranks").typ.asInstanceOf[PCanonicalArray]
        val values = sIndexableFromArray(cb, r.region, javaValues, valuesPType)
        val ranks = sIndexableFromArray(cb, r.region, javaRanks, ranksPType)

        resultPType.constructFromFields(cb, r.region, FastSeq(EmitCode.present(cb.emb, values), EmitCode.present(cb.emb, ranks), EmitCode.present(cb.emb, counts)), false)
    }
  }
}
