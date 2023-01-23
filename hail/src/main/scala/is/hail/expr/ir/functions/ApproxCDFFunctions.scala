package is.hail.expr.ir.functions

import is.hail.annotations.Region
import is.hail.asm4s.{Code, TypeInfo, Value, toCodeObject, valueToCodeObject}
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, IEmitCode}
import is.hail.expr.ir.agg.{ApproxCDFCombiner, QuantilesAggregator}
import is.hail.types.physical.{PCanonicalArray, PCanonicalStruct, PFloat64, PInt32, PInt64}
import is.hail.types.physical.stypes.concrete.SBaseStructPointer
import is.hail.types.physical.stypes.interfaces.{SBaseStructValue, SIndexableValue, primitive}
import is.hail.utils.FastIndexedSeq

import scala.reflect.ClassTag

object ApproxCDFFunctions extends RegistryFunctions {
  val stateType = QuantilesAggregator.resultPType.virtualType
  val resultPType: PCanonicalStruct =
    PCanonicalStruct(required = false,
      "values" -> PCanonicalArray(PFloat64(true), required = true),
      "ranks" -> PCanonicalArray(PInt64(true), required = true),
      "_compaction_counts" -> PCanonicalArray(PInt32(true), required = true))
  val resultSType = SBaseStructPointer(resultPType)

  def sIndexableFromJava[A: ClassTag: TypeInfo](cb: EmitCodeBuilder, r: Value[Region], array: Value[IndexedSeq[A]], pt: PCanonicalArray): SIndexableValue = {
    pt.constructFromElements(cb, r, cb.memoize(array.invoke[Int]("length")), true) { (cb, i) =>
      IEmitCode.present(cb, primitive(pt.elementType.virtualType, cb.memoize(array.invoke[Int, A]("apply", i))))
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
        val javaItems = cb.memoize(Code.checkcast[IndexedSeq[Double]](svalueToJavaValue(cb, r.region, items)).invoke[Array[Double]]("toArray"))
        val javaCounts = cb.memoize(Code.checkcast[IndexedSeq[Int]](svalueToJavaValue(cb, r.region, counts)).invoke[Array[Int]]("toArray"))
        val combiner = cb.memoize(Code.newInstance[ApproxCDFCombiner, Array[Int], Array[Double], Array[Int], Int, java.util.Random](
          javaLevels,
          javaItems,
          javaCounts,
          levels.loadLength() - 1,
          Code.newInstance[java.util.Random]()))
        val javaCDF = cb.memoize(combiner.invoke[(Array[Double], Array[Long])]("computeCDF"))
        val javaValues = cb.memoize(Code.checkcast[IndexedSeq[Double]](javaCDF.invoke[Array[Double]]("_1")))
        val javaRanks = cb.memoize(Code.checkcast[IndexedSeq[Long]](javaCDF.invoke[Array[Long]]("_2")))
        val valuesPType = resultPType.field("values").typ.asInstanceOf[PCanonicalArray]
        val ranksPType = resultPType.field("ranks").typ.asInstanceOf[PCanonicalArray]
        val values = sIndexableFromJava(cb, r.region, javaValues, valuesPType)
        val ranks = sIndexableFromJava(cb, r.region, javaRanks, ranksPType)

        resultPType.constructFromFields(cb, r.region, FastIndexedSeq(EmitCode.present(cb.emb, values), EmitCode.present(cb.emb, ranks), EmitCode.present(cb.emb, counts)), false)
    }
  }
}
