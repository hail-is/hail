package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.types.physical._
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.interfaces.{SBaseStruct, SBaseStructCode, SContainer, SInterval, SIntervalCode}
import is.hail.utils.FastIndexedSeq

import scala.language.existentials

class BinarySearch[C](mb: EmitMethodBuilder[C], containerType: SContainer, eltType: EmitType, keyOnly: Boolean) {

  val containerElementType: EmitType = containerType.elementEmitType

  val (compare: CodeOrdering.F[Int], equiv: CodeOrdering.F[Boolean], findElt: EmitMethodBuilder[C]) = if (keyOnly) {
    val kt: EmitType = containerElementType.st match {
      case s: SBaseStruct =>
        require(s.size == 2)
        s.fieldEmitTypes(0)
      case interval: SInterval =>
        interval.pointEmitType
    }
    val findMB = mb.genEmitMethod("findElt", FastIndexedSeq[ParamType](containerType.paramType, eltType.paramType), typeInfo[Int])

    val comp: CodeOrdering.F[Int] = {
      (cb: EmitCodeBuilder, ec1: EmitCode, _ec2: EmitCode) =>
        val ec2 = EmitCode.fromI(cb.emb) { cb =>
          val iec = _ec2.toI(cb)
          iec.flatMap(cb) {
            case v2: SBaseStructCode =>
              v2.memoize(cb, "bs_comp_v2").loadField(cb, 0)
            case v2: SIntervalCode =>
              v2.memoize(cb, "bs_comp_v2").loadStart(cb)
          }
        }
        findMB.ecb.getOrderingFunction(eltType.st, kt.st, CodeOrdering.Compare())(cb, ec1, ec2)
    }
    val ceq: CodeOrdering.F[Boolean] = {
      (cb: EmitCodeBuilder, ec1: EmitCode, _ec2: EmitCode) =>
        val ec2 = EmitCode.fromI(cb.emb) { cb =>
          val iec = _ec2.toI(cb)
          iec.flatMap(cb) {
            case v2: SBaseStructCode =>
              v2.memoize(cb, "bs_eq_v2").loadField(cb, 0)
            case v2: SIntervalCode =>
              v2.memoize(cb, "bs_comp_v2").loadStart(cb)
          }
        }
      findMB.ecb.getOrderingFunction(eltType.st, kt.st, CodeOrdering.Equiv())(cb, ec1, ec2)
    }
    (comp, ceq, findMB)
  } else
    (mb.ecb.getOrderingFunction(eltType.st, containerElementType.st, CodeOrdering.Compare()),
      mb.ecb.getOrderingFunction(eltType.st, containerElementType.st, CodeOrdering.Equiv()),
      mb.genEmitMethod("findElt", FastIndexedSeq[ParamType](containerType.paramType, eltType.paramType), typeInfo[Int]))

  // Returns smallest i, 0 <= i < n, for which a(i) >= key, or returns n if a(i) < key for all i
  findElt.emitWithBuilder[Int] { cb =>
    val indexable = findElt.getPCodeParam(1).asIndexable.memoize(cb, "findElt_indexable")

    val elt = findElt.getEmitParam(2, null) // no streams

    val len = cb.newLocal[Int]("findelt_length", indexable.loadLength())
    val low = cb.newLocal("findelt_low", 0)
    val high = cb.newLocal("findelt_high", len)

    cb.whileLoop(low < high, {
      val i = cb.newLocal("findelt_i", (low + high) / 2)
      cb.ifx(compare(cb, elt, EmitCode.fromI(findElt)(cb => indexable.loadElement(cb, i))) <= 0,
        cb.assign(high, i),
        cb.assign(low, i + 1)
      )
    })
    low
  }

  // check missingness of v before calling
  def getClosestIndex(cb: EmitCodeBuilder, array: SCode, v: EmitCode): Code[Int] = {
    cb.invokeCode[Int](findElt, array, v)
  }
}
