package is.hail.expr.ir

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.types.physical._
import is.hail.types.physical.stypes._
import is.hail.utils.FastIndexedSeq

import scala.language.existentials

class BinarySearch[C](mb: EmitMethodBuilder[C], typ: PContainer, eltType: PType, keyOnly: Boolean) {

  val elt: PType = typ.elementType
  val ti: TypeInfo[_] = typeToTypeInfo(elt)

  val (compare: CodeOrdering.F[Int], equiv: CodeOrdering.F[Boolean], findElt: EmitMethodBuilder[C]) = if (keyOnly) {
    val kt = elt match {
      case t: PBaseStruct =>
        require(t.size == 2)
        t.types(0)
      case t: PCanonicalInterval =>
        t.pointType
    }
    val findMB = mb.genEmitMethod("findElt", FastIndexedSeq[ParamType](typeInfo[Long], typeInfo[Boolean], typeToTypeInfo(kt)), typeInfo[Int])

    val comp: CodeOrdering.F[Int] = {
      (cb: EmitCodeBuilder, ec1: EmitCode, _ec2: EmitCode) =>
        val ec2 = EmitCode.fromI(cb.emb) { cb =>
          val iec = _ec2.toI(cb)
          iec.flatMap(cb) {
            case v2: PBaseStructCode =>
              v2.memoize(cb, "bs_comp_v2").loadField(cb, 0)
            case v2: PIntervalCode =>
              v2.memoize(cb, "bs_comp_v2").loadStart(cb)
          }.map(cb)(_.asPCode)
        }
        findMB.ecb.getOrderingFunction(eltType.sType, kt.sType, CodeOrdering.Compare())(cb, ec1, ec2)
    }
    val ceq: CodeOrdering.F[Boolean] = {
      (cb: EmitCodeBuilder, ec1: EmitCode, _ec2: EmitCode) =>
        val ec2 = EmitCode.fromI(cb.emb) { cb =>
          val iec = _ec2.toI(cb)
          iec.flatMap(cb) {
            case v2: PBaseStructCode =>
              v2.memoize(cb, "bs_eq_v2").loadField(cb, 0)
            case v2: PIntervalCode =>
              v2.memoize(cb, "bs_comp_v2").loadStart(cb)
          }.map(cb)(_.asPCode)
        }
      findMB.ecb.getOrderingFunction(eltType.sType, kt.sType, CodeOrdering.Equiv())(cb, ec1, ec2)
    }
    (comp, ceq, findMB)
  } else
    (mb.ecb.getOrderingFunction(eltType.sType, elt.sType, CodeOrdering.Compare()),
      mb.ecb.getOrderingFunction(eltType.sType, elt.sType, CodeOrdering.Equiv()),
      mb.genEmitMethod("findElt", FastIndexedSeq[ParamType](typeInfo[Long], typeInfo[Boolean], elt.ti), typeInfo[Int]))

  private[this] val array = findElt.getCodeParam[Long](1)
  private[this] val m = findElt.getCodeParam[Boolean](2)
  private[this] val e = findElt.getCodeParam(3)(eltType.ti)
  private[this] val len = findElt.newLocal[Int]()
  private[this] val i = findElt.newLocal[Int]()
  private[this] val low = findElt.newLocal[Int]()
  private[this] val high = findElt.newLocal[Int]()

  def cmp(i: Code[Int]): Code[Int] = EmitCodeBuilder.scopedCode(findElt) { cb =>
    val ec1 = EmitCode(Code._empty, m, PCode(eltType, e))
    val ec2 = EmitCode.fromI(findElt) { cb =>
      PCode(typ, array).asIndexable.memoize(cb, "binsearch_cmp_i").loadElement(cb, i).map(cb)(_.asPCode)
    }
    compare(cb, ec1, ec2)
  }

  // Returns smallest i, 0 <= i < n, for which a(i) >= key, or returns n if a(i) < key for all i
  findElt.emit(Code(
    len := typ.loadLength(array),
    low := 0,
    high := len,
    Code.whileLoop(low < high,
      i := (low + high) / 2,
      (cmp(i) <= 0).mux(
        high := i,
        low := i + 1)),
    low))

  // check missingness of v before calling
  def getClosestIndex(array: Code[Long], m: Code[Boolean], v: Code[_]): Code[Int] = {
    findElt.invokeCode[Int](array, m, v)
  }
}
