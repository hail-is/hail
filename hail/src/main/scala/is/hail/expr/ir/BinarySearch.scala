package is.hail.expr.ir

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s._
import is.hail.expr.types.physical._

import scala.language.existentials

class BinarySearch[C](mb: EmitMethodBuilder[C], typ: PContainer, eltType: PType, keyOnly: Boolean) {

  val elt: PType = typ.elementType
  val ti: TypeInfo[_] = typeToTypeInfo(elt)

  val (compare: CodeOrdering.F[Int], equiv: CodeOrdering.F[Boolean], findElt: EmitMethodBuilder[C], t: PType) = if (keyOnly) {
    val ttype = elt match {
      case t: PBaseStruct =>
        require(t.size == 2)
        t
      case t: PInterval => t.representation.asInstanceOf[PStruct]
    }
    val kt = ttype.types(0)
    val findMB = mb.genEmitMethod("findElt", Array[TypeInfo[_]](typeInfo[Long], typeInfo[Boolean], typeToTypeInfo(kt)), typeInfo[Int])
    val mk2l = findMB.newLocal[Boolean]()
    val mk2l1 = mb.newLocal[Boolean]()

    val comp: CodeOrdering.F[Int] = {
      case ((mk1: Code[Boolean], k1: Code[_]), (m2: Code[Boolean], v2: Code[Long] @unchecked)) =>
        Code.memoize(v2, "bs_comp_v2") { v2 =>
          val mk2 = Code(mk2l := m2 || ttype.isFieldMissing(v2, 0), mk2l)
          val k2 = mk2l.mux(defaultValue(kt), Region.loadIRIntermediate(kt)(ttype.fieldOffset(v2, 0)))
          findMB.getCodeOrdering(eltType, kt, CodeOrdering.compare)((mk1, k1), (mk2, k2))
        }
    }
    val ceq: CodeOrdering.F[Boolean] = {
      case ((mk1: Code[Boolean], k1: Code[_]), (m2: Code[Boolean], v2: Code[Long] @unchecked)) =>
        Code.memoize(v2, "bs_comp_v2") { v2 =>
          val mk2 = Code(mk2l1 := m2 || ttype.isFieldMissing(v2, 0), mk2l1)
          val k2 = mk2l1.mux(defaultValue(kt), Region.loadIRIntermediate(kt)(ttype.fieldOffset(v2, 0)))
          mb.getCodeOrdering(eltType, kt, CodeOrdering.equiv)((mk1, k1), (mk2, k2))
        }
    }
    (comp, ceq, findMB, kt)
  } else
    (mb.getCodeOrdering(eltType, elt, CodeOrdering.compare),
      mb.getCodeOrdering(eltType, elt, CodeOrdering.equiv),
      mb.genEmitMethod("findElt", Array[TypeInfo[_]](typeInfo[Long], typeInfo[Boolean], typeToTypeInfo(elt)), typeInfo[Int]), elt)

  private[this] val array = findElt.getArg[Long](1)
  private[this] val m = findElt.getArg[Boolean](2)
  private[this] val e = findElt.getArg(3)(typeToTypeInfo(t))
  private[this] val len = findElt.newLocal[Int]()
  private[this] val i = findElt.newLocal[Int]()
  private[this] val low = findElt.newLocal[Int]()
  private[this] val high = findElt.newLocal[Int]()

  def cmp(i: Code[Int]): Code[Int] =
    Code.memoize(i, "binsearch_cmp_i") { i =>
      compare((m, e),
        (typ.isElementMissing(array, i),
          Region.loadIRIntermediate(elt)(typ.elementOffset(array, len, i))))
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
    findElt.invoke(array, m, v)
  }
}
