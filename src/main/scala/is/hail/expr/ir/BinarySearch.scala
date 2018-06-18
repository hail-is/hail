package is.hail.expr.ir

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.utils._

class BinarySearch(mb: EmitMethodBuilder, typ: TContainer, keyOnly: Boolean) {

  val elt: Type = typ.elementType
  val ti: TypeInfo[_] = typeToTypeInfo(elt)

  val (compare: CodeOrdering.F[Int], equiv: CodeOrdering.F[Boolean], findElt: EmitMethodBuilder, t: Type) = if (keyOnly) {
    val ttype = coerce[TBaseStruct](elt)
    require(ttype.size == 2)
    val kt = ttype.types(0)
    val findMB = mb.fb.newMethod(Array[TypeInfo[_]](typeInfo[Region], typeInfo[Long], typeInfo[Boolean], typeToTypeInfo(kt)), typeInfo[Int])
    val mk2l = findMB.newLocal[Boolean]
    val mk2l1 = mb.newLocal[Boolean]
    val v2l = findMB.newLocal[Long]
    val v2l1 = mb.newLocal[Long]

    val comp: CodeOrdering.F[Int] = {
      case (r1: Code[Region],
      (mk1: Code[Boolean], k1: Code[_]),
      r2: Code[Region],
      (m2: Code[Boolean], v2: Code[Long] @unchecked)) =>
        val mk2 = Code(mk2l := m2 || ttype.isFieldMissing(r2, v2, 0), mk2l)
        val k2 = mk2l.mux(defaultValue(kt), r2.loadIRIntermediate(kt)(ttype.fieldOffset(v2, 0)))
        findMB.getCodeOrdering[Int](kt, CodeOrdering.compare, missingGreatest = true)(r1, (mk1, k1), r2, (mk2, k2))
    }
    val ceq: CodeOrdering.F[Boolean] = {
      case (r1: Code[Region],
      (mk1: Code[Boolean], k1: Code[_]),
      r2: Code[Region],
      (m2: Code[Boolean], v2: Code[Long] @unchecked)) =>
        val mk2 = Code(mk2l1 := m2 || ttype.isFieldMissing(r2, v2, 0), mk2l1)
        val k2 = mk2l1.mux(defaultValue(kt), r2.loadIRIntermediate(kt)(ttype.fieldOffset(v2, 0)))
        mb.getCodeOrdering[Boolean](kt, CodeOrdering.equiv, missingGreatest = true)(r1, (mk1, k1), r2, (mk2, k2))
    }
    (comp, ceq, findMB, kt)
  } else
    (mb.getCodeOrdering[Int](elt, CodeOrdering.compare, missingGreatest = true),
      mb.getCodeOrdering[Boolean](elt, CodeOrdering.equiv, missingGreatest = true),
      mb.fb.newMethod(Array[TypeInfo[_]](typeInfo[Region], typeInfo[Long], typeInfo[Boolean], typeToTypeInfo(elt)), typeInfo[Int]), elt)

  private[this] val region = findElt.getArg[Region](1).load()
  private[this] val array = findElt.getArg[Long](2)
  private[this] val m = findElt.getArg[Boolean](3)
  private[this] val e = findElt.getArg(4)(typeToTypeInfo(t))
  private[this] val len = findElt.newLocal[Int]
  private[this] val i = findElt.newLocal[Int]
  private[this] val low = findElt.newLocal[Int]
  private[this] val high = findElt.newLocal[Int]

  def cmp(i: Code[Int]): Code[Int] =
    compare(region, (m, e),
      region, (typ.isElementMissing(region, array, i),
        region.loadIRIntermediate(elt)(typ.elementOffset(array, len, i))))

  // return smallest i such that elem <= array(i)
  findElt.emit(Code(
    len := typ.loadLength(region, array),
    low := 0,
    high := len - 1,
    Code.whileLoop(low < high,
      i := (low + high) / 2,
      (cmp(i) <= 0).mux(
        high := i,
        low := i + 1)),
    low))

  def getClosestIndex(array: Code[Long], m: Code[Boolean], v: Code[_]): Code[Int] = {
    val region = mb.getArg[Region](1).load()
    findElt.invoke(region, array, m, v)
  }
}