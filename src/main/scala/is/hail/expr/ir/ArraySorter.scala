package is.hail.expr.ir

import is.hail.annotations.{CodeOrdering, Region, StagedRegionValueBuilder}
import is.hail.expr.types._
import is.hail.asm4s._
import is.hail.utils._

class ArraySorter(mb: EmitMethodBuilder, array: StagedArrayBuilder, keyOnly: Boolean = false) {
  val typ: Type = array.elt
  val ti: TypeInfo[_] = typeToTypeInfo(typ)

  val compare: EmitMethodBuilder = mb.fb.newMethod(Array[TypeInfo[_]](typeInfo[Region], BooleanInfo, ti, BooleanInfo, ti), IntInfo)
  if (keyOnly) {
    require(typ.isInstanceOf[TBaseStruct] && coerce[TBaseStruct](typ).size == 2)
    val ttype = coerce[TBaseStruct](typ)
    val ktype = ttype.types(0)
    val kord: CodeOrdering = CodeOrdering(ktype, missingGreatest = true)
    val cregion: Code[Region] = compare.getArg[Region](1)

    val km1 = ttype.isFieldMissing(cregion, compare.getArg[Long](3), 0)
    val kv1 = cregion.loadIRIntermediate(ktype)(ttype.fieldOffset(compare.getArg[Long](3), 0))
    val km2 = ttype.isFieldMissing(cregion, compare.getArg[Long](5), 0)
    val kv2 = cregion.loadIRIntermediate(ktype)(ttype.fieldOffset(compare.getArg[Long](5), 0))

    val c = compare.getArg[Boolean](2).ceq(compare.getArg[Boolean](4)).mux(
      compare.getArg[Boolean](2).mux(0, kord.compare(compare, (km1, kv1), (km2, kv2))),
      compare.getArg[Boolean](2).mux(1, -1)
    )
    compare.emit(c)
  } else {
    val ord: CodeOrdering = CodeOrdering(typ, missingGreatest = true)
    val x1: (Code[Boolean], Code[_]) = (compare.getArg[Boolean](2), compare.getArg(3)(ti))
    val x2: (Code[Boolean], Code[_]) = (compare.getArg[Boolean](4), compare.getArg(5)(ti))
    compare.emit(ord.compare(compare, x1, x2))
  }

  def sort(): Code[Unit] = {

    val sort = mb.fb.newMethod[Region, Int, Int, Unit]
    val region = sort.getArg[Region](1)
    val start = sort.getArg[Int](2)
    val end = sort.getArg[Int](3)

    val pi: LocalRef[Int] = sort.newLocal[Int]
    val i: LocalRef[Int] = sort.newLocal[Int]

    val m1: LocalRef[Boolean] = sort.newLocal[Boolean]
    val v1: LocalRef[_] = sort.newLocal(ti)

    def loadPivot(start: Code[Int], end: Code[Int]): Code[Unit] =
      pi := end

    def lt(m1: Code[Boolean], v1: Code[_], m2: Code[Boolean], v2: Code[_]): Code[Boolean] =
      coerce[Int](compare.invoke(region, m1, v1, m2, v2)) < 0

    def swap(i: Code[Int], j: Code[Int]): Code[Unit] = {
      Code(
        m1 := array.isMissing(i),
        v1.storeAny(array(i)),
        array.setMissing(i, array.isMissing(j)),
        array.isMissing(i).mux(Code._empty, array.update(i, array(j))),
        array.setMissing(j, m1),
        m1.mux(Code._empty, array.update(j, v1)))
    }

    sort.emit(Code(
      loadPivot(start, end),
      i := start,
      Code.whileLoop(i < pi,
        lt(array.isMissing(pi), array(pi), array.isMissing(i), array(i)).mux(
          Code(
            i.ceq(pi - 1).mux(
              Code(swap(i, pi), i += 1),
              Code(swap(pi, pi - 1), swap(i, pi))),
            pi += -1),
          i += 1)),
      (start < pi - 1).mux(sort.invoke(region, start, pi - 1), Code._empty),
      (pi + 1 < end).mux(sort.invoke(region, pi + 1, end), Code._empty)))

    sort.invoke(mb.getArg[Region](1), 0, array.size - 1)
  }

  def toRegion(): Code[Long] = {
    val srvb = new StagedRegionValueBuilder(mb, TArray(typ))
    Code(
      srvb.start(array.size),
      Code.whileLoop(srvb.arrayIdx < array.size,
        array.isMissing(srvb.arrayIdx).mux(
          srvb.setMissing(),
          srvb.addIRIntermediate(typ)(array(srvb.arrayIdx))),
        srvb.advance()),
      srvb.end())
  }

  def distinctFromSorted: Code[Unit] = {
    def ceq(m1: Code[Boolean], v1: Code[_], m2: Code[Boolean], v2: Code[_]): Code[Boolean] = {
      coerce[Int](compare.invoke(mb.getArg[Region](1), m1, v1, m2, v2)).ceq(0)
    }

    val i = mb.newLocal[Int]
    val n = mb.newLocal[Int]

    val removeMissing = Code(i := array.size - 1,
      Code.whileLoop(i > 0 && array.isMissing(i), i += -1),
      array.size.ceq(i + 1).mux(Code._empty, array.setSize(i + 1)))

    Code(
      if (keyOnly) removeMissing else Code._empty,
      n := 0,
      i := 0,
      Code.whileLoop(i < array.size,
        Code.whileLoop(i < array.size && ceq(array.isMissing(n), array(n), array.isMissing(i), array(i)),
          i += 1),
        (i < array.size && i.cne(n + 1)).mux(
          Code(
            array.setMissing(n + 1, array.isMissing(i)),
            array.isMissing(n + 1).mux(
              Code._empty,
              array.update(n + 1, array(i)))),
          Code._empty),
        n += 1),
      array.setSize(n))
  }
}
