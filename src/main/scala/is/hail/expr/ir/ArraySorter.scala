package is.hail.expr.ir

import is.hail.annotations.{CodeOrdering, Region, StagedRegionValueBuilder}
import is.hail.expr.types._
import is.hail.asm4s._
import is.hail.utils._

class ArraySorter(mb: EmitMethodBuilder, array: StagedArrayBuilder, keyOnly: Boolean) {
  val typ: Type = array.elt
  val ti: TypeInfo[_] = typeToTypeInfo(typ)
  val sortmb: EmitMethodBuilder = mb.fb.newMethod[Region, Int, Int, Boolean, Unit]

  val equiv: CodeOrdering.F[Boolean] = if (keyOnly) {
    val ttype = coerce[TBaseStruct](typ)
    require(ttype.size == 2)
    val kt = ttype.types(0)
    val ceq: CodeOrdering.F[Boolean] = {
      val mk1l = mb.newLocal[Boolean]
      val mk2l = mb.newLocal[Boolean]

      { case (r1: Code[Region], (m1: Code[Boolean]@unchecked, v1: Code[Long]@unchecked),
              r2: Code[Region]@unchecked, (m2: Code[Boolean]@unchecked, v2: Code[Long]@unchecked)) =>
          val mk1 = Code(mk1l := m1 || ttype.isFieldMissing(r1, v1, 0), mk1l)
          val mk2 = Code(mk2l := m2 || ttype.isFieldMissing(r2, v2, 0), mk2l)
          val k1 = mk1l.mux(defaultValue(kt), r1.loadIRIntermediate(kt)(ttype.fieldOffset(v1, 0)))
          val k2 = mk2l.mux(defaultValue(kt), r2.loadIRIntermediate(kt)(ttype.fieldOffset(v2, 0)))

          mb.getCodeOrdering[Boolean](-kt, CodeOrdering.equiv, missingGreatest = true)(r1, (mk1, k1), r2, (mk2, k2))
      }
    }
    ceq
  } else
      mb.getCodeOrdering[Boolean](typ, CodeOrdering.equiv, missingGreatest = true)

  def sort(ascending: Code[Boolean]): Code[Unit] = {

    val (sorter, localF) = ti match {
      case BooleanInfo => (mb.fb.newDependentCompareFunction[Boolean](), mb.newField[BooleanOrderingFunction])
      case IntInfo => (mb.fb.newDependentCompareFunction[Int](), mb.newField[IntOrderingFunction])
      case LongInfo => (mb.fb.newDependentCompareFunction[Long](), mb.newField[LongOrderingFunction])
      case FloatInfo => (mb.fb.newDependentCompareFunction[Float](), mb.newField[FloatOrderingFunction])
      case DoubleInfo => (mb.fb.newDependentCompareFunction[Double](), mb.newField[DoubleOrderingFunction])
    }
    val sorterRegion: Code[Region] = sorter.addField[Region](mb.getArg[Region](1))
    val asc: Code[Boolean] = sorter.addField[Boolean](ascending)
    if (keyOnly) {
      val ttype = coerce[TBaseStruct](typ)
      require(ttype.size == 2)
      val kt = ttype.types(0)

      val mk1 = sorter.newLocal[Boolean]
      val mk2 = sorter.newLocal[Boolean]
      val v1 = sorter.getArg[Long](1)
      val v2 = sorter.getArg[Long](2)

      val k1 = mk1.mux(defaultValue(kt), sorterRegion.loadIRIntermediate(kt)(ttype.fieldOffset(v1, 0)))
      val k2 = mk2.mux(defaultValue(kt), sorterRegion.loadIRIntermediate(kt)(ttype.fieldOffset(v2, 0)))
      val cmp = sorter.getCodeOrdering[Int](kt, CodeOrdering.compare, missingGreatest = true)
      sorter.emit(Code(
        mk1 := ttype.isFieldMissing(sorterRegion, v1, 0),
        mk2 := ttype.isFieldMissing(sorterRegion, v2, 0),
        cmp(sorterRegion, (mk1, k1), sorterRegion, (mk2, k2)) < 0
      ))
    } else {
      val cmp = sorter.getCodeOrdering[Int](+typ, CodeOrdering.compare, missingGreatest = true)(
        sorterRegion, (false, sorter.getArg(1)(ti)),
        sorterRegion, (false, sorter.getArg(2)(ti)))
      sorter.emit(asc.mux(cmp < 0, cmp > 0))
    }

    Code(
      localF.storeAny(sorter.newInstance()),
      array.sort(localF))
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

  def distinctFromSorted(): Code[Unit] = {
    def ceq(m1: Code[Boolean], v1: Code[_], m2: Code[Boolean], v2: Code[_]): Code[Boolean] = {
      equiv(mb.getArg[Region](1), (m1, v1), mb.getArg[Region](1), (m2, v2))
    }

    val i = mb.newLocal[Int]
    val n = mb.newLocal[Int]

    val removeMissing = Code(i := array.size - 1,
      Code.whileLoop(i >= 0 && array.isMissing(i), i += -1),
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

  def sortIntoRegion(ascending: Code[Boolean], distinct: Boolean): Code[Long] = {
    Code(sort(ascending), if (distinct) distinctFromSorted() else Code._empty, toRegion())
  }
}
