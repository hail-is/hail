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

          mb.getCodeOrdering[Boolean](kt, CodeOrdering.equiv, missingGreatest = true)(r1, (mk1, k1), r2, (mk2, k2))
      }
    }

    ceq
  } else
      mb.getCodeOrdering[Boolean](typ, CodeOrdering.equiv, missingGreatest = true)

  def sort(): Code[Unit] = {

    def newSortFunction[T: TypeInfo](
      fb: EmitFunctionBuilder[_]
    ): DependentEmitFunction[AsmFunction4[T, Boolean, T, Boolean, Boolean]] =
      fb.newDependentFunction[T, Boolean, T, Boolean, Boolean]

    val sorter = if (keyOnly) {
      val ttype = coerce[TBaseStruct](typ)
      require(ttype.size == 2)
      val kt = ttype.types(0)
      val kti = typeToTypeInfo(kt)

      val sorter = newSortFunction[Long](mb.fb)

      val mk1 = sorter.newLocal[Boolean]
      val mk2 = sorter.newLocal[Boolean]
      val sorterRegion: Code[Region] = sorter.addField[Region](mb.getArg[Region](1))

      val m1 = sorter.getArg[Boolean](2)
      val m2 = sorter.getArg[Boolean](4)
      val v1 = sorter.getArg[Long](1)
      val v2 = sorter.getArg[Long](3)

      val k1 = mk1.mux(defaultValue(kt), sorterRegion.loadIRIntermediate(kt)(ttype.fieldOffset(v1, 0)))
      val k2 = mk2.mux(defaultValue(kt), sorterRegion.loadIRIntermediate(kt)(ttype.fieldOffset(v2, 0)))
      val cmp = sorter.getCodeOrdering[Int](
        kt, CodeOrdering.compare, missingGreatest = true)(
        sorterRegion, (mk1, k1), sorterRegion, (mk2, k2))
      val comparison = Code(
        mk1 := ttype.isFieldMissing(sorterRegion, v1, 0),
        mk2 := ttype.isFieldMissing(sorterRegion, v2, 0),
        (m1 || m2).mux((m1 && m2).mux(0, m1.mux(1, -1)), cmp)
      )
      sorter.emit(comparison < 0)
      sorter
    } else {
      val sorter = ti match {
        case BooleanInfo => newSortFunction[Boolean](mb.fb)
        case IntInfo => newSortFunction[Int](mb.fb)
        case LongInfo => newSortFunction[Long](mb.fb)
        case FloatInfo => newSortFunction[Float](mb.fb)
        case DoubleInfo => newSortFunction[Double](mb.fb)
      }

      val sorterRegion: Code[Region] = sorter.addField[Region](mb.getArg[Region](1))
      val cmp = sorter.getCodeOrdering[Int](typ, CodeOrdering.compare, missingGreatest = true)
      sorter.emit(cmp(
        sorterRegion,
        (sorter.getArg[Boolean](2), sorter.getArg(1)(ti)),
        sorterRegion,
        (sorter.getArg[Boolean](2), sorter.getArg(1)(ti))) < 0)
      sorter
    }
    val sorterField = mb.newField()(sorter.interfaceTi)
    Code(
      sorter.newInstance(sorterField),
      array.sort(sorterField.load()))
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
