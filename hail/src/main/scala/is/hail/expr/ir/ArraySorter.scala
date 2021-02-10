package is.hail.expr.ir

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.types.physical.{PCanonicalArray, PCode, PType, typeToTypeInfo}

class ArraySorter(r: EmitRegion, array: StagedArrayBuilder) {
  val typ: PType = array.elt
  val ti: TypeInfo[_] = typeToTypeInfo(typ)
  val mb: EmitMethodBuilder[_] = r.mb

  def sort(sorter: DependentEmitFunctionBuilder[_]): Code[Unit] = {
    val localF = ti match {
      case BooleanInfo => mb.genFieldThisRef[AsmFunction2[Boolean, Boolean, Boolean]]()
      case IntInfo => mb.genFieldThisRef[AsmFunction2[Int, Int, Boolean]]()
      case LongInfo => mb.genFieldThisRef[AsmFunction2[Int, Int, Boolean]]()
      case FloatInfo => mb.genFieldThisRef[AsmFunction2[Long, Long, Boolean]]()
      case DoubleInfo => mb.genFieldThisRef[AsmFunction2[Double, Double, Boolean]]()
    }
    Code(localF.storeAny(Code.checkcast(sorter.newInstance(mb))(localF.ti)), array.sort(localF))
  }

  def toRegion(cb: EmitCodeBuilder, t: PType): PCode = {
    t match {
      case pca: PCanonicalArray =>
        val len = cb.newLocal[Int]("arraysorter_to_region_len", array.size)
        pca.constructFromElements(cb, r.region, len, deepCopy = false) { (cb, idx) =>
          IEmitCode(cb, array.isMissing(idx), PCode(typ, array(idx)))
        }
    }
  }

  def pruneMissing: Code[Unit] = {
    val i = mb.newLocal[Int]()
    val n = mb.newLocal[Int]()

    Code(
      n := 0,
      i := 0,
      Code.whileLoop(i < array.size,
        Code(
          array.isMissing(i).mux(
            Code._empty,
            i.ceq(n).mux(
              n += 1,
              Code(array.setMissing(n, false), array.update(n, array(i)), n += 1))),
          i += 1)),
      array.setSize(n))
  }

  def distinctFromSorted(discardNext: (Code[Region], Code[_], Code[Boolean], Code[_], Code[Boolean]) => Code[Boolean]): Code[Unit] = {
    val i = mb.newLocal[Int]()
    val n = mb.newLocal[Int]()

    Code(
      i := 0,
      n := 0,
      Code.whileLoop(i < array.size,
        i += 1,
        Code.whileLoop(i < array.size && discardNext(r.region, array(n), array.isMissing(n), array(i), array.isMissing(i)),
          i += 1),
        n += 1,
        (i < array.size && i.cne(n)).mux(
          Code(
            array.setMissing(n, array.isMissing(i)),
            array.isMissing(n).mux(
              Code._empty,
              array.update(n, array(i)))),
          Code._empty)),
      array.setSize(n))
  }
}
