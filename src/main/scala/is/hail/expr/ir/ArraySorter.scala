package is.hail.expr.ir

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.expr.types._
import is.hail.asm4s._
import is.hail.utils._

object ArraySorter {
  def apply(mb: EmitMethodBuilder, typ: Type, array: Code[Long]): ArraySorter = {

  }


}

class ArraySorter(mb: EmitMethodBuilder, array: StagedArrayBuilder) {
  val typ: Type = array.elt
  val ti: TypeInfo[_] = typeToTypeInfo(typ)
  val ord: CodeOrdering = CodeOrdering(typ, missingGreatest = true)

  def sort(): Code[Unit] = {

    val sort = mb.fb.newMethod[Region, Int, Int, Int]
    val region = sort.getArg[Region](1)
    val start = sort.getArg[Int](2)
    val end = sort.getArg[Int](3)

    val pi: LocalRef[Int] = sort.newLocal[Int]
    val i: LocalRef[Int] = sort.newLocal[Int]

    val m1: LocalRef[Boolean] = sort.newLocal[Boolean]
    val v1: LocalRef[_] = sort.newLocal(ti)

    def loadPivot(start: Code[Int], end: Code[Int]): Code[Unit] = {
      pi := end
    }

    def lt(m1: Code[Boolean], v1: Code[_], m2: Code[Boolean], v2: Code[_]): Code[Boolean] = {
      m1.mux(false, m2.mux(true, ord.compare(mb, v1, v2) < 0))
    }

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
      (start < pi - 2).mux(sort.invoke(region, start, pi - 1), Code._empty),
      (pi + 2 < end).mux(sort.invoke(region, pi + 1, end), Code._empty)))

    sort.invoke(mb.getArg[Region](1), 0, array.size - 1)
  }
}
