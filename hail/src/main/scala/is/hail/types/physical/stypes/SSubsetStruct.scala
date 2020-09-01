package is.hail.types.physical.stypes

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{Code, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, IEmitCode, SortOrder}
import is.hail.types.physical.{PBaseStruct, PBaseStructCode, PBaseStructValue, PCode, PStruct, PSubsetStruct, PType}
import is.hail.types.virtual.TStruct
import is.hail.utils.FastIndexedSeq

case class SSubsetStruct(parent: SStruct, fieldNames: IndexedSeq[String]) extends SStruct {
  val fieldIdx: Map[String, Int] = fieldNames.zipWithIndex.toMap
  val newToOldFieldMapping: Map[Int, Int] = fieldIdx
    .map { case (f, i) => (i, parent.pType.virtualType.asInstanceOf[TStruct].fieldIdx(f)) }

  val pType: PSubsetStruct = PSubsetStruct(parent.pType.asInstanceOf[PStruct], fieldNames.toArray
  )

  def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering = pType.codeOrdering(mb)

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: PCode, deepCopy: Boolean): PCode = {
    value.st match {
      case SSubsetStruct(parent2, fd2) if parent == parent2 && fieldNames == fd2 && !deepCopy =>
        value
    }
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = parent.codeTupleTypes()

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): PCode = {
    throw new UnsupportedOperationException
  }
}

class PSubsetStructSettable(val st: SSubsetStruct, prev: SStructSettable) extends SStructSettable {
  def pt: PBaseStruct = st.pType.asInstanceOf[PBaseStruct]

  def get: SSubsetStructCode = new SSubsetStructCode(st, prev.load().asBaseStruct)

  def settableTuple(): IndexedSeq[Settable[_]] = prev.settableTuple()

  def loadField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode = {
    prev.loadField(cb, st.newToOldFieldMapping(fieldIdx))
  }

  def isFieldMissing(fieldIdx: Int): Code[Boolean] =
    prev.isFieldMissing(st.newToOldFieldMapping(fieldIdx))

  def store(cb: EmitCodeBuilder, pv: PCode): Unit = prev.store(cb, pv)
}

class SSubsetStructCode(val st: SSubsetStruct, val prev: PBaseStructCode) extends PBaseStructCode {

  val pt: PBaseStruct = st.pType

  def code: Code[_] = prev.code

  def codeTuple(): IndexedSeq[Code[_]] = prev.codeTuple()

  def memoize(cb: EmitCodeBuilder, name: String): PBaseStructValue = {
    val s = new PSubsetStructSettable(st, prev.memoize(cb, name).asInstanceOf[SStructSettable])
    s.store(cb, prev)
    s
  }

  def memoizeField(cb: EmitCodeBuilder, name: String): PBaseStructValue = {
    val s = new PSubsetStructSettable(st, prev.memoizeField(cb, name).asInstanceOf[SStructSettable])
    s.store(cb, prev)
    s
  }
}
