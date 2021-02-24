package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s.{Code, LongInfo, Settable, TypeInfo, Value}
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, IEmitCode, IEmitSCode, SortOrder}
import is.hail.types.physical.stypes.{SCode, SType}
import is.hail.types.physical.stypes.interfaces.{SBaseStruct, SStructSettable}
import is.hail.types.physical.{PBaseStruct, PBaseStructCode, PBaseStructValue, PCode, PStruct, PStructSettable, PSubsetStruct, PType}
import is.hail.types.virtual.TStruct

case class SSubsetStruct(parent: SBaseStruct, fieldNames: IndexedSeq[String]) extends SBaseStruct {

  val size: Int = fieldNames.size

  val fieldIdx: Map[String, Int] = fieldNames.zipWithIndex.toMap
  val newToOldFieldMapping: Map[Int, Int] = fieldIdx
    .map { case (f, i) => (i, parent.pType.virtualType.asInstanceOf[TStruct].fieldIdx(f)) }

  val fieldTypes: Array[SType] = Array.tabulate(size)(i => parent.fieldTypes(newToOldFieldMapping(i)))

  val pType: PSubsetStruct = PSubsetStruct(parent.pType.asInstanceOf[PStruct], fieldNames.toArray
  )

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    value.st match {
      case SSubsetStruct(parent2, fd2) if parent == parent2 && fieldNames == fd2 && !deepCopy =>
        value
    }
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = parent.codeTupleTypes()

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): SCode = {
    throw new UnsupportedOperationException
  }

  def fromSettables(settables: IndexedSeq[Settable[_]]): SSubsetStructSettable = {
    new SSubsetStructSettable(this, parent.fromSettables(settables).asInstanceOf[PStructSettable])
  }

  def fromCodes(codes: IndexedSeq[Code[_]]): SSubsetStructCode = {
    new SSubsetStructCode(this, parent.fromCodes(codes).asInstanceOf[PBaseStructCode])
  }

  def canonicalPType(): PType = pType
}

// FIXME: prev should be SStructSettable, not PStructSettable
class SSubsetStructSettable(val st: SSubsetStruct, prev: PStructSettable) extends PStructSettable {
  def pt: PBaseStruct = st.pType.asInstanceOf[PBaseStruct]

  def get: SSubsetStructCode = new SSubsetStructCode(st, prev.load().asBaseStruct)

  def settableTuple(): IndexedSeq[Settable[_]] = prev.settableTuple()

  def loadField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitSCode = {
    prev.loadField(cb, st.newToOldFieldMapping(fieldIdx))
  }

  def isFieldMissing(fieldIdx: Int): Code[Boolean] =
    prev.isFieldMissing(st.newToOldFieldMapping(fieldIdx))

  def store(cb: EmitCodeBuilder, pv: PCode): Unit = prev.store(cb, pv.asInstanceOf[SSubsetStructCode].prev)
}

class SSubsetStructCode(val st: SSubsetStruct, val prev: PBaseStructCode) extends PBaseStructCode {

  val pt: PBaseStruct = st.pType

  def code: Code[_] = prev.code

  def codeTuple(): IndexedSeq[Code[_]] = prev.codeTuple()

  def memoize(cb: EmitCodeBuilder, name: String): PBaseStructValue = {
    new SSubsetStructSettable(st, prev.memoize(cb, name).asInstanceOf[PStructSettable])
  }

  def memoizeField(cb: EmitCodeBuilder, name: String): PBaseStructValue = {
    new SSubsetStructSettable(st, prev.memoizeField(cb, name).asInstanceOf[PStructSettable])
  }
}
