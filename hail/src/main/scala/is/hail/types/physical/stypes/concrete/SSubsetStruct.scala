package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s.{Code, LongInfo, Settable, TypeInfo, Value}
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, IEmitCode, SortOrder}
import is.hail.types.physical.stypes.{EmitType, SCode, SType}
import is.hail.types.physical.stypes.interfaces.{SBaseStruct, SBaseStructCode, SBaseStructValue, SStructSettable}
import is.hail.types.physical.{PCanonicalStruct, PType, StoredSTypePType}
import is.hail.types.virtual.{TStruct, Type}

case class SSubsetStruct(parent: SBaseStruct, fieldNames: IndexedSeq[String]) extends SBaseStruct {

  val size: Int = fieldNames.size

  val _fieldIdx: Map[String, Int] = fieldNames.zipWithIndex.toMap
  val newToOldFieldMapping: Map[Int, Int] = _fieldIdx
    .map { case (f, i) => (i, parent.virtualType.asInstanceOf[TStruct].fieldIdx(f)) }

  val fieldTypes: IndexedSeq[SType] = Array.tabulate(size)(i => parent.fieldTypes(newToOldFieldMapping(i)))
  val fieldEmitTypes: IndexedSeq[EmitType] = Array.tabulate(size)(i => parent.fieldEmitTypes(newToOldFieldMapping(i)))

  lazy val virtualType: TStruct = {
    val vparent = parent.virtualType.asInstanceOf[TStruct]
    TStruct(fieldNames.map(f => (f, vparent.field(f).typ)): _*)
  }

  override def fieldIdx(fieldName: String): Int = _fieldIdx(fieldName)

  override def castRename(t: Type): SType = {
    val renamedVType = t.asInstanceOf[TStruct]
    val newNames = renamedVType.fieldNames
    val subsetPrevVirtualType = virtualType
    val vparent = parent.virtualType.asInstanceOf[TStruct]
    val newParent = TStruct(vparent.fieldNames.map(f => subsetPrevVirtualType.fieldIdx.get(f) match {
      case Some(idxInSelectedFields) =>
        val renamed = renamedVType.fields(idxInSelectedFields)
        (renamed.name, renamed.typ)
      case None => (f, vparent.fieldType(f))
    }): _*)
    val newType = SSubsetStruct(parent.castRename(newParent).asInstanceOf[SBaseStruct], newNames)
    assert(newType.virtualType == t)
    newType
  }

  def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    if (deepCopy)
      throw new NotImplementedError("Deep copy on subset struct")
    value.st match {
      case SSubsetStruct(parent2, fd2) if parent == parent2 && fieldNames == fd2 && !deepCopy =>
        value
    }
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = parent.codeTupleTypes()

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = parent.settableTupleTypes()

  def fromSettables(settables: IndexedSeq[Settable[_]]): SSubsetStructSettable = {
    new SSubsetStructSettable(this, parent.fromSettables(settables).asInstanceOf[SStructSettable])
  }

  def fromCodes(codes: IndexedSeq[Code[_]]): SSubsetStructCode = {
    new SSubsetStructCode(this, parent.fromCodes(codes))
  }

  override def copiedType: SType = {
    if (virtualType.size < 64)
      SStackStruct(virtualType, fieldEmitTypes.map(_.copiedType))
    else {
      val ct = SBaseStructPointer(PCanonicalStruct(false, virtualType.fieldNames.zip(fieldEmitTypes.map(_.copiedType.storageType)): _*))
      assert(ct.virtualType == virtualType, s"ct=$ct, this=$this")
      ct
    }
  }

  def storageType(): PType = {
    val pt = PCanonicalStruct(false, virtualType.fieldNames.zip(fieldEmitTypes.map(_.copiedType.storageType)): _*)
    assert(pt.virtualType == virtualType, s"pt=$pt, this=$this")
    pt
  }

//  aspirational implementation
//  def storageType(): PType = StoredSTypePType(this, false)

  def containsPointers: Boolean = parent.containsPointers
}

class SSubsetStructSettable(val st: SSubsetStruct, prev: SStructSettable) extends SStructSettable {
  def get: SSubsetStructCode = new SSubsetStructCode(st, prev.load().asBaseStruct)

  def settableTuple(): IndexedSeq[Settable[_]] = prev.settableTuple()

  def loadField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode = {
    prev.loadField(cb, st.newToOldFieldMapping(fieldIdx))
  }

  def isFieldMissing(fieldIdx: Int): Code[Boolean] =
    prev.isFieldMissing(st.newToOldFieldMapping(fieldIdx))

  def store(cb: EmitCodeBuilder, pv: SCode): Unit = prev.store(cb, pv.asInstanceOf[SSubsetStructCode].prev)
}

class SSubsetStructCode(val st: SSubsetStruct, val prev: SBaseStructCode) extends SBaseStructCode {
  def makeCodeTuple(cb: EmitCodeBuilder): IndexedSeq[Code[_]] = prev.makeCodeTuple(cb)

  def memoize(cb: EmitCodeBuilder, name: String): SBaseStructValue = {
    new SSubsetStructSettable(st, prev.memoize(cb, name).asInstanceOf[SStructSettable])
  }

  def memoizeField(cb: EmitCodeBuilder, name: String): SBaseStructValue = {
    new SSubsetStructSettable(st, prev.memoizeField(cb, name).asInstanceOf[SStructSettable])
  }
}
