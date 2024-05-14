package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s.{Settable, TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.{PCanonicalStruct, PType}
import is.hail.types.physical.stypes.{EmitType, SType, SValue}
import is.hail.types.physical.stypes.interfaces.{SBaseStruct, SBaseStructSettable, SBaseStructValue}
import is.hail.types.virtual.{TStruct, Type}

final case class SSubsetStruct(parent: SBaseStruct, fieldNames: IndexedSeq[String])
    extends SBaseStruct {

  override val size: Int = fieldNames.size

  val _fieldIdx: Map[String, Int] = fieldNames.zipWithIndex.toMap

  val newToOldFieldMapping: Map[Int, Int] = _fieldIdx
    .map { case (f, i) => (i, parent.virtualType.asInstanceOf[TStruct].fieldIdx(f)) }

  override val fieldTypes: IndexedSeq[SType] =
    Array.tabulate(size)(i => parent.fieldTypes(newToOldFieldMapping(i)))

  override val fieldEmitTypes: IndexedSeq[EmitType] =
    Array.tabulate(size)(i => parent.fieldEmitTypes(newToOldFieldMapping(i)))

  override lazy val virtualType: TStruct = {
    val vparent = parent.virtualType.asInstanceOf[TStruct]
    TStruct(fieldNames.map(f => (f, vparent.field(f).typ)): _*)
  }

  override def fieldIdx(fieldName: String): Int = _fieldIdx(fieldName)

  override def castRename(t: Type): SType = {
    val renamedVType = t.asInstanceOf[TStruct]
    val newNames = renamedVType.fieldNames
    val subsetPrevVirtualType = virtualType
    val vparent = parent.virtualType.asInstanceOf[TStruct]
    val newParent = TStruct(vparent.fieldNames.map(f =>
      subsetPrevVirtualType.fieldIdx.get(f) match {
        case Some(idxInSelectedFields) =>
          val renamed = renamedVType.fields(idxInSelectedFields)
          (renamed.name, renamed.typ)
        case None => (f, vparent.fieldType(f))
      }
    ): _*)
    val newType = SSubsetStruct(parent.castRename(newParent).asInstanceOf[SBaseStruct], newNames)
    assert(newType.virtualType == t)
    newType
  }

  override def _coerceOrCopy(
    cb: EmitCodeBuilder,
    region: Value[Region],
    value: SValue,
    deepCopy: Boolean,
  ): SValue = {
    if (deepCopy)
      throw new NotImplementedError("Deep copy on subset struct")
    value.st match {
      case SSubsetStruct(parent2, fd2) if parent == parent2 && fieldNames == fd2 && !deepCopy =>
        value
    }
  }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = parent.settableTupleTypes()

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SSubsetStructSettable =
    new SSubsetStructSettable(
      this,
      parent.fromSettables(settables).asInstanceOf[SBaseStructSettable],
    )

  override def fromValues(values: IndexedSeq[Value[_]]): SSubsetStructValue =
    new SSubsetStructValue(this, parent.fromValues(values).asInstanceOf[SBaseStructValue])

  override def copiedType: SType = {
    if (virtualType.size < 64)
      SStackStruct(virtualType, fieldEmitTypes.map(_.copiedType))
    else {
      val ct = SBaseStructPointer(PCanonicalStruct(
        false,
        virtualType.fieldNames.zip(fieldEmitTypes.map(_.copiedType.storageType)): _*
      ))
      assert(ct.virtualType == virtualType, s"ct=$ct, this=$this")
      ct
    }
  }

  def storageType(): PType = {
    val pt = PCanonicalStruct(
      false,
      virtualType.fieldNames.zip(fieldEmitTypes.map(_.copiedType.storageType)): _*
    )
    assert(pt.virtualType == virtualType, s"pt=$pt, this=$this")
    pt
  }

//  aspirational implementation
//  def storageType(): PType = StoredSTypePType(this, false)

  override def containsPointers: Boolean = parent.containsPointers
}

class SSubsetStructValue(val st: SSubsetStruct, val prev: SBaseStructValue)
    extends SBaseStructValue {
  override lazy val valueTuple: IndexedSeq[Value[_]] = prev.valueTuple

  override def loadField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode =
    prev.loadField(cb, st.newToOldFieldMapping(fieldIdx))

  override def isFieldMissing(cb: EmitCodeBuilder, fieldIdx: Int): Value[Boolean] =
    prev.isFieldMissing(cb, st.newToOldFieldMapping(fieldIdx))
}

final class SSubsetStructSettable(st: SSubsetStruct, prev: SBaseStructSettable)
    extends SSubsetStructValue(st, prev) with SBaseStructSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = prev.settableTuple()

  override def store(cb: EmitCodeBuilder, pv: SValue): Unit =
    prev.store(cb, pv.asInstanceOf[SSubsetStructValue].prev)
}
