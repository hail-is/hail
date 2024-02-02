package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s.{Settable, TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.stypes.interfaces.{SBaseStruct, SBaseStructSettable, SBaseStructValue}
import is.hail.types.physical.stypes.{EmitType, SType, SValue}
import is.hail.types.physical.{PCanonicalStruct, PType}
import is.hail.types.virtual.{Field, TStruct, Type}

class SSubsetStruct(
  private val parent: SBaseStruct,
  private val fieldNames: IndexedSeq[String],
) extends SBaseStruct {

  override val size: Int = fieldNames.size

  val _fieldIdx: Map[String, Int] = fieldNames.zipWithIndex.toMap

  lazy val newToOldFieldMapping: Map[Int, Int] = {
    val parentFieldIdx = parent.virtualType.asInstanceOf[TStruct]
    _fieldIdx.map { case (f, i) => i -> parentFieldIdx.fieldIdx(f) }
  }

  override lazy val fieldTypes: IndexedSeq[SType] =
    Array.tabulate(size)(i => parent.fieldTypes(newToOldFieldMapping(i)))

  override lazy val fieldEmitTypes: IndexedSeq[EmitType] =
    Array.tabulate(size)(i => parent.fieldEmitTypes(newToOldFieldMapping(i)))

  override lazy val virtualType: TStruct = {
    val vparentTypes = parent.virtualType.asInstanceOf[TStruct].types
    TStruct(fieldNames.zipWithIndex.map { case (f, i) =>
      Field(f, vparentTypes(newToOldFieldMapping(i)), i)
    })
  }

  override def fieldIdx(fieldName: String): Int =
    _fieldIdx(fieldName)

  override def castRename(t: Type): SType = {
    val newVirtualType = t.asInstanceOf[TStruct]
    val oldToNewFieldMapping = newToOldFieldMapping.map(n => n._2 -> n._1)

    // note we may have subsetted a parent struct{x,y,z} to struct{z} then renamed to struct{x}
    // must only tell parent to castRename what it knows as `z`, leaving others intact
    val newParent = parent.castRename(
      TStruct(parent.virtualType.fields.zipWithIndex.map { case (f, i) =>
        oldToNewFieldMapping.get(i) match {
          case Some(idx) => f.copy(typ = newVirtualType.types(idx))
          case None => f
        }
      })
    )

    new SSubsetStruct(newParent.asInstanceOf[SBaseStruct], newVirtualType.fieldNames) {
      override lazy val newToOldFieldMapping: Map[Int, Int] =
        SSubsetStruct.this.newToOldFieldMapping
      override lazy val virtualType: TStruct =
        newVirtualType
    }
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
      case ss: SSubsetStruct if parent == ss.parent && fieldNames == ss.fieldNames && !deepCopy =>
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

  override def equals(obj: Any): Boolean =
    obj match {
      case s: SSubsetStruct =>
        newToOldFieldMapping == s.newToOldFieldMapping &&
        parent.fieldTypes == s.parent.fieldTypes
      case _ =>
        false
    }
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
