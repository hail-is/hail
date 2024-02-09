package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s.{Settable, TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.{PCanonicalStruct, PType}
import is.hail.types.physical.stypes.{EmitType, SType, SValue}
import is.hail.types.physical.stypes.interfaces.{SBaseStruct, SBaseStructSettable, SBaseStructValue}
import is.hail.types.virtual.{TBaseStruct, TStruct, Type}

object SStructView {
  def subset(fieldnames: IndexedSeq[String], struct: SBaseStruct): SStructView =
    struct match {
      case s: SStructView =>
        val pfields = s.parent.virtualType.fields
        new SStructView(
          s.parent,
          fieldnames.map(f => pfields(s.fieldIdx(f)).name),
          s.rename.typeAfterSelectNames(fieldnames),
        )

      case s =>
        val restrict = s.virtualType.asInstanceOf[TStruct].typeAfterSelectNames(fieldnames)
        new SStructView(s, fieldnames, restrict)
    }
}

// A 'view' on `SBaseStruct`s, ie one that presents an upcast and/or renamed facade on another
final class SStructView(
  private val parent: SBaseStruct,
  private val restrict: IndexedSeq[String],
  private val rename: TStruct,
) extends SBaseStruct {

  assert(
    parent.virtualType.asInstanceOf[TStruct].typeAfterSelectNames(restrict) canCastTo rename,
    s"""Renamed type is not isomorphic to subsetted type
       |   parent: '${parent.virtualType._toPretty}'
       | restrict: '${restrict.mkString("[", ",", "]")}'
       |   rename: '${rename._toPretty}'
       |""".stripMargin,
  )

  override def size: Int =
    restrict.length

  lazy val newToOldFieldMapping: Map[Int, Int] =
    restrict.view.zipWithIndex.map { case (f, i) => i -> parent.fieldIdx(f) }.toMap

  override lazy val fieldTypes: IndexedSeq[SType] =
    Array.tabulate(size) { i =>
      parent
        .fieldTypes(newToOldFieldMapping(i))
        .castRename(rename.fields(i).typ)
    }

  override lazy val fieldEmitTypes: IndexedSeq[EmitType] =
    Array.tabulate(size) { i =>
      parent
        .fieldEmitTypes(newToOldFieldMapping(i))
        .copy(st = fieldTypes(i))
    }

  override def virtualType: TBaseStruct =
    rename

  override def fieldIdx(fieldName: String): Int =
    rename.fieldIdx(fieldName)

  override def castRename(t: Type): SType =
    new SStructView(parent, restrict, rename = t.asInstanceOf[TStruct])

  override def _coerceOrCopy(
    cb: EmitCodeBuilder,
    region: Value[Region],
    value: SValue,
    deepCopy: Boolean,
  ): SValue = {
    if (deepCopy)
      throw new NotImplementedError("Deep copy on struct view")

    value.st match {
      case s: SStructView if this == s && !deepCopy =>
        value
    }
  }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] =
    parent.settableTupleTypes()

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SStructViewSettable =
    new SStructViewSettable(
      this,
      parent.fromSettables(settables).asInstanceOf[SBaseStructSettable],
    )

  override def fromValues(values: IndexedSeq[Value[_]]): SStructViewValue =
    new SStructViewValue(this, parent.fromValues(values).asInstanceOf[SBaseStructValue])

  override def copiedType: SType =
    if (virtualType.size < 64)
      SStackStruct(virtualType, fieldEmitTypes.map(_.copiedType))
    else {
      val ct = SBaseStructPointer(storageType().asInstanceOf[PCanonicalStruct])
      assert(ct.virtualType == virtualType, s"ct=$ct, this=$this")
      ct
    }

  def storageType(): PType = {
    val pt = PCanonicalStruct(
      required = false,
      args = rename.fieldNames.zip(fieldEmitTypes.map(_.copiedType.storageType)): _*,
    )
    assert(pt.virtualType == virtualType, s"pt=$pt, this=$this")
    pt
  }

  //  aspirational implementation
  //  def storageType(): PType = StoredSTypePType(this, false)

  override def containsPointers: Boolean =
    parent.containsPointers

  override def equals(obj: Any): Boolean =
    obj match {
      case s: SStructView =>
        rename == s.rename &&
        newToOldFieldMapping == s.newToOldFieldMapping &&
        parent == s.parent // todo test isIsomorphicTo
      case _ =>
        false
    }
}

class SStructViewValue(val st: SStructView, val prev: SBaseStructValue) extends SBaseStructValue {

  override lazy val valueTuple: IndexedSeq[Value[_]] =
    prev.valueTuple

  override def subset(fieldNames: String*): SBaseStructValue =
    new SStructViewValue(SStructView.subset(fieldNames.toIndexedSeq, st), prev)

  override def loadField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode =
    prev
      .loadField(cb, st.newToOldFieldMapping(fieldIdx))
      .map(cb)(_.castRename(st.virtualType.fields(fieldIdx).typ))

  override def isFieldMissing(cb: EmitCodeBuilder, fieldIdx: Int): Value[Boolean] =
    prev.isFieldMissing(cb, st.newToOldFieldMapping(fieldIdx))
}

final class SStructViewSettable(st: SStructView, prev: SBaseStructSettable)
    extends SStructViewValue(st, prev) with SBaseStructSettable {
  override def subset(fieldNames: String*): SBaseStructValue =
    new SStructViewSettable(SStructView.subset(fieldNames.toIndexedSeq, st), prev)

  override def settableTuple(): IndexedSeq[Settable[_]] =
    prev.settableTuple()

  override def store(cb: EmitCodeBuilder, pv: SValue): Unit =
    prev.store(cb, pv.asInstanceOf[SStructViewValue].prev)
}
