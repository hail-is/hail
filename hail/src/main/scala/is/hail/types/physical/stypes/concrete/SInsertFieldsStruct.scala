package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s.{Code, Settable, TypeInfo, Value}
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, EmitSettable, EmitValue, IEmitCode}
import is.hail.types.physical.stypes.interfaces.{SBaseStruct, SBaseStructCode, SBaseStructSettable, SBaseStructValue}
import is.hail.types.physical.stypes.{EmitType, SCode, SType, SValue}
import is.hail.types.physical.{PCanonicalStruct, PType}
import is.hail.types.virtual.{TStruct, Type}
import is.hail.utils._

object SInsertFieldsStruct {
  def merge(cb: EmitCodeBuilder, s1: SBaseStructValue, s2: SBaseStructValue): SInsertFieldsStructValue = {
    val lt = s1.st.virtualType.asInstanceOf[TStruct]
    val rt = s2.st.virtualType.asInstanceOf[TStruct]
    val resultVType = TStruct.concat(lt, rt)

    val st1 = s1.st
    val st2 = s2.st
    val st = SInsertFieldsStruct(resultVType, st1, rt.fieldNames.zip(st2.fieldEmitTypes))

    if (st2.size == 1) {
      new SInsertFieldsStructValue(st, s1, FastIndexedSeq(cb.memoize(s2.loadField(cb, 0), "InsertFieldsStruct_merge")))
    } else {
      new SInsertFieldsStructValue(st, s1, (0 until st2.size).map(i => cb.memoize(s2.loadField(cb, i), "InsertFieldsStruct_merge")))
    }
  }
}

final case class SInsertFieldsStruct(virtualType: TStruct, parent: SBaseStruct, insertedFields: IndexedSeq[(String, EmitType)]) extends SBaseStruct {
  override def size: Int = virtualType.size

  // Maps index in result struct to index in insertedFields.
  // Indices that refer to parent fields are not present.
  lazy val insertedFieldIndices: Map[Int, Int] = insertedFields.zipWithIndex
    .map { case ((name, _), idx) => virtualType.fieldIdx(name) -> idx }
    .toMap

  def getFieldIndexInNewOrParent(idx: Int): Either[Int, Int] = {
    insertedFieldIndices.get(idx) match {
      case Some(idx) => Right(idx)
      case None => Left(parent.fieldIdx(virtualType.fieldNames(idx)))
    }
  }

  override val fieldEmitTypes: IndexedSeq[EmitType] = virtualType.fieldNames.zipWithIndex.map { case (f, idx) =>
    insertedFieldIndices.get(idx) match {
      case Some(idx) => insertedFields(idx)._2
      case None => parent.fieldEmitTypes(parent.fieldIdx(f))
    }
  }
  private lazy val insertedFieldSettableStarts = insertedFields.map(_._2.nSettables).scanLeft(0)(_ + _).init

  override lazy val fieldTypes: IndexedSeq[SType] = fieldEmitTypes.map(_.st)

  override def fieldIdx(fieldName: String): Int = virtualType.fieldIdx(fieldName)

  override def copiedType: SType = {
    if (virtualType.size < 64)
      SStackStruct(virtualType, fieldEmitTypes.map(_.copiedType))
    else {
      val ct = SBaseStructPointer(PCanonicalStruct(false, virtualType.fieldNames.zip(fieldEmitTypes.map(_.copiedType.storageType)): _*))
      assert(ct.virtualType == virtualType, s"ct=$ct, this=$this")
      ct
    }
  }

  override def storageType(): PType = {
    val pt = PCanonicalStruct(false, virtualType.fieldNames.zip(fieldEmitTypes.map(_.copiedType.storageType)): _*)
    assert(pt.virtualType == virtualType, s"cp=$pt, this=$this")
    pt
  }

//  aspirational implementation
//  def storageType(): PType = StoredSTypePType(this, false)

  override def containsPointers: Boolean = parent.containsPointers || insertedFields.exists(_._2.st.containsPointers)

  override lazy val settableTupleTypes: IndexedSeq[TypeInfo[_]] = parent.settableTupleTypes() ++ insertedFields.flatMap(_._2.settableTupleTypes)

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SInsertFieldsStructSettable = {
    assert(settables.map(_.ti) == settableTupleTypes)
    new SInsertFieldsStructSettable(this, parent.fromSettables(settables.take(parent.nSettables)).asInstanceOf[SBaseStructSettable], insertedFields.indices.map { i =>
      val et = insertedFields(i)._2
      val start = insertedFieldSettableStarts(i) + parent.nSettables
      et.fromSettables(settables.slice(start, start + et.nSettables))
    })
  }

  override def fromValues(values: IndexedSeq[Value[_]]): SInsertFieldsStructValue = {
    assert(values.map(_.ti) == settableTupleTypes)
    new SInsertFieldsStructValue(this, parent.fromValues(values.take(parent.nSettables)).asInstanceOf[SBaseStructValue], insertedFields.indices.map { i =>
      val et = insertedFields(i)._2
      val start = insertedFieldSettableStarts(i) + parent.nSettables
      et.fromValues(values.slice(start, start + et.nSettables))
    })
  }

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SValue = {
    value match {
      case ss: SInsertFieldsStructValue if ss.st == this => value
      case _ => throw new RuntimeException(s"copy insertfields struct")
    }
  }

  override def castRename(t: Type): SType = {
    val ts = t.asInstanceOf[TStruct]

    val parentType = parent.virtualType.asInstanceOf[TStruct]

    val renamedInsertedFields = Array.fill[(String, EmitType)](insertedFields.size)(null)
    val parentPassThroughFieldBuilder = new BoxedArrayBuilder[(String, (String, Type))]()

    (0 until ts.size).foreach { i =>
      val newField = ts.fields(i)
      val newName = newField.name
      val oldName = virtualType.fieldNames(i)
      insertedFieldIndices.get(i) match {
        case Some(idx) =>
          val et = insertedFields(idx)._2
          renamedInsertedFields(idx) = ((newName, et.copy(st = et.st.castRename(newField.typ))))
        case None => parentPassThroughFieldBuilder += ((oldName, (newName, newField.typ)))
      }
    }

    val parentPassThroughMap = parentPassThroughFieldBuilder.result().toMap
    val parentCastType = TStruct(parentType.fieldNames.map(f => parentPassThroughMap.getOrElse(f, (f, parentType.fieldType(f)))): _*)
    val renamedParentType = parent.castRename(parentCastType)
    SInsertFieldsStruct(ts,
      renamedParentType.asInstanceOf[SBaseStruct],
      renamedInsertedFields
    )
  }
}

class SInsertFieldsStructValue(
  val st: SInsertFieldsStruct,
  parent: SBaseStructValue,
  newFields: IndexedSeq[EmitValue]
) extends SBaseStructValue {
  override def get: SInsertFieldsStructCode = new SInsertFieldsStructCode(st, parent.get, newFields.map(_.load))

  override lazy val valueTuple: IndexedSeq[Value[_]] = parent.valueTuple ++ newFields.flatMap(_.valueTuple())

  override def loadField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode = {
    st.getFieldIndexInNewOrParent(fieldIdx) match {
      case Left(parentIdx) => parent.loadField(cb, parentIdx)
      case Right(newFieldsIdx) => newFields(newFieldsIdx).toI(cb)
    }
  }

  override def isFieldMissing(fieldIdx: Int): Code[Boolean] =
    st.getFieldIndexInNewOrParent(fieldIdx) match {
      case Left(parentIdx) => parent.isFieldMissing(parentIdx)
      case Right(newFieldsIdx) => newFields(newFieldsIdx).m
    }

  override def _insert(newType: TStruct, fields: (String, EmitValue)*): SBaseStructValue = {
    val newFieldSet = fields.map(_._1).toSet
    val filteredNewFields = st.insertedFields.map(_._1)
      .zipWithIndex
      .filter { case (name, idx) => !newFieldSet.contains(name) }
      .map { case (name, idx) => (name, newFields(idx)) }
    parent._insert(newType, filteredNewFields ++ fields: _*)
  }
}

final class SInsertFieldsStructSettable(
  st: SInsertFieldsStruct,
  parent: SBaseStructSettable,
  newFields: IndexedSeq[EmitSettable]
) extends SInsertFieldsStructValue(st, parent, newFields) with SBaseStructSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = parent.settableTuple() ++ newFields.flatMap(_.settableTuple())

  override def store(cb: EmitCodeBuilder, pv: SCode): Unit = {
    val sifc = pv.asInstanceOf[SInsertFieldsStructCode]
    parent.store(cb, sifc.parent)
    newFields.zip(sifc.newFields).foreach { case (settable, code) => cb.assign(settable, code) }
  }
}

class SInsertFieldsStructCode(val st: SInsertFieldsStruct, val parent: SBaseStructCode, val newFields: IndexedSeq[EmitCode]) extends SBaseStructCode {
  override def memoize(cb: EmitCodeBuilder, name: String): SInsertFieldsStructSettable = {
    new SInsertFieldsStructSettable(st, parent.memoize(cb, name + "_parent").asInstanceOf[SBaseStructSettable], newFields.indices.map { i =>
      val code = newFields(i)
      val es = cb.emb.newEmitLocal(s"${ name }_nf_$i", code.emitType)
      es.store(cb, code)
      es
    })
  }

  override def memoizeField(cb: EmitCodeBuilder, name: String): SInsertFieldsStructSettable = {
    new SInsertFieldsStructSettable(st, parent.memoizeField(cb, name + "_parent").asInstanceOf[SBaseStructSettable], newFields.indices.map { i =>
      val code = newFields(i)
      val es = cb.emb.newEmitField(s"${ name }_nf_$i", code.emitType)
      es.store(cb, code)
      es
    })
  }

  override def _insert(newType: TStruct, fields: (String, EmitCode)*): SBaseStructCode = {
    val newFieldSet = fields.map(_._1).toSet
    val filteredNewFields = st.insertedFields.map(_._1)
      .zipWithIndex
      .filter { case (name, idx) => !newFieldSet.contains(name) }
      .map { case (name, idx) => (name, newFields(idx)) }
    parent._insert(newType, filteredNewFields ++ fields: _*)
  }

  override def loadSingleField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode = {
    st.getFieldIndexInNewOrParent(fieldIdx) match {
      case Left(parentIdx) => parent.loadSingleField(cb, parentIdx)
      case Right(newIdx) => newFields(newIdx).toI(cb)
    }
  }
}