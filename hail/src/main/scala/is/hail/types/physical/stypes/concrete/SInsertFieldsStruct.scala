package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s.{Code, Settable, TypeInfo, Value}
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, EmitSettable, IEmitCode}
import is.hail.types.physical.stypes.interfaces.{SBaseStruct, SBaseStructCode, SStructSettable}
import is.hail.types.physical.stypes.{EmitType, SCode, SType}
import is.hail.types.physical.{PCanonicalStruct, PType}
import is.hail.types.virtual.{TStruct, Type}
import is.hail.utils.BoxedArrayBuilder

case class SInsertFieldsStruct(virtualType: TStruct, parent: SBaseStruct, insertedFields: IndexedSeq[(String, EmitType)]) extends SBaseStruct {
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

  val fieldEmitTypes: IndexedSeq[EmitType] = virtualType.fieldNames.zipWithIndex.map { case (f, idx) =>
    insertedFieldIndices.get(idx) match {
      case Some(idx) => insertedFields(idx)._2
      case None => parent.fieldEmitTypes(parent.fieldIdx(f))
    }
  }

  private lazy val insertedFieldCodeStarts = insertedFields.map(_._2.nCodes).scanLeft(0)(_ + _).init
  private lazy val insertedFieldSettableStarts = insertedFields.map(_._2.nSettables).scanLeft(0)(_ + _).init

  override lazy val fieldTypes: IndexedSeq[SType] = fieldEmitTypes.map(_.st)

  override def fieldIdx(fieldName: String): Int = virtualType.fieldIdx(fieldName)

  override def canonicalPType(): PType = PCanonicalStruct(false, virtualType.fieldNames.zip(fieldEmitTypes).map { case (f, et) => (f, et.canonicalPType) }: _*)

  lazy val codeTupleTypes: IndexedSeq[TypeInfo[_]] = parent.codeTupleTypes ++ insertedFields.flatMap(_._2.codeTupleTypes)

  override lazy val settableTupleTypes: IndexedSeq[TypeInfo[_]] = parent.settableTupleTypes() ++ insertedFields.flatMap(_._2.settableTupleTypes)

  override def fromCodes(codes: IndexedSeq[Code[_]]): SInsertFieldsStructCode = {
    assert(codes.map(_.ti) == codeTupleTypes)
    new SInsertFieldsStructCode(this, parent.fromCodes(codes.take(parent.nCodes)).asInstanceOf[SBaseStructCode], insertedFields.indices.map { i =>
      val et = insertedFields(i)._2
      val start = insertedFieldCodeStarts(i) + parent.nCodes
      et.fromCodes(codes.slice(start, start + et.nCodes))
    })
  }

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SInsertFieldsStructSettable = {
    assert(settables.map(_.ti) == settableTupleTypes)
    new SInsertFieldsStructSettable(this, parent.fromSettables(settables.take(parent.nSettables)).asInstanceOf[SStructSettable], insertedFields.indices.map { i =>
      val et = insertedFields(i)._2
      val start = insertedFieldSettableStarts(i) + parent.nSettables
      et.fromSettables(settables.slice(start, start + et.nSettables))
    })
  }

  override def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    value match {
      case ss: SInsertFieldsStructCode if ss.st == this => value
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

class SInsertFieldsStructSettable(val st: SInsertFieldsStruct, parent: SStructSettable, newFields: IndexedSeq[EmitSettable]) extends SStructSettable {
  def get: SInsertFieldsStructCode = new SInsertFieldsStructCode(st, parent.load().asBaseStruct, newFields.map(_.load))

  def settableTuple(): IndexedSeq[Settable[_]] = parent.settableTuple() ++ newFields.flatMap(_.settableTuple())

  def loadField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode = {
    st.getFieldIndexInNewOrParent(fieldIdx) match {
      case Left(parentIdx) => parent.loadField(cb, parentIdx)
      case Right(newFieldsIdx) => newFields(newFieldsIdx).toI(cb)
    }
  }

  def isFieldMissing(fieldIdx: Int): Code[Boolean] =
    st.getFieldIndexInNewOrParent(fieldIdx) match {
      case Left(parentIdx) => parent.isFieldMissing(parentIdx)
      case Right(newFieldsIdx) => newFields(newFieldsIdx).m
    }

  def store(cb: EmitCodeBuilder, pv: SCode): Unit = {
    val sifc = pv.asInstanceOf[SInsertFieldsStructCode]
    parent.store(cb, sifc.parent)
    newFields.zip(sifc.newFields).foreach { case (settable, code) => cb.assign(settable, code) }
  }
}

class SInsertFieldsStructCode(val st: SInsertFieldsStruct, val parent: SBaseStructCode, val newFields: IndexedSeq[EmitCode]) extends SBaseStructCode {
  override def makeCodeTuple(cb: EmitCodeBuilder): IndexedSeq[Code[_]] = parent.makeCodeTuple(cb) ++ newFields.flatMap(_.makeCodeTuple(cb))

  override def memoize(cb: EmitCodeBuilder, name: String): SInsertFieldsStructSettable = {
    new SInsertFieldsStructSettable(st, parent.memoize(cb, name + "_parent").asInstanceOf[SStructSettable], newFields.indices.map { i =>
      val code = newFields(i)
      val es = cb.emb.newEmitLocal(s"${ name }_nf_$i", code.emitType)
      es.store(cb, code)
      es
    })
  }

  override def memoizeField(cb: EmitCodeBuilder, name: String): SInsertFieldsStructSettable = {
    new SInsertFieldsStructSettable(st, parent.memoizeField(cb, name + "_parent").asInstanceOf[SStructSettable], newFields.indices.map { i =>
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