package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s.{Code, Settable, TypeInfo, Value}
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, EmitSettable, IEmitCode}
import is.hail.types.physical.stypes.interfaces.{SBaseStruct, SBaseStructCode, SStructSettable}
import is.hail.types.physical.stypes.{EmitType, SCode, SType}
import is.hail.types.physical.{PCanonicalStruct, PType}
import is.hail.types.virtual.{TStruct, Type}
import is.hail.utils.BoxedArrayBuilder

case class SInsertFieldsStruct(virtualType: TStruct, parent: SBaseStruct, newFields: IndexedSeq[(String, EmitType)]) extends SBaseStruct {
  override def size: Int = virtualType.size

  lazy val newFieldMap: Map[String, Int] = newFields.map(_._1).zipWithIndex.toMap

  def getFieldIndexInNewOrParent(idx: Int): Either[Int, Int] = {
    val fieldName = virtualType.fieldNames(idx)
    newFieldMap.get(fieldName) match {
      case Some(idx) => Right(idx)
      case None => Left(parent.fieldIdx(fieldName))
    }
  }

  override lazy val fieldEmitTypes: IndexedSeq[EmitType] = virtualType.fieldNames.map(f => newFieldMap.get(f) match {
    case Some(idx) => newFields(idx)._2
    case None => parent.fieldEmitTypes(parent.fieldIdx(f))
  })

  private lazy val newFieldCodeStarts = fieldEmitTypes.map(_.nCodes).scanLeft(0)(_ + _).init
  private lazy val newFieldSettableStarts = fieldEmitTypes.map(_.nSettables).scanLeft(0)(_ + _).init

  override lazy val fieldTypes: IndexedSeq[SType] = fieldEmitTypes.map(_.st)

  override def fieldIdx(fieldName: String): Int = virtualType.fieldIdx(fieldName)

  override def canonicalPType(): PType = PCanonicalStruct(false, virtualType.fieldNames.zip(fieldEmitTypes).map { case (f, et) => (f, et.canonicalPType) }: _*)

  lazy val codeTupleTypes: IndexedSeq[TypeInfo[_]] = parent.codeTupleTypes ++ newFields.flatMap(_._2.codeTupleTypes)

  override lazy val settableTupleTypes: IndexedSeq[TypeInfo[_]] = parent.settableTupleTypes() ++ newFields.flatMap(_._2.settableTupleTypes)

  override def fromCodes(codes: IndexedSeq[Code[_]]): SInsertFieldsStructCode = {
    assert(codes.map(_.ti) == codeTupleTypes)
    new SInsertFieldsStructCode(this, parent.fromCodes(codes.take(parent.nCodes)).asInstanceOf[SBaseStructCode], newFields.indices.map { i =>
      val et = newFields(i)._2
      val start = newFieldCodeStarts(i) + parent.nCodes
      et.fromCodes(codes.slice(start, start + et.nCodes))
    })
  }

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SInsertFieldsStructSettable = {
    assert(settables.map(_.ti) == settableTupleTypes)
    new SInsertFieldsStructSettable(this, parent.fromSettables(settables.take(parent.nSettables)).asInstanceOf[SStructSettable], newFields.indices.map { i =>
      val et = newFields(i)._2
      val start = newFieldSettableStarts(i) + parent.nSettables
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

    val newFieldBuilder = new BoxedArrayBuilder[(String, EmitType)]()
    val parentPassThroughFieldBuilder = new BoxedArrayBuilder[(String, Type)]()

    (0 until ts.size).foreach { i =>
      val newField = ts.fields(i)
      val name = newField.name
      newFieldMap.get(name) match {
        case Some(idx) =>
          val et = fieldEmitTypes(idx)
          newFieldBuilder += ((name, et.copy(st = et.st.castRename(newField.typ))))
        case None => parentPassThroughFieldBuilder += ((name, newField.typ))
      }
    }

    val parentPassThroughMap = parentPassThroughFieldBuilder.result().toMap
    val parentCastType = TStruct(parentType.fieldNames.map(f => (f, parentPassThroughMap.getOrElse(f, parentType.fieldType(f)))): _*)
    val renamedParentType = parent.castRename(parentCastType)
    SInsertFieldsStruct(ts,
      renamedParentType.asInstanceOf[SBaseStruct],
      newFieldBuilder.result()
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

  override def insert(newType: TStruct, fields: (String, EmitCode)*): SBaseStructCode = {
    val newFieldSet = fields.map(_._1).toSet
    val filteredNewFields = st.newFields.map(_._1)
      .zipWithIndex
      .filter { case (name, idx) => !newFieldSet.contains(name) }
      .map { case (name, idx) => (name, newFields(idx)) }
    parent.insert(newType, filteredNewFields ++ fields: _*)
  }
}