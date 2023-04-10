package is.hail.types.physical.stypes.interfaces

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, EmitValue, IEmitCode}
import is.hail.types.physical.PCanonicalStruct
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.primitives.{SInt32Value, SInt64Value}
import is.hail.types.virtual.{TBaseStruct, TStruct, TTuple}
import is.hail.types.{RField, RStruct, RTuple, TypeWithRequiredness}
import is.hail.utils._

object SBaseStruct {
  def merge(cb: EmitCodeBuilder, s1: SBaseStructValue, s2: SBaseStructValue): SBaseStructValue = {
    val lt = s1.st.virtualType.asInstanceOf[TStruct]
    val rt = s2.st.virtualType.asInstanceOf[TStruct]
    val resultVType = TStruct.concat(lt, rt)

    val st1 = s1.st
    val st2 = s2.st

    (s1, s2) match {
      case (s1, s2: SStackStructValue) =>
        s1._insert(resultVType, rt.fieldNames.zip(s2.values): _*)
      case (s1: SStackStructValue, s2) =>
        s2._insert(resultVType, lt.fieldNames.zip(s1.values): _*)
      case _ =>
        val newVals = (0 until st2.size).map(i => cb.memoize(s2.loadField(cb, i), "InsertFieldsStruct_merge"))
        s1._insert(resultVType, rt.fieldNames.zip(newVals): _*)
    }
  }
}

trait SBaseStruct extends SType {
  def virtualType: TBaseStruct

  def size: Int

  val fieldTypes: IndexedSeq[SType]
  val fieldEmitTypes: IndexedSeq[EmitType]

  def fieldIdx(fieldName: String): Int

  def _typeWithRequiredness: TypeWithRequiredness = {
    virtualType match {
      case ts: TStruct => RStruct.fromNamesAndTypes(ts.fieldNames.zip(fieldEmitTypes).map {
        case (name, et) => (name, et.typeWithRequiredness.r)
      })
      case tt: TTuple => RTuple.fromNamesAndTypes(tt._types.zip(fieldEmitTypes).map {
        case (f, et) => (f.index.toString, et.typeWithRequiredness.r)
      })
    }
  }
}

trait SBaseStructSettable extends SBaseStructValue with SSettable

trait SBaseStructValue extends SValue {
  def st: SBaseStruct

  def isFieldMissing(cb: EmitCodeBuilder, fieldIdx: Int): Value[Boolean]

  def isFieldMissing(cb: EmitCodeBuilder, fieldName: String): Value[Boolean] =
    isFieldMissing(cb, st.fieldIdx(fieldName))

  def loadField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode

  def loadField(cb: EmitCodeBuilder, fieldName: String): IEmitCode = loadField(cb, st.fieldIdx(fieldName))

  def subset(fieldNames: String*): SBaseStructValue = {
    val st = SSubsetStruct(this.st, fieldNames.toIndexedSeq)
    new SSubsetStructValue(st, this)
  }

  override def hash(cb: EmitCodeBuilder): SInt32Value = {
    val hash_result = cb.newLocal[Int]("hash_result_struct", 1)
    (0 until st.size).foreach(i => {
      loadField(cb, i).consume(cb, { cb.assign(hash_result, hash_result * 31) },
        {field => cb.assign(hash_result, (hash_result * 31) + field.hash(cb).value)})
    })
    new SInt32Value(hash_result)
  }

  override def sizeToStoreInBytes(cb: EmitCodeBuilder): SInt64Value = {
    // Size in bytes of the struct that must represent this thing, plus recursive call on any non-missing children.
    val pStructSize = this.st.storageType().byteSize
    val sizeSoFar = cb.newLocal[Long]("sstackstruct_size_in_bytes", pStructSize)
    (0 until st.size).foreach { idx =>
      if (this.st.fieldTypes(idx).containsPointers) {
        val sizeAtThisIdx: Value[Long] = this.loadField(cb, idx).consumeCode(cb, {
          const(0L)
        }, { sv =>
          sv.sizeToStoreInBytes(cb).value
        })
        cb.assign(sizeSoFar, sizeSoFar + sizeAtThisIdx)
      }
    }
    new SInt64Value(sizeSoFar)
  }

  def toStackStruct(cb: EmitCodeBuilder): SStackStructValue = {
    new SStackStructValue(
      SStackStruct(st.virtualType, st.fieldEmitTypes),
      Array.tabulate(st.size)( i => cb.memoize(loadField(cb, i))))
  }

  def _insert(newType: TStruct, fields: (String, EmitValue)*): SBaseStructValue = {
    new SInsertFieldsStructValue(
      SInsertFieldsStruct(newType, st, fields.map { case (name, ec) => (name, ec.emitType) }.toFastIndexedSeq),
      this,
      fields.map(_._2).toFastIndexedSeq
    )
  }

  def insert(cb: EmitCodeBuilder, region: Value[Region], newType: TStruct, fields: (String, EmitValue)*): SBaseStructValue = {
    if (st.settableTupleTypes().length + fields.map(_._2.emitType.settableTupleTypes.length).sum < 64)
      return _insert(newType, fields: _*)

    val newFieldMap = fields.toMap
    val allFields = newType.fieldNames.map { f =>
      (f, newFieldMap.getOrElse(f, cb.memoize(EmitCode.fromI(cb.emb)(cb => loadField(cb, f)), "insert"))) }

    val pcs = PCanonicalStruct(false, allFields.map { case (f, ec) => (f, ec.emitType.storageType) }: _*)
    pcs.constructFromFields(cb, region, allFields.map(_._2.load), false)
  }
}
