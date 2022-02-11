package is.hail.types.physical.stypes.interfaces

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, EmitValue, IEmitCode}
import is.hail.types.physical.PCanonicalStruct
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.primitives.SInt32Value
import is.hail.types.virtual.{TBaseStruct, TStruct, TTuple}
import is.hail.types.{RField, RStruct, RTuple, TypeWithRequiredness}
import is.hail.utils._

trait SBaseStruct extends SType {
  def virtualType: TBaseStruct

  def size: Int

  val fieldTypes: IndexedSeq[SType]
  val fieldEmitTypes: IndexedSeq[EmitType]

  def fieldIdx(fieldName: String): Int

  def _typeWithRequiredness: TypeWithRequiredness = {
    virtualType match {
      case ts: TStruct => RStruct(ts.fieldNames.zip(fieldEmitTypes).map { case (name, et) => (name, et.typeWithRequiredness.r) })
      case tt: TTuple => RTuple(tt.fields.zip(fieldEmitTypes).map { case (f, et) => RField(f.name, et.typeWithRequiredness.r, f.index) })
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

  protected[stypes] def _insert(newType: TStruct, fields: (String, EmitValue)*): SBaseStructValue = {
    new SInsertFieldsStructValue(
      SInsertFieldsStruct(newType, st, fields.map { case (name, ec) => (name, ec.emitType) }.toFastIndexedSeq),
      this,
      fields.map(_._2).toFastIndexedSeq
    )
  }

  def insert(cb: EmitCodeBuilder, region: Value[Region], newType: TStruct, fields: (String, EmitValue)*): SBaseStructValue = {
    if (newType.size < 64 || fields.length < 16)
      return _insert(newType, fields: _*)

    val newFieldMap = fields.toMap
    val allFields = newType.fieldNames.map { f =>
      (f, newFieldMap.getOrElse(f, cb.memoize(EmitCode.fromI(cb.emb)(cb => loadField(cb, f)), "insert"))) }

    val pcs = PCanonicalStruct(false, allFields.map { case (f, ec) => (f, ec.emitType.storageType) }: _*)
    pcs.constructFromFields(cb, region, allFields.map(_._2.load), false)
  }
}
