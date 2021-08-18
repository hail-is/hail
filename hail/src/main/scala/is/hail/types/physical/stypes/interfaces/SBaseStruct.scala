package is.hail.types.physical.stypes.interfaces

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.PCanonicalStruct
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.concrete.{SInsertFieldsStruct, SInsertFieldsStructCode, SSubsetStruct, SSubsetStructCode}
import is.hail.types.physical.stypes.primitives.SInt32Code
import is.hail.types.virtual.{TBaseStruct, TStruct, TTuple}
import is.hail.types.{RField, RStruct, RTuple, TypeWithRequiredness}
import is.hail.utils._

trait SBaseStruct extends SType {
  def virtualType: TBaseStruct

  override def fromCodes(codes: IndexedSeq[Code[_]]): SBaseStructCode

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

  override def get: SBaseStructCode

  def isFieldMissing(fieldIdx: Int): Code[Boolean]

  def isFieldMissing(fieldName: String): Code[Boolean] = isFieldMissing(st.fieldIdx(fieldName))

  def loadField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode

  def loadField(cb: EmitCodeBuilder, fieldName: String): IEmitCode = loadField(cb, st.fieldIdx(fieldName))

  override def hash(cb: EmitCodeBuilder): SInt32Code = {
    val hash_result = cb.newLocal[Int]("hash_result_struct", 1)
    (0 until st.size).foreach(i => {
      loadField(cb, i).consume(cb, { cb.assign(hash_result, hash_result * 31) },
        {field => cb.assign(hash_result, (hash_result * 31) + field.memoize(cb, "struct_hash").hash(cb).intCode(cb))})
    })
    new SInt32Code(hash_result)
  }
}

trait SBaseStructCode extends SCode {
  self =>
  def st: SBaseStruct

  def memoize(cb: EmitCodeBuilder, name: String): SBaseStructValue

  def memoizeField(cb: EmitCodeBuilder, name: String): SBaseStructValue

  final def loadSingleField(cb: EmitCodeBuilder, fieldName: String): IEmitCode = loadSingleField(cb, st.fieldIdx(fieldName))

  def loadSingleField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode = {
    memoize(cb, "structcode_loadsinglefield")
      .loadField(cb, fieldIdx)
  }

  def subset(fieldNames: String*): SBaseStructCode = {
    val st = SSubsetStruct(self.st, fieldNames.toIndexedSeq)
    new SSubsetStructCode(st, self)
  }

  protected[stypes] def _insert(newType: TStruct, fields: (String, EmitCode)*): SBaseStructCode = {
    new SInsertFieldsStructCode(
      SInsertFieldsStruct(newType, st, fields.map { case (name, ec) => (name, ec.emitType) }.toFastIndexedSeq),
      this,
      fields.map(_._2).toFastIndexedSeq
    )
  }

  def insert(cb: EmitCodeBuilder, region: Value[Region], newType: TStruct, fields: (String, EmitCode)*): SBaseStructCode = {
    if (newType.size < 64 || fields.length < 16)
      return _insert(newType, fields: _*)

    val newFieldMap = fields.toMap
    val oldPV = memoize(cb, "insert_fields_old")
    val allFields = newType.fieldNames.map { f =>
      (f, newFieldMap.getOrElse(f, EmitCode.fromI(cb.emb)(cb => oldPV.loadField(cb, f)))) }

    val pcs = PCanonicalStruct(false, allFields.map { case (f, ec) => (f, ec.emitType.storageType) }: _*)
    pcs.constructFromFields(cb, region, allFields.map(_._2), false)
  }
}
