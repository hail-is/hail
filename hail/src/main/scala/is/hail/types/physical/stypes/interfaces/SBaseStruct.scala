package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s.Code
import is.hail.expr.ir.{EmitCodeBuilder, IEmitSCode}
import is.hail.types.physical.PBaseStruct
import is.hail.types.physical.stypes.concrete.{SSubsetStruct, SSubsetStructCode}
import is.hail.types.physical.stypes.{SCode, SSettable, SType, SValue}

trait SBaseStruct extends SType {
  override def fromCodes(codes: IndexedSeq[Code[_]]): SBaseStructCode

  def size: Int

  val fieldTypes: Array[SType]
}

trait SStructSettable extends SBaseStructValue with SSettable

trait SBaseStructValue extends SValue {
  def pt: PBaseStruct

  def isFieldMissing(fieldIdx: Int): Code[Boolean]

  def isFieldMissing(fieldName: String): Code[Boolean] = isFieldMissing(pt.fieldIdx(fieldName))

  def loadField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitSCode

  def loadField(cb: EmitCodeBuilder, fieldName: String): IEmitSCode = loadField(cb, pt.fieldIdx(fieldName))
}

trait SBaseStructCode extends SCode { self =>
  def st: SBaseStruct

  def memoize(cb: EmitCodeBuilder, name: String): SBaseStructValue

  def memoizeField(cb: EmitCodeBuilder, name: String): SBaseStructValue

  def subset(fieldNames: String*): SSubsetStructCode = {
    val st = SSubsetStruct(self.st, fieldNames.toIndexedSeq)
    new SSubsetStructCode(st, self.asPCode.asBaseStruct) // FIXME, should be sufficient to just use self here
  }
}
