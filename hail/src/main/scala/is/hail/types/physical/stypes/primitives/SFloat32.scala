package is.hail.types.physical.stypes.primitives

import is.hail.annotations.Region
import is.hail.asm4s.{Code, FloatInfo, Settable, SettableBuilder, TypeInfo, Value}
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.stypes.{SCode, SSettable, SType, SValue}
import is.hail.types.physical.{PFloat32, PType}
import is.hail.types.virtual.{TFloat32, Type}
import is.hail.utils.FastIndexedSeq

case object SFloat32 extends SPrimitive {
  override def ti: TypeInfo[_] = FloatInfo

  override lazy val virtualType: Type = TFloat32

  override def castRename(t: Type): SType = this

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SValue = {
    value.st match {
      case SFloat32 => value
    }
  }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(FloatInfo)

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SFloat32Settable = {
    val IndexedSeq(x: Settable[Float@unchecked]) = settables
    assert(x.ti == FloatInfo)
    new SFloat32Settable(x)
  }

  override def fromValues(values: IndexedSeq[Value[_]]): SFloat32Value = {
    val IndexedSeq(x: Value[Float@unchecked]) = values
    assert(x.ti == FloatInfo)
    new SFloat32Value(x)
  }

  override def storageType(): PType = PFloat32()
}

class SFloat32Code(val code: Code[Float]) extends SPrimitiveCode {
  override def _primitiveCode: Code[_] = code

  val pt: PFloat32 = PFloat32(false)

  override def st: SFloat32.type = SFloat32

  private[this] def memoizeWithBuilder(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): SFloat32Value = {
    val s = new SFloat32Settable(sb.newSettable[Float]("sint64_memoize"))
    s.store(cb, this)
    s
  }

  override def memoize(cb: EmitCodeBuilder, name: String): SFloat32Value = memoizeWithBuilder(cb, name, cb.localBuilder)

  override def memoizeField(cb: EmitCodeBuilder, name: String): SFloat32Value = memoizeWithBuilder(cb, name, cb.fieldBuilder)

  def floatCode(cb: EmitCodeBuilder): Code[Float] = code
}

class SFloat32Value(x: Value[Float]) extends SPrimitiveValue {
  val pt: PFloat32 = PFloat32()

  override def valueTuple: IndexedSeq[Value[_]] = FastIndexedSeq(x)

  override def st: SFloat32.type = SFloat32

  override def _primitiveValue: Value[_] = x

  override def get: SCode = new SFloat32Code(x)

  def floatCode(cb: EmitCodeBuilder): Value[Float] = x

  override def hash(cb: EmitCodeBuilder): SInt32Code =
    new SInt32Code(Code.invokeStatic1[java.lang.Float, Float, Int]("floatToIntBits", floatCode(cb)))
}

object SFloat32Settable {
  def apply(sb: SettableBuilder, name: String): SFloat32Settable = {
    new SFloat32Settable(sb.newSettable[Float](name))
  }
}

final class SFloat32Settable(x: Settable[Float]) extends SFloat32Value(x) with SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(x)

  override def store(cb: EmitCodeBuilder, v: SCode): Unit = cb.assign(x, v.asFloat.floatCode(cb))
}
