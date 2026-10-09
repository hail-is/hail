package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.collection.FastSeq
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder}
import is.hail.types.physical.{PPackedLocus, PType}
import is.hail.types.physical.stypes.{SSettable, SType, SValue}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives.{SInt32Value, SInt64Value}
import is.hail.types.virtual.{TLocus, Type}

final case class SPackedLocus(rg: String) extends SLocus {
  override lazy val virtualType: TLocus = TLocus(rg)

  override def contigType: SString = SJavaString

  override def castRename(t: Type): SType = this

  override def _coerceOrCopy(
    cb: EmitCodeBuilder,
    region: Value[Region],
    value: SValue,
    deepCopy: Boolean,
  ): SValue =
    value match {
      case value: SPackedLocusValue => value
      case value: SLocusValue => new SPackedLocusValue(this, value.packed(cb))
    }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastSeq(LongInfo)

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SPackedLocusSettable = {
    val IndexedSeq(packed: Settable[Long @unchecked]) = settables
    assert(packed.ti == LongInfo)
    new SPackedLocusSettable(this, packed)
  }

  override def fromValues(values: IndexedSeq[Value[_]]): SPackedLocusValue = {
    val IndexedSeq(packed: Value[Long @unchecked]) = values
    assert(packed.ti == LongInfo)
    new SPackedLocusValue(this, packed)
  }

  override def storageType(): PType = PPackedLocus(rg, required = false)

  override def copiedType: SType = this

  override def containsPointers: Boolean = false
}

class SPackedLocusValue(val st: SPackedLocus, val _packed: Value[Long]) extends SLocusValue {
  override lazy val valueTuple: IndexedSeq[Value[_]] = FastSeq(_packed)

  override def contig(cb: EmitCodeBuilder): SStringValue =
    SJavaString.construct(
      cb,
      cb.emb.getReferenceGenome(st.rg).invoke[Int, String]("getContig", contigIdx(cb)),
    )

  override def contigIdx(cb: EmitCodeBuilder): Value[Int] = cb.memoize((_packed >>> 32).toI)

  override def position(cb: EmitCodeBuilder): Value[Int] = cb.memoize(_packed.toI)

  override def packed(cb: EmitCodeBuilder): Value[Long] = _packed

  override def structRepr(cb: EmitCodeBuilder): SBaseStructValue =
    SStackStruct.constructFromArgs(
      cb,
      null,
      st.virtualType.representation,
      EmitCode.present(cb.emb, contig(cb)),
      EmitCode.present(cb.emb, primitive(position(cb))),
    )

  override def hash(cb: EmitCodeBuilder): SInt32Value =
    new SInt32Value(cb.memoize(Code.invokeStatic1[java.lang.Long, Long, Int]("hashCode", _packed)))

  override def sizeToStoreInBytes(cb: EmitCodeBuilder): SInt64Value =
    new SInt64Value(st.storageType().byteSize)
}

object SPackedLocusSettable {
  def apply(sb: SettableBuilder, st: SPackedLocus, name: String): SPackedLocusSettable =
    new SPackedLocusSettable(st, sb.newSettable[Long](s"${name}_packed"))
}

final class SPackedLocusSettable(st: SPackedLocus, override val _packed: Settable[Long])
    extends SPackedLocusValue(st, _packed) with SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = FastSeq(_packed)

  override def store(cb: EmitCodeBuilder, v: SValue): Unit =
    cb.assign(_packed, v.asLocus.packed(cb))
}
