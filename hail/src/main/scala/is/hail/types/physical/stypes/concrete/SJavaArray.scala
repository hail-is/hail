package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.stypes.interfaces.{SContainer, SIndexableCode, SIndexableValue}
import is.hail.types.physical.stypes.{EmitType, SCode, SSettable, SType}
import is.hail.types.physical.{PCanonicalArray, PCanonicalString, PType}
import is.hail.types.virtual.{TArray, TString, Type}
import is.hail.utils.FastIndexedSeq

object SJavaArrayHelpers {
  def hasNulls(a: Array[String]): Boolean = {
    var i = 0
    while (i < a.length) {
      if (a(i) == null)
        return true
      i += 1
    }
    false
  }
}

final case class SJavaArrayString(elementRequired: Boolean) extends SContainer {
  override def elementType: SType = SJavaString

  override def storageType(): PType = PCanonicalArray(PCanonicalString(elementRequired), false)

  override def copiedType: SType = this

  override def containsPointers: Boolean = false

  override lazy val virtualType: Type = TArray(TString)

  override def castRename(t: Type): SType = this

  override def elementEmitType: EmitType = EmitType(elementType, elementRequired)

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    value.st match {
      case SJavaArrayString(_) => new SJavaArrayStringCode(this, value.asInstanceOf[SJavaArrayStringCode].array)
    }
  }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(arrayInfo[String])

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SJavaArrayStringSettable = {
    val IndexedSeq(a: Settable[Array[String]@unchecked]) = settables
    new SJavaArrayStringSettable(this, a)
  }

  override def fromValues(values: IndexedSeq[Value[_]]): SJavaArrayStringValue = {
    val IndexedSeq(a: Value[Array[String]@unchecked]) = values
    new SJavaArrayStringValue(this, a)
  }

  def construct(arr: Code[Array[String]]): SJavaArrayStringCode = new SJavaArrayStringCode(this, arr)
}

class SJavaArrayStringCode(val st: SJavaArrayString, val array: Code[Array[String]]) extends SIndexableCode {
  def codeLoadLength(): Code[Int] = array.length()

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): SIndexableValue = {
    val s = SJavaArrayStringSettable(sb, st, name)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): SIndexableValue = memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): SIndexableValue = memoize(cb, name, cb.fieldBuilder)

  def castToArray(cb: EmitCodeBuilder): SIndexableCode = this
}

class SJavaArrayStringValue(
  val st: SJavaArrayString,
  val array: Value[Array[String]]
) extends SIndexableValue {
  override lazy val valueTuple: IndexedSeq[Value[_]] = FastIndexedSeq(array)

  override def get: SIndexableCode = new SJavaArrayStringCode(st, array)

  override def loadLength(): Value[Int] = new Value[Int] {
    override def get: Code[Int] = array.length()
  }

  override def loadElement(cb: EmitCodeBuilder, i: Code[Int]): IEmitCode = {
    if (st.elementRequired)
      IEmitCode.present(cb, new SJavaStringCode(array(i)))
    else {
      val iv = cb.newLocal("pcindval_i", i)
      IEmitCode(cb,
        isElementMissing(iv),
        new SJavaStringCode(array(i)))
    }
  }

  override def isElementMissing(i: Code[Int]): Code[Boolean] = array(i).isNull

  override def hasMissingValues(cb: EmitCodeBuilder): Code[Boolean] = {
    if (st.elementRequired)
      const(false)
    else
      Code.invokeScalaObject1[Array[String], Boolean](SJavaArrayHelpers.getClass, "hasNulls", array)
  }

  override def castToArray(cb: EmitCodeBuilder): SIndexableValue = this
}

object SJavaArrayStringSettable {
  def apply(sb: SettableBuilder, st: SJavaArrayString, name: String): SJavaArrayStringSettable = {
    new SJavaArrayStringSettable(st,
      sb.newSettable[Array[String]](s"${ name }_arr"))
  }
}

final class SJavaArrayStringSettable(
  st: SJavaArrayString,
  override val array: Settable[Array[String]]
) extends SJavaArrayStringValue(st, array) with SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(array)

  override def store(cb: EmitCodeBuilder, pc: SCode): Unit = {
    cb.assign(array, pc.asInstanceOf[SJavaArrayStringCode].array)
  }
}
