package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.{PCanonicalArray, PCanonicalString, PString, PType}
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.interfaces.{SContainer, SIndexableValue, SStringValue}
import is.hail.types.virtual.{TArray, TString, Type}
import is.hail.utils.FastSeq

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

  override def _coerceOrCopy(
    cb: EmitCodeBuilder,
    region: Value[Region],
    value: SValue,
    deepCopy: Boolean,
  ): SValue = {
    value.st match {
      case SJavaArrayString(_) =>
        new SJavaArrayStringValue(this, value.asInstanceOf[SJavaArrayStringValue].array)
      case SIndexablePointer(pc) if pc.elementType.isInstanceOf[PString] =>
        val sv = value.asInstanceOf[SIndexableValue]
        val len = sv.loadLength()
        val array = cb.memoize[Array[String]](Code.newArray[String](len))
        (pc.elementType.required, elementRequired) match {
          case (true, _) =>
            sv.forEachDefined(cb) { case (cb, i, v: SStringValue) =>
              cb += (array(i) = v.loadString(cb))
            }
          case (false, r) =>
            sv.forEachDefinedOrMissing(cb)(
              { case (cb, i) =>
                if (r)
                  cb._fatal(
                    "requiredness mismatch: found missing value at index ",
                    i.toS,
                    s" coercing ${sv.st} to $this",
                  )
              },
              { case (cb, i, elt) =>
                cb += (array(i) = elt.asString.loadString(cb))
              },
            )
          case (false, false) =>
        }
        new SJavaArrayStringValue(this, array)
    }
  }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastSeq(arrayInfo[String])

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SJavaArrayStringSettable = {
    val IndexedSeq(a: Settable[Array[String] @unchecked]) = settables
    new SJavaArrayStringSettable(this, a)
  }

  override def fromValues(values: IndexedSeq[Value[_]]): SJavaArrayStringValue = {
    val IndexedSeq(a: Value[Array[String] @unchecked]) = values
    new SJavaArrayStringValue(this, a)
  }

  def construct(cb: EmitCodeBuilder, arr: Code[Array[String]]): SJavaArrayStringValue =
    new SJavaArrayStringValue(this, cb.memoize(arr))

  override def isIsomorphicTo(st: SType): Boolean =
    st match {
      case a: SJavaArrayString => elementRequired == a.elementRequired
      case _ => false
    }
}

class SJavaArrayStringValue(
  val st: SJavaArrayString,
  val array: Value[Array[String]],
) extends SIndexableValue {
  override lazy val valueTuple: IndexedSeq[Value[_]] = FastSeq(array)

  override def loadLength(): Value[Int] = new Value[Int] {
    override def get: Code[Int] = array.length()
  }

  override def loadElement(cb: EmitCodeBuilder, i: Code[Int]): IEmitCode = {
    if (st.elementRequired)
      IEmitCode.present(cb, new SJavaStringValue(cb.memoize(array(i))))
    else {
      val iv = cb.memoize(i)
      IEmitCode(cb, isElementMissing(cb, iv), new SJavaStringValue(cb.memoize(array(iv))))
    }
  }

  override def isElementMissing(cb: EmitCodeBuilder, i: Code[Int]): Value[Boolean] =
    cb.memoize(array(i).isNull)

  override def hasMissingValues(cb: EmitCodeBuilder): Value[Boolean] =
    if (st.elementRequired)
      const(false)
    else
      cb.memoize(Code.invokeScalaObject1[Array[String], Boolean](
        SJavaArrayHelpers.getClass,
        "hasNulls",
        array,
      ))

  override def castToArray(cb: EmitCodeBuilder): SIndexableValue = this
}

object SJavaArrayStringSettable {
  def apply(sb: SettableBuilder, st: SJavaArrayString, name: String): SJavaArrayStringSettable =
    new SJavaArrayStringSettable(st, sb.newSettable[Array[String]](s"${name}_arr"))
}

final class SJavaArrayStringSettable(
  st: SJavaArrayString,
  override val array: Settable[Array[String]],
) extends SJavaArrayStringValue(st, array) with SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = FastSeq(array)

  override def store(cb: EmitCodeBuilder, v: SValue): Unit =
    cb.assign(array, v.asInstanceOf[SJavaArrayStringValue].array)
}
