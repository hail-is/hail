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

case class SJavaArrayString(elementRequired: Boolean) extends SContainer {
  def elementType: SType = SJavaString

  override def storageType(): PType = PCanonicalArray(PCanonicalString(elementRequired), false)

  override def copiedType: SType = this

  def containsPointers: Boolean = false

  lazy val virtualType: Type = TArray(TString)

  override def castRename(t: Type): SType = this

  def elementEmitType: EmitType = EmitType(elementType, elementRequired)

  def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    value.st match {
      case SJavaArrayString(_) => new SJavaArrayStringCode(this, value.asInstanceOf[SJavaArrayStringCode].array)
    }
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(arrayInfo[String])

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(arrayInfo[String])

  def fromSettables(settables: IndexedSeq[Settable[_]]): SJavaArrayStringSettable = {
    val IndexedSeq(a: Settable[Array[String]@unchecked]) = settables
    new SJavaArrayStringSettable(this, a)
  }

  def fromCodes(codes: IndexedSeq[Code[_]]): SJavaArrayStringCode = {
    val IndexedSeq(a: Code[Array[String]@unchecked]) = codes
    new SJavaArrayStringCode(this, a)
  }

  def construct(arr: Code[Array[String]]): SJavaArrayStringCode = new SJavaArrayStringCode(this, arr)

  def constructFromFunctionsKnownLength(cb: EmitCodeBuilder, region: Value[Region], length: Value[Int], deepCopy: Boolean):
  ((EmitCodeBuilder, IEmitCode) => Unit, EmitCodeBuilder => SJavaArrayStringCode) = {
    val arr = cb.newLocal("javastringarray", Code.newArray[String](length))
    val i = cb.newLocal[Int]("construct_javastringarray_idx", 0)
    val push: (EmitCodeBuilder, IEmitCode) => Unit = { (cb, iec) =>
      iec.consume(cb, {}, { sc => cb += (arr(i) = sc.asString.loadString()) })
      cb.assign(i, i + 1)
    }
    val finish: EmitCodeBuilder => SJavaArrayStringCode = { cb =>
      cb.ifx(i.cne(length),
        cb._fatal("SJavaArray.constructFromFunctions push was called the wrong number of times: len=",
          length.toS, ", calls=", i.toS))
      new SJavaArrayStringCode(this, arr)
    }
    push -> finish
  }

  def constructFromFunctionsUnknownLength(cb: EmitCodeBuilder, region: Value[Region], deepCopy: Boolean):
  ((EmitCodeBuilder, IEmitCode) => Unit, EmitCodeBuilder => SJavaArrayStringCode) = {
    val arr = cb.newLocal("javastringarray2", Code.newArray[String](8))
    val len = cb.newLocal[Int]("construct2_javastringarray_idx", 0)
    val push: (EmitCodeBuilder, IEmitCode) => Unit = { (cb, iec) =>
      cb.ifx(len.ceq(arr.length()), {
        val narr = cb.newLocal("new_javastringarray2", Code.newArray[String](arr.length() * 2))
        cb += Code.invokeStatic5[System, Array[String], Int, Array[String], Int, Int, Unit]("arraycopy", arr, 0, narr, 0, arr.length())
        cb.assign(arr, narr)
      })
      iec.consume(cb, {}, { sc => cb += (arr(len) = sc.asString.loadString()) })
      cb.assign(len, len + 1)
    }
    val finish: (EmitCodeBuilder => SJavaArrayStringCode) = { cb =>
      cb.ifx(len.cne(arr.length()), {
        val narr = cb.newLocal("final_javastringarray2", Code.newArray[String](len))
        cb += Code.invokeStatic5[System, Array[String], Int, Array[String], Int, Int, Unit]("arraycopy", arr, 0, narr, 0, len)
        cb.assign(arr, narr)
      })
      new SJavaArrayStringCode(this, arr)
    }
    push -> finish
  }
}


class SJavaArrayStringCode(val st: SJavaArrayString, val array: Code[Array[String]]) extends SIndexableCode {
  def makeCodeTuple(cb: EmitCodeBuilder): IndexedSeq[Code[_]] = FastIndexedSeq(array)

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

object SJavaArrayStringSettable {
  def apply(sb: SettableBuilder, st: SJavaArrayString, name: String): SJavaArrayStringSettable = {
    new SJavaArrayStringSettable(st,
      sb.newSettable[Array[String]](s"${ name }_arr"))
  }
}

class SJavaArrayStringSettable(
  val st: SJavaArrayString,
  val array: Settable[Array[String]]
) extends SIndexableValue with SSettable {
  def get: SIndexableCode = new SJavaArrayStringCode(st, array)

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(array)

  def loadLength(): Value[Int] = new Value[Int] {
    def get: Code[Int] = array.length()
  }

  def loadElement(cb: EmitCodeBuilder, i: Code[Int]): IEmitCode = {
    if (st.elementRequired)
      IEmitCode.present(cb, new SJavaStringCode(array(i)))
    else {
      val iv = cb.newLocal("pcindval_i", i)
      IEmitCode(cb,
        isElementMissing(cb, iv),
        new SJavaStringCode(array(i)))
    }
  }

  def isElementMissing(cb: EmitCodeBuilder, i: Code[Int]): Code[Boolean] = array(i).isNull

  def store(cb: EmitCodeBuilder, pc: SCode): Unit = {
    cb.assign(array, pc.asInstanceOf[SJavaArrayStringCode].array)
  }

  override def hasMissingValues(cb: EmitCodeBuilder): Code[Boolean] = {
    if (st.elementRequired)
      const(false)
    else
      Code.invokeScalaObject1[Array[String], Boolean](SJavaArrayHelpers.getClass, "hasNulls", array)
  }
}
