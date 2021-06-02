package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.PType
import is.hail.types.physical.stypes.interfaces.{SContainer, SIndexableCode, SIndexableValue}
import is.hail.types.physical.stypes.{EmitType, SCode, SSettable, SType}
import is.hail.types.virtual.{TArray, Type}
import is.hail.utils._

// levels indicate level of nesting 0 is array<T>, 1 is array<array<T>>, etc.
case class SNestedArray(levels: Int, baseContainerType: SContainer) extends SContainer {
   def elementType: SType = levels match {
     case 0 => baseContainerType.elementType
     case _ => SNestedArray(levels - 1, baseContainerType)
   }

  def elementEmitType: EmitType = levels match {
    case 0 => baseContainerType.elementEmitType
    case _ => ???
  }

  lazy val virtualType: Type = {
    var ty = baseContainerType.virtualType
    for (_ <- 0 until levels) {
      ty = TArray(ty)
    }
    ty
  }

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = ???

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = IndexedSeq.fill(2)(IntInfo) ++ IndexedSeq.fill(levels * 2)(LongInfo) ++ baseContainerType.codeTupleTypes()

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SNestedArraySettable = {
    val IndexedSeq(start: Settable[Int], end: Settable[Int], rest @ _*) = settables
    val (missing, rest1) = rest.toIndexedSeq.splitAt(levels)
    val (offsets, values) = rest1.splitAt(levels)
    new SNestedArraySettable(this,
      start, end,
      missing.map(_.asInstanceOf),
      offsets.map(_.asInstanceOf),
      baseContainerType.fromSettables(values).asInstanceOf)
  }

  override def fromCodes(codes: IndexedSeq[Code[_]]): SNestedArrayCode = {
    val IndexedSeq(start: Code[Int], end: Code[Int], rest @ _*) = codes
    val (missing, rest1) = rest.toIndexedSeq.splitAt(levels)
    val (offsets, values) = rest1.splitAt(levels)
    new SNestedArrayCode(this,
      start, end,
      missing.map(_.asInstanceOf),
      offsets.map(_.asInstanceOf),
      baseContainerType.fromCodes(values).asIndexable)
  }

  def canonicalPType(): PType = ???

  def castRename(t: Type): SType = t match {
    case TArray(t: TArray) =>
      castRename(t) // FIXME: does no checks
    case TArray(_) =>
      val st = baseContainerType.castRename(t)
      SNestedArray(levels, st.asInstanceOf[SContainer])
  }
}

class SNestedArraySettable(
  override val st: SNestedArray,
  val start: Settable[Int], val end: Settable[Int],
  val missing: IndexedSeq[Settable[Long]],
  val offsets: IndexedSeq[Settable[Long]],
  val values: SIndexableValue with SSettable
) extends SIndexableValue with SSettable {
  require(missing.length == st.levels)
  require(offsets.length == st.levels)

  private lazy val length: Value[Int] = new Value[Int] {
    def get: Code[Int] = end - start
  }

  def loadLength(): Value[Int] = length

  def isElementMissing(i: Code[Int]): Code[Boolean] = if (st.levels == 0)
    values.isElementMissing(i + start)
  else if (missing(0) == null)
    const(false)
  else
    Region.loadBit(missing(0), (i + start).toL)

  def loadElement(cb: EmitCodeBuilder, i: Code[Int]): IEmitCode = if (st.levels == 0) {
    values.loadElement(cb, i + start)
  } else {
    val iv = cb.newLocal("iv", i)
    IEmitCode(cb, isElementMissing(iv),
      new SNestedArrayCode(st.elementType.asInstanceOf, loadOffset(iv), loadOffset(iv + 1), missing.tail, offsets.tail, values.get))
  }

  def hasMissingValues(cb: EmitCodeBuilder): Code[Boolean] = {
    // FIXME: need to slice a bitvector to properly handle this
    if (st.levels == 0)
      values.hasMissingValues(cb) // wrong
    else if (missing(0) == null)
      const(false)
    else
      Region.containsNonZeroBits(missing(0), loadLength().toL) // also wrong
  }

  def get: SIndexableCode = new SNestedArrayCode(st, start, end, missing, offsets, values.get)

  def store(cb: EmitCodeBuilder, v: SCode): Unit = v match {
    case v: SNestedArrayCode =>
      cb.assign(start, v.start)
      cb.assign(end, v.end)
      missing.zip(v.missing).foreach(ms => if (ms._1 != null) cb.assign(ms._1, ms._2))
      offsets.zip(v.offsets).foreach(os => cb.assign(os._1, os._2))
      cb.assign(values, v.values)
  }

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(start, end) ++ missing ++ offsets ++ values.settableTuple()

  def loadOffset(i: Code[Int]): Code[Int] = {
    Region.loadInt(offsets(0) + (i + start).toL * 4L)
  }
}

class SNestedArrayCode(
  override val st: SNestedArray,
  val start: Code[Int], val end: Code[Int],
  val missing: IndexedSeq[Code[Long]], // all outer array missing bits, if null, then that level is required
  val offsets: IndexedSeq[Code[Long]], // all outer array offsets
  val values: SIndexableCode,          // the innermost, flat array
) extends SIndexableCode {
  require(missing.length == st.levels)
  require(offsets.length == st.levels)

  def loadLength(): Code[Int] = end - start

  private def memoize(cb: EmitCodeBuilder, name: String, values: SIndexableValue, sb: SettableBuilder): SNestedArraySettable = {
    val startSettable = sb.newSettable[Int](s"${ name }_start")
    val endSettable = sb.newSettable[Int](s"${ name }_end")
    val missingSettable = IndexedSeq.tabulate(st.levels)(i => sb.newSettable[Long](s"${ name }_missing_$i"))
    val offsetsSettable = IndexedSeq.tabulate(st.levels)(i => sb.newSettable[Long](s"${ name }_offsets_$i"))
    val sv = new SNestedArraySettable(st, startSettable, endSettable, missingSettable, offsetsSettable, values.asInstanceOf)
    cb.assign(sv, this)
    sv
  }

  override def memoize(cb: EmitCodeBuilder, name: String): SNestedArraySettable = memoize(cb, name, values.memoize(cb, s"${name}_values"), cb.localBuilder)

  override def memoizeField(cb: EmitCodeBuilder, name: String): SNestedArraySettable = memoize(cb, name, values.memoizeField(cb, s"${name}_values"), cb.fieldBuilder)

  def castToArray(cb: EmitCodeBuilder): SIndexableCode = this

  def makeCodeTuple(cb: EmitCodeBuilder): IndexedSeq[Code[_]] = FastIndexedSeq(start, end) ++ missing ++ offsets ++ values.makeCodeTuple(cb)
}