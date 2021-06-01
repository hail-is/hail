package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.PType
import is.hail.types.physical.stypes.interfaces.{SContainer, SIndexableCode, SIndexableValue}
import is.hail.types.physical.stypes.{EmitType, SCode, SSettable, SType}
import is.hail.types.virtual.{TArray, Type}

// levels indicate level of nesting 0 would be array<T>, but is disallowed, 1 is array<array<T>>, etc.
case class SNestedArray(levels: Int, baseContainerType: SContainer) extends SContainer {
  require(levels > 0)

  def elementType: SType = ???

  def elementEmitType: EmitType = ???

  lazy val virtualType: Type = {
    var ty = baseContainerType.virtualType
    for (_ <- 0 until levels) {
      ty = TArray(ty)
    }
    ty
  }

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = ???

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = IndexedSeq.fill(levels)(IntInfo) ++ IndexedSeq.fill(levels * 2)(LongInfo) ++ baseContainerType.codeTupleTypes()

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SNestedArraySettable = {
    val (lengths, rest) = settables.splitAt(levels)
    val (missing, rest1) = rest.splitAt(levels)
    val (offsets, values) = rest1.splitAt(levels)
    new SNestedArraySettable(this,
      lengths.map(_.asInstanceOf),
      missing.map(_.asInstanceOf),
      offsets.map(_.asInstanceOf),
      baseContainerType.fromSettables(values).asInstanceOf)
  }

  override def fromCodes(codes: IndexedSeq[Code[_]]): SNestedArrayCode = {
    val (lengths, rest) = codes.splitAt(levels)
    val (missing, rest1) = rest.splitAt(levels)
    val (offsets, values) = rest1.splitAt(levels)
    new SNestedArrayCode(this,
      lengths.map(_.asInstanceOf),
      missing.map(_.asInstanceOf),
      offsets.map(_.asInstanceOf),
      baseContainerType.fromCodes(values).asIndexable)
  }

  def canonicalPType(): PType = ???

  def castRename(t: Type): SType = ???
}

class SNestedArraySettable(
  override val st: SNestedArray,
  val lengths: IndexedSeq[Settable[Int]],
  val missing: IndexedSeq[Settable[Long]],
  val offsets: IndexedSeq[Settable[Long]],
  val values: SIndexableValue with SSettable
) extends SIndexableValue with SSettable {
  require(lengths.length == st.levels)
  require(missing.length == st.levels)
  require(offsets.length == st.levels)

  def loadLength(): Value[Int] = lengths(0)

  def isElementMissing(i: Code[Int]): Code[Boolean] = if (missing(0) == null)
    const(false)
  else
    Region.loadBit(missing(0), i.toL)

  def loadElement(cb: EmitCodeBuilder, i: Code[Int]): IEmitCode = {
    val iv = cb.newLocal("iv", i)
    IEmitCode(cb, isElementMissing(iv), ???)
  }

  def hasMissingValues(cb: EmitCodeBuilder): Code[Boolean] = if (missing(0) == null)
    const(false)
  else
    Region.containsNonZeroBits(offsets(0), loadLength().toL)

  def get: SIndexableCode = new SNestedArrayCode(st, lengths, missing, offsets, values.get)

  def store(cb: EmitCodeBuilder, v: SCode): Unit = v match {
    case v: SNestedArrayCode =>
      lengths.zip(v.lengths).foreach(ls => cb.assign(ls._1, ls._2))
      missing.zip(v.missing).foreach(ms => if (ms._1 != null) cb.assign(ms._1, ms._2))
      offsets.zip(v.offsets).foreach(os => cb.assign(os._1, os._2))
      cb.assign(values, v.values)
  }

  def settableTuple(): IndexedSeq[Settable[_]] = lengths ++ missing ++ offsets ++ values.settableTuple()
}

class SNestedArrayCode(
  override val st: SNestedArray,
  val lengths: IndexedSeq[Code[Int]],  // all outer array lengths
  val missing: IndexedSeq[Code[Long]], // all outer array missing bits, if null, then that level is required
  val offsets: IndexedSeq[Code[Long]], // all outer array offsets
  val values: SIndexableCode,          // the innermost, flat array
) extends SIndexableCode {
  require(lengths.length == st.levels)
  require(missing.length == st.levels)
  require(offsets.length == st.levels)

  def loadLength(): Code[Int] = lengths(0)

  private def memoize(cb: EmitCodeBuilder, name: String, values: SIndexableValue, sb: SettableBuilder): SNestedArraySettable = {
    val lengthsSettable = IndexedSeq.tabulate(st.levels)(i => sb.newSettable[Int](s"${ name }_length_$i"))
    val missingSettable = IndexedSeq.tabulate(st.levels)(i => sb.newSettable[Long](s"${ name }_missing_$i"))
    val offsetsSettable = IndexedSeq.tabulate(st.levels)(i => sb.newSettable[Long](s"${ name }_offsets_$i"))
    val sv = new SNestedArraySettable(st, lengthsSettable, missingSettable, offsetsSettable, values.asInstanceOf)
    cb.assign(sv, this)
    sv
  }

  override def memoize(cb: EmitCodeBuilder, name: String): SNestedArraySettable = memoize(cb, name, values.memoize(cb, s"${name}_values"), cb.localBuilder)

  override def memoizeField(cb: EmitCodeBuilder, name: String): SNestedArraySettable = memoize(cb, name, values.memoizeField(cb, s"${name}_values"), cb.fieldBuilder)

  def castToArray(cb: EmitCodeBuilder): SIndexableCode = this

  def codeTuple(): IndexedSeq[Code[_]] = lengths ++ missing ++ offsets ++ values.codeTuple()
}