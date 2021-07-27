package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s.{Code, IntInfo, LongInfo, Settable, TypeInfo, Value, coerce}
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.PType
import is.hail.types.physical.stypes.{EmitType, SCode, SSettable, SType}
import is.hail.types.physical.stypes.interfaces.{SContainer, SIndexableCode, SIndexableValue}
import is.hail.types.virtual.{TArray, TBaseStruct, Type}
import is.hail.utils.FastIndexedSeq

case class SStructOfArrays(virtualType: TArray, elementsRequired: Boolean, fields: IndexedSeq[SContainer]) extends SContainer {
  require(virtualType.elementType.isInstanceOf[TBaseStruct])
  require(fields.forall(st => st.virtualType.isInstanceOf[TArray]))

  private val structVirtualType: TBaseStruct = virtualType.elementType.asInstanceOf

  override val elementType: SStackStruct = SStackStruct(structVirtualType, fields.map(_.elementEmitType))

  val elementEmitType: EmitType = EmitType(elementType, elementsRequired)

  protected[stypes] def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = ???

  private lazy val codeStarts = fields.map(_.nCodes).scanLeft(baseTupleTypes.length)(_ + _).init
  private lazy val settableStarts = fields.map(_.nSettables).scanLeft(baseTupleTypes.length)(_ + _).init
  private lazy val baseTupleTypes: IndexedSeq[TypeInfo[_]] =
    IndexedSeq(IntInfo) ++ (if (elementsRequired) None else Some(LongInfo))

  lazy val codeTupleTypes: IndexedSeq[TypeInfo[_]] = baseTupleTypes ++ fields.flatMap(_.codeTupleTypes())

  override lazy val settableTupleTypes: IndexedSeq[TypeInfo[_]] = baseTupleTypes ++ fields.flatMap(_.settableTupleTypes())

  def fromCodes(codes: IndexedSeq[Code[_]]): SCode = {
    new SStructOfArraysCode(this,
      coerce(codes(0)),
      if (elementsRequired) None else Some(coerce(codes(1))),
      fields.zipWithIndex.map { case (f, i) =>
        val start = codeStarts(i)
        f.fromCodes(codes.slice(start, start + f.nCodes)).asIndexable
      }
    )
  }

  def fromSettables(settables: IndexedSeq[Settable[_]]): SSettable = {
    new SStructOfArraysSettable(this,
      coerce(settables(0)),
      if (elementsRequired) None else Some(coerce(settables(1))),
      fields.zipWithIndex.map { case (f, i) =>
        val start = settableStarts(i)
        f.fromSettables(settables.slice(start, start + f.nSettables)).asInstanceOf
      }
    )
  }


  def canonicalPType(): PType = PType.canonical(virtualType)

  def castRename(t: Type): SType = {
    val arrayType = t.asInstanceOf[TArray]
    val structType = arrayType.elementType.asInstanceOf[TBaseStruct]

    SStructOfArrays(
      arrayType,
      elementsRequired,
      structType.types.zip(fields).map { case (v, f) => f.castRename(TArray(v)).asInstanceOf }
    )
  }
}

class SStructOfArraysSettable(
  val st: SStructOfArrays,
  val length: Settable[Int],
  val missing: Option[Settable[Long]],
  val fields: IndexedSeq[SIndexableValue with SSettable]
) extends SIndexableValue with SSettable {
  def loadLength(): Value[Int] = length

  def isElementMissing(i: Code[Int]): Code[Boolean] = missing.map(m => Region.loadBit(m, i.toL)).getOrElse(false)

  def loadElement(cb: EmitCodeBuilder, i: Code[Int]): IEmitCode = {
    val iv = cb.newLocal("load_element_i", i)
    IEmitCode(cb, isElementMissing(iv),
      new SStackStructCode(st.elementType, fields.map { fv =>
        EmitCode.fromI(cb.emb)(cb => fv.loadElement(cb, iv))
      }
    ))
  }

  def hasMissingValues(cb: EmitCodeBuilder): Code[Boolean] =
    missing.map(m => Region.containsNonZeroBits(m, loadLength().toL)).getOrElse(false)

  def store(cb: EmitCodeBuilder, v: SCode): Unit = {
    val vs: SStructOfArraysCode = v.asInstanceOf
    require(st == vs.st)
    cb.assign(length, vs.length)
    missing.foreach(m => cb.assign(m, vs.missing.get))
    fields.zip(vs.fields).foreach { case (fs, fc) =>
      cb.assign(fs, fc)
    }
  }

  def settableTuple(): IndexedSeq[Settable[_]] =
    FastIndexedSeq(length) ++ missing ++ fields.flatMap(_.settableTuple())

  def get: SCode = new SStructOfArraysCode(st, length, missing.map(_.get), fields.map(_.get.asIndexable))
}

class SStructOfArraysCode(
  val st: SStructOfArrays,
  val length: Code[Int],
  val missing: Option[Code[Long]],
  val fields: IndexedSeq[SIndexableCode]
) extends SIndexableCode {
  require(st.elementsRequired == missing.isEmpty)

  def codeLoadLength(): Code[Int] = length

  def memoize(cb: EmitCodeBuilder, name: String): SIndexableValue = {
    val ssas = new SStructOfArraysSettable(
      st,
      cb.newLocal(s"${ name }_length"),
      missing.map(_ => cb.newLocal(s"${ name }_missing")),
      fields.zipWithIndex.map { case (f, i) => cb.emb.newPLocal(s"${ name }_$i", f.st).asInstanceOf }
    )
    ssas.store(cb, this)
    ssas
  }

  def memoizeField(cb: EmitCodeBuilder, name: String): SIndexableValue = {
    val ssas = new SStructOfArraysSettable(
      st,
      cb.newField(s"${ name }_length"),
      missing.map(_ => cb.newField(s"${ name }_missing")),
      fields.zipWithIndex.map { case (f, i) => cb.emb.newPField(s"${ name }_$i", f.st).asInstanceOf }
    )
    ssas.store(cb, this)
    ssas
  }

  def castToArray(cb: EmitCodeBuilder): SIndexableCode = this

  def makeCodeTuple(cb: EmitCodeBuilder): IndexedSeq[Code[_]] =
    FastIndexedSeq(length) ++ missing ++ fields.flatMap(_.makeCodeTuple(cb))
}