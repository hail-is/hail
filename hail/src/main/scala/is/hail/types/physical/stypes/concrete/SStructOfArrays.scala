package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.{PCanonicalArray, PInt32Required, PType}
import is.hail.types.physical.stypes.{EmitType, SCode, SSettable, SType}
import is.hail.types.physical.stypes.interfaces.{SBaseStruct, SContainer, SIndexableCode, SIndexableSettable, SIndexableValue, primitive}
import is.hail.types.virtual.{Field, TArray, TBaseStruct, TContainer, Type}
import is.hail.utils.FastIndexedSeq
import SStructOfArrays._

case class SStructOfArrays(virtualType: TContainer, elementsRequired: Boolean, fields: IndexedSeq[SContainer]) extends SContainer {
  require(virtualType.elementType.isInstanceOf[TBaseStruct])
  require(fields.forall(st => st.virtualType.isInstanceOf[TArray]))

  private[concrete] val structVirtualType: TBaseStruct = virtualType.elementType.asInstanceOf

  override val elementType: SStackStruct = SStackStruct(structVirtualType, fields.map(_.elementEmitType))

  val elementEmitType: EmitType = EmitType(elementType, elementsRequired)

  protected[stypes] def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = value match {
    case v: SStructOfArraysCode =>
      new SStructOfArraysCode(this,
        v.lookupOrLength match {
          case Right(lookup) => Right(lookup.st.coerceOrCopy(cb, region, lookup, deepCopy).asInstanceOf[SIndexablePointerCode])
          case x => x
        },
        fields.zip(v.fields).map { case (field, code) =>
          field.coerceOrCopy(cb, region, code, deepCopy).asIndexable
        })
    case vc: SIndexablePointerCode if vc.st.elementType.isInstanceOf[SBaseStruct] =>
      assert(vc.st.virtualType == virtualType)
      val v = vc.memoize(cb, "convert_to_structofarrays_arr")
      val innerLength = cb.newLocal[Int]("inner_length", v.loadLength() - v.numberMissingValues(cb))
      val lookupConstructor = if (elementsRequired) None else Some({
        val (push, finish) = LOOKUP_TYPE.constructFromFunctions(cb, region, v.loadLength(), deepCopy)
        (push, cb.newLocal[Int]("lookup_value"), cb.newLocal[Int]("next_lookup", 0), finish)
      })
      val fieldConstructors = fields.map(f => f.constructFromFunctions(cb, region, innerLength, deepCopy))
      val i = cb.newLocal[Int]("i")
      cb.forLoop(cb.assign(i, 0), i < v.loadLength(), cb.assign(i, i + 1), {
        v.loadElement(cb, i).consume(cb, {
          lookupConstructor.foreach(c => cb.assign(c._2, MISSING_SENTINEL))
        }, { sc =>
          lookupConstructor.foreach { case (_, lookupValue, next, _) =>
            cb.assign(lookupValue, next)
            cb.assign(next, next + 1)
          }
          val structValue = sc.asBaseStruct.memoize(cb, "struct_value")
          fieldConstructors.indices.foreach { idx =>
            val push = fieldConstructors(idx)._1
            val code = structValue.loadField(cb, idx)
            push(cb, code)
          }
        })
        lookupConstructor.foreach { case (push, lookupElement, _, _) =>
          push(cb, IEmitCode.present(cb, primitive(lookupElement)))
        }
      })
      val lookupOrLength = lookupConstructor match {
        case None => Left(v.loadLength().get)
        case Some((_, _, _, finish)) => Right(finish(cb))
      }
      val fieldCodes = fieldConstructors.map { case (_, finish) => finish(cb) }
      new SStructOfArraysCode(this, lookupOrLength, fieldCodes)
  }

  private lazy val codeStarts = fields.map(_.nCodes).scanLeft(baseCodeTupleTypes.length)(_ + _).init
  private lazy val settableStarts = fields.map(_.nSettables).scanLeft(baseSettableTupleTypes.length)(_ + _).init
  private lazy val baseCodeTupleTypes: IndexedSeq[TypeInfo[_]] = if (elementsRequired)
    IndexedSeq(IntInfo)
  else
    LOOKUP_TYPE.codeTupleTypes()
  private lazy val baseSettableTupleTypes: IndexedSeq[TypeInfo[_]] = if (elementsRequired)
    IndexedSeq(IntInfo)
  else
    LOOKUP_TYPE.settableTupleTypes()

  lazy val codeTupleTypes: IndexedSeq[TypeInfo[_]] = baseCodeTupleTypes ++ fields.flatMap(_.codeTupleTypes())
  override lazy val settableTupleTypes: IndexedSeq[TypeInfo[_]] = baseSettableTupleTypes ++ fields.flatMap(_.settableTupleTypes())

  def fromCodes(codes: IndexedSeq[Code[_]]): SCode = {
    new SStructOfArraysCode(this,
      if (elementsRequired)
        Left(coerce(codes(0)))
      else
        Right(LOOKUP_TYPE.fromCodes(codes.slice(0, LOOKUP_TYPE.nCodes))),
      fields.zipWithIndex.map { case (f, i) =>
        val start = codeStarts(i)
        f.fromCodes(codes.slice(start, start + f.nCodes)).asIndexable
      }
    )
  }

  def fromSettables(settables: IndexedSeq[Settable[_]]): SSettable = {
    new SStructOfArraysSettable(this,
      if (elementsRequired)
        Left(coerce(settables(0)))
      else
        Right(LOOKUP_TYPE.fromSettables(settables.slice(0, LOOKUP_TYPE.nSettables))),
      fields.zipWithIndex.map { case (f, i) =>
        val start = settableStarts(i)
        f.fromSettables(settables.slice(start, start + f.nSettables)).asInstanceOf
      }
    )
  }

  def storageType(): PType = ???

  def copiedType: SType = SStructOfArrays(virtualType, elementsRequired, fields.map(_.copiedType.asInstanceOf[SContainer]))

  def containsPointers: Boolean = (!elementsRequired && LOOKUP_TYPE.containsPointers) || fields.exists(_.containsPointers)

  def constructFromFunctions(cb: EmitCodeBuilder, region: Value[Region], length: Value[Int], deepCopy: Boolean): ((EmitCodeBuilder, IEmitCode) => Unit, EmitCodeBuilder => SStructOfArraysCode) = ???

  def castRename(t: Type): SType = {
    val arrayType = t.asInstanceOf[TContainer]
    val structType = arrayType.elementType.asInstanceOf[TBaseStruct]

    SStructOfArrays(
      arrayType,
      elementsRequired,
      structType.types.zip(fields).map { case (v, f) => f.castRename(TArray(v)).asInstanceOf }
    )
  }
}

object SStructOfArrays {
  val MISSING_SENTINEL: Value[Int] = const(-1)
  val LOOKUP_TYPE: SIndexablePointer = SIndexablePointer(PCanonicalArray(PInt32Required))

  def fromIndexablePointer(sip: SIndexablePointer): SStructOfArrays = {
    // TODO smarter choices for the field SContainers
    val fields = sip.elementType.asInstanceOf[SBaseStruct].fieldEmitTypes.map { case EmitType(st, required) =>
      SIndexablePointer(PCanonicalArray(st.storageType().setRequired(required)))
    }
    SStructOfArrays(sip.virtualType, sip.elementEmitType.required, fields)
  }
}

object SStructOfArraysSettable {
  def apply(sb: SettableBuilder, st: SStructOfArrays, name: String): SStructOfArraysSettable = {
    new SStructOfArraysSettable(st,
      if (st.elementsRequired)
        Left(sb.newSettable[Int](s"${name}_length"))
      else
        Right(SIndexablePointerSettable(sb, LOOKUP_TYPE, s"${name}_lookup")),
      st.fields.zip(st.structVirtualType.fields).map { case (st, Field(fieldName, _, _)) =>
        SSettable(sb, st, s"${name}_$fieldName").asInstanceOf
      }
    )
  }
}

class SStructOfArraysSettable(
  val st: SStructOfArrays,
  val lookupOrLength: Either[Settable[Int], SIndexablePointerSettable],
  val fields: IndexedSeq[SIndexableSettable]
) extends SIndexableValue with SSettable {
  lookupOrLength match {
    case Right(lookup) => require(lookup.st == LOOKUP_TYPE)
    case Left(_) => require(st.elementsRequired)
  }

  def loadLength(): Value[Int] = lookupOrLength match {
    case Left(length) => length
    case Right(lookup) => lookup.loadLength()
  }

  private def lookupIndex(cb: EmitCodeBuilder, i: Code[Int]): Code[Int] = lookupOrLength match {
    case Left(_) => i
    case Right(lookup) =>
      lookup.loadElement(cb, i).get(cb, "required cannot be missing").asInt32.intCode(cb)
  }

  def isElementMissing(cb: EmitCodeBuilder, i: Code[Int]): Code[Boolean] = lookupOrLength match {
    case Left(_) => const(false)
    case Right(_) => lookupIndex(cb, i).ceq(MISSING_SENTINEL)
  }

  def loadElement(cb: EmitCodeBuilder, i: Code[Int]): IEmitCode = {
    val iv = cb.newLocal("load_element_real_idx", lookupIndex(cb, i))
    IEmitCode(cb, iv.ceq(MISSING_SENTINEL),
      new SStackStructCode(st.elementType, fields.map { fv =>
        EmitCode.fromI(cb.emb)(cb => fv.loadElement(cb, iv))
      }
    ))
  }

  def hasMissingValues(cb: EmitCodeBuilder): Code[Boolean] = lookupOrLength match {
    case Left(_) => const(false)
    case Right(lookup) =>
      val hasMissing = cb.newLocal("hasMissing", false)
      val out = CodeLabel()
      lookup.forEachDefined(cb) { (cb, _, idxCode) =>
        cb.assign(hasMissing, idxCode.asInt32.intCode(cb).ceq(MISSING_SENTINEL))
        cb.ifx(hasMissing, cb.goto(out))
      }
      cb.define(out)
      hasMissing
  }

  def store(cb: EmitCodeBuilder, v: SCode): Unit = {
    val vs: SStructOfArraysCode = v.asInstanceOf
    require(st == vs.st)
    (lookupOrLength, vs.lookupOrLength) match {
      case (Left(lookupValue), Left(lookupCode)) => cb.assign(lookupValue, lookupCode)
      case (Right(lengthValue), Right(lengthCode)) => cb.assign(lengthValue, lengthCode)
    }
    fields.zip(vs.fields).foreach { case (fs, fc) =>
      cb.assign(fs, fc)
    }
  }

  def settableTuple(): IndexedSeq[Settable[_]] =
    lookupOrLength.fold(FastIndexedSeq(_), _.settableTuple()) ++ fields.flatMap(_.settableTuple())

  private def getLookupOrLength: Either[Code[Int], SIndexablePointerCode] = lookupOrLength match {
    case Right(lookup) => Right(lookup.get)
    case Left(length) => Left(length)
  }

  def get: SCode = new SStructOfArraysCode(st, getLookupOrLength, fields.map(_.get.asIndexable))
}

class SStructOfArraysCode(
  val st: SStructOfArrays,
  val lookupOrLength: Either[Code[Int], SIndexablePointerCode],
  val fields: IndexedSeq[SIndexableCode]
) extends SIndexableCode {
  lookupOrLength match {
    case Left(_) => require(st.elementsRequired)
    case Right(lookup) => require(lookup.st == LOOKUP_TYPE)
  }

  def codeLoadLength(): Code[Int] = lookupOrLength match {
    case Left(length) => length
    case Right(lookup) => lookup.codeLoadLength()
  }

  def memoize(cb: EmitCodeBuilder, name: String): SIndexableValue = {
    val ssas = SStructOfArraysSettable(cb.emb.localBuilder, st, name)
    ssas.store(cb, this)
    ssas
  }

  def memoizeField(cb: EmitCodeBuilder, name: String): SIndexableValue = {
    val ssas = SStructOfArraysSettable(cb.emb.fieldBuilder, st, name)
    ssas.store(cb, this)
    ssas
  }

  def castToArray(cb: EmitCodeBuilder): SIndexableCode = this

  def makeCodeTuple(cb: EmitCodeBuilder): IndexedSeq[Code[_]] =
    lookupOrLength.fold(FastIndexedSeq(_), _.makeCodeTuple(cb)) ++ fields.flatMap(_.makeCodeTuple(cb))
}
