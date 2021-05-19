package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s.{Code, Settable, TypeInfo, Value}
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, EmitSettable, IEmitCode}
import is.hail.types.physical.stypes.interfaces.{SBaseStruct, SBaseStructCode, SStructSettable}
import is.hail.types.physical.stypes.{EmitType, SCode, SType}
import is.hail.types.physical.{PCanonicalStruct, PType}
import is.hail.types.virtual.{TStruct, Type}

object SStackStruct {
  val MAX_FIELDS_FOR_CONSTRUCT: Int = 64

  def constructFromArgs(cb: EmitCodeBuilder, region: Value[Region], args: (String, EmitCode)*): SBaseStructCode = {
    val as = args.toArray
    if (as.length > MAX_FIELDS_FOR_CONSTRUCT) {
      PCanonicalStruct(as.map { case (name, a) => (name, a.emitType.canonicalPType) }: _*)
        .constructFromFields(cb, region, as.map(_._2), false)
    } else {
      val st = SStackStruct(as.map(_._1), as.map(_._2.emitType))
      new SStackStructCode(st, as.map(_._2))
    }
  }
}

case class SStackStruct(fieldNames: IndexedSeq[String], fieldEmitTypes: IndexedSeq[EmitType]) extends SBaseStruct {
  override def size: Int = fieldNames.size

  private lazy val codeStarts = fieldEmitTypes.map(_.nCodes).scanLeft(0)(_ + _).init
  private lazy val settableStarts = fieldEmitTypes.map(_.nSettables).scanLeft(0)(_ + _).init

  override lazy val fieldTypes: IndexedSeq[SType] = fieldEmitTypes.map(_.st)

  private lazy val _fieldIdx: Map[String, Int] = fieldNames.zipWithIndex.toMap

  override def fieldIdx(fieldName: String): Int = _fieldIdx(fieldName)

  lazy val virtualType: TStruct = TStruct(fieldNames.zip(fieldEmitTypes).map { case (f, et) => (f, et.st.virtualType) }: _*)

  override def canonicalPType(): PType = PCanonicalStruct(false, fieldNames.zip(fieldEmitTypes).map { case (f, et) => (f, et.canonicalPType) }: _*)

  lazy val codeTupleTypes: IndexedSeq[TypeInfo[_]] = fieldEmitTypes.flatMap(_.codeTupleTypes)

  override lazy val settableTupleTypes: IndexedSeq[TypeInfo[_]] = fieldEmitTypes.flatMap(_.settableTupleTypes)

  override def fromCodes(codes: IndexedSeq[Code[_]]): SBaseStructCode = {
    new SStackStructCode(this, fieldEmitTypes.indices.map { i =>
      val et = fieldEmitTypes(i)
      val start = codeStarts(i)
      et.fromCodes(codes.slice(start, start + et.nCodes))
    })
  }

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SStackStructSettable = {
    assert(settables.length == fieldEmitTypes.map(_.nSettables).sum, s"mismatch: ${ settables.length } settables, expect ${ fieldEmitTypes.map(_.nSettables).sum }\n  ${ settables.map(_.ti).mkString(",") }\n  ${ fieldEmitTypes.map(_.settableTupleTypes).mkString(" | ") }")
    new SStackStructSettable(this, fieldEmitTypes.indices.map { i =>
      val et = fieldEmitTypes(i)
      val start = settableStarts(i)
      et.fromSettables(settables.slice(start, start + et.nSettables))
    })
  }

  override def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    value match {
      case ss: SStackStructCode =>
        if (ss.st == this && !deepCopy)
          ss
        else
          new SStackStructCode(this, fieldEmitTypes.zip(ss.codes).map { case (newType, ec) =>
            EmitCode.fromI(cb.emb) { cb =>
              val iec = ec.toI(cb).map(cb) { field => newType.st.coerceOrCopy(cb, region, field, deepCopy) }
              (newType.required, iec.required) match {
                case (true, false) => IEmitCode.present(cb, iec.get(cb))
                case (false, true) => iec.setOptional
                case _ => iec
              }
            }
          })
      case _ =>
        val sv = value.asBaseStruct.memoize(cb, "stackstruct_coerce_value")
        new SStackStructCode(this, Array.tabulate[EmitCode](fieldNames.length) { i =>
          EmitCode.fromI(cb.emb) { cb =>
            val newType = fieldEmitTypes(i)
            val iec = sv.loadField(cb, i).map(cb) { field => newType.st.coerceOrCopy(cb, region, field, deepCopy) }
            (newType.required, iec.required) match {
              case (true, false) => IEmitCode.present(cb, iec.get(cb))
              case (false, true) => iec.setOptional
              case _ => iec
            }
          }
        })
    }
  }

  override def castRename(t: Type): SType = {
    val ts = t.asInstanceOf[TStruct]
    SStackStruct(
      ts.fieldNames,
      ts.types.zip(fieldEmitTypes).map { case (v, e) => e.copy(st = e.st.castRename(v)) }
    )
  }
}

class SStackStructSettable(val st: SStackStruct, settables: IndexedSeq[EmitSettable]) extends SStructSettable {
  def get: SStackStructCode = new SStackStructCode(st, settables.map(_.load))

  def settableTuple(): IndexedSeq[Settable[_]] = settables.flatMap(_.settableTuple())

  def loadField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode = {
    settables(fieldIdx).toI(cb)
  }

  def isFieldMissing(fieldIdx: Int): Code[Boolean] =
    settables(fieldIdx).m

  def store(cb: EmitCodeBuilder, pv: SCode): Unit = {
    val ssc = pv.asInstanceOf[SStackStructCode]
    settables.zip(ssc.codes).foreach { case (s, c) => s.store(cb, c) }
  }
}

class SStackStructCode(val st: SStackStruct, val codes: IndexedSeq[EmitCode]) extends SBaseStructCode {

  def code: Code[_] = throw new UnsupportedOperationException

  override def makeCodeTuple(cb: EmitCodeBuilder): IndexedSeq[Code[_]] = codes.flatMap(_.makeCodeTuple(cb))

  override def memoize(cb: EmitCodeBuilder, name: String): SStackStructSettable = {
    new SStackStructSettable(st, codes.indices.map { i =>
      val code = codes(i)
      val es = cb.emb.newEmitLocal(s"${ name }_$i", code.emitType)
      es.store(cb, code)
      es
    })
  }

  override def memoizeField(cb: EmitCodeBuilder, name: String): SStackStructSettable = {
    new SStackStructSettable(st, codes.indices.map { i =>
      val code = codes(i)
      val es = cb.emb.newEmitField(s"${ name }_$i", code.emitType)
      es.store(cb, code)
      es
    })
  }

  override def loadSingleField(cb: EmitCodeBuilder, fieldIdx: Int): IEmitCode = {
    codes(fieldIdx).toI(cb)
  }

  override def subset(fieldNames: String*): SStackStructCode = {
    val newToOld = fieldNames.map(st.fieldIdx).toArray
    new SStackStructCode(SStackStruct(newToOld.map(st.fieldNames), newToOld.map(st.fieldEmitTypes)), newToOld.map(codes))
  }
}