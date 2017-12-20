package is.hail.annotations

import is.hail.asm4s.{Code, _}
import is.hail.asm4s.Code._
import is.hail.expr._
import is.hail.io.Decoder
import is.hail.utils._
import org.objectweb.asm.Opcodes.IADD
import org.objectweb.asm.tree.{AbstractInsnNode, InsnNode}

import scala.collection.generic.Growable
import scala.language.implicitConversions

object StagedDecoder {

  private def storeNonNested(typ: Type, srvb: StagedRegionValueBuilder): Code[Unit] = {
    assert(!typ.fundamentalType.isInstanceOf[TStruct])
    assert(!typ.fundamentalType.isInstanceOf[TArray])

    val dec: Code[Decoder] = srvb.fb.getArg[Decoder](2)

    typ.fundamentalType match {
      case _: TBinary => srvb.addAddress(dec.readBinary(srvb.region))
      case _ => srvb.addIRIntermediate(typ)(dec.readPrimitive(typ))
    }
  }

  private def storeStruct(ssb: StagedRegionValueBuilder): Code[Unit] = {
    val t = ssb.typ.asInstanceOf[TStruct]
    val dec: Code[Decoder] = ssb.fb.getArg[Decoder](2)

    val region: Code[Region] = ssb.region

    var c = ssb.start(init = false)
    if (t.nMissingBytes > 0)
      c = Code(c, dec.readBytes(region, ssb.offset, t.nMissingBytes))

    for (i <- 0 until t.size) {
      val getNonmissingValue = t.fieldType(i).fundamentalType match {
        case t2: TStruct => ssb.addStruct(t2, storeStruct)
        case t2: TArray =>
          val length: LocalRef[Int] = ssb.fb.newLocal[Int]
          Code(length := dec.readInt(),
            ssb.addArray(t2, sab => storeArray(sab, length)))
        case t2 => storeNonNested(t2, ssb)
      }

      if (t.isFieldRequired(i))
        c = Code(c, getNonmissingValue, ssb.advance())
      else
        c = Code(c,
          t.isFieldDefined(region, ssb.offset, i).mux(getNonmissingValue, _empty),
          ssb.advance())
    }
    c
  }

  private def storeArray(sab: StagedRegionValueBuilder, length: LocalRef[Int]): Code[Unit] = {
    val t = sab.typ.asInstanceOf[TArray]
    val dec: Code[Decoder] = sab.fb.getArg[Decoder](2)

    val region: Code[Region] = sab.region

    var c = Code(sab.start(length, init = false),
      region.storeInt(sab.offset, length))
    if (!t.elementType.required)
      c = Code(c, dec.readBytes(region, sab.offset + 4L, (length + 7) >>> 3))
    val getNonmissingValue = t.elementType.fundamentalType match {
      case t2: TArray =>
        val l: LocalRef[Int] = sab.fb.newLocal[Int]
        Code(l := dec.readInt(),
          sab.addArray(t2, sab => storeArray(sab, l)))
      case t2: TStruct => sab.addStruct(t2, storeStruct)
      case _ => storeNonNested(t.elementType, sab)
    }
    if (t.elementType.required)
      c = Code(c, whileLoop(sab.arrayIdx < length, getNonmissingValue, sab.advance()))
    else
      c = Code(c,
        whileLoop(sab.arrayIdx < length,
          t.isElementDefined(region, sab.offset, sab.arrayIdx).mux(getNonmissingValue, _empty),
          sab.advance()))
    c
  }

  def getRVReader(t: Type): () => AsmFunction2[Region, Decoder, Long] = {
    val fb = FunctionBuilder.functionBuilder[Region, Decoder, Long]
    val srvb = new StagedRegionValueBuilder(fb, t)
    t.fundamentalType match {
      case t2: TArray =>
        val dec: Code[Decoder] = fb.getArg[Decoder](2)
        val length: LocalRef[Int] = srvb.fb.newLocal[Int]
        fb.emit(length := dec.readInt())
        fb.emit(StagedDecoder.storeArray(srvb, length))
      case t2: TStruct => fb.emit(StagedDecoder.storeStruct(srvb))
      case t2: TBinary =>
        val dec: Code[Decoder] = fb.getArg[Decoder](2)
        fb.emit(Code(srvb.start(), dec.readBinary(srvb.region)))
      case t2 => fb.emit(Code(srvb.start(), StagedDecoder.storeNonNested(t, srvb)))
    }
    fb.emit(srvb.end())
    fb.result()
  }
}
