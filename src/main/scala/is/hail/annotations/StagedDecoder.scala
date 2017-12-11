package is.hail.annotations

import is.hail.asm4s._
import is.hail.asm4s.Code._
import is.hail.expr._
import is.hail.io.Decoder
import is.hail.utils._

import scala.language.implicitConversions

object StagedDecoder {

  private def storeType(typ: Type, srvb: StagedRegionValueBuilder): Code[Unit] = {
    assert(!typ.isInstanceOf[TStruct])
    assert(!typ.isInstanceOf[TArray])
    val dec = srvb.fb.getArg[Decoder](2)
    typ.fundamentalType match {
      case _: TBoolean => srvb.region.storeByte(srvb.currentOffset, dec.invoke[Byte]("readByte"))
      case _: TInt32 => srvb.addInt(dec.invoke[Int]("readInt"))
      case _: TInt64 => srvb.addLong(dec.invoke[Long]("readLong"))
      case _: TFloat32 => srvb.addFloat(dec.invoke[Float]("readFloat"))
      case _: TFloat64 => srvb.addDouble(dec.invoke[Double]("readDouble"))
      case _: TBinary => dec.invoke[Region, Long, Unit]("readBinary", srvb.region, srvb.currentOffset)
    }
  }

  private def storeStruct(ssb: StagedRegionValueBuilder): Code[Unit] = {
    val t = ssb.typ.asInstanceOf[TStruct]
    val dec = ssb.fb.getArg[Decoder](2)

    val region: Code[Region] = ssb.region

    var c = ssb.start(init = false)
    if (t.nMissingBytes > 0)
      c = Code(c, dec.invoke[Region, Long, Int, Unit]("readBytes", region, ssb.offset, t.nMissingBytes))

    for (i <- 0 until t.size) {
      val getNonmissingValue = t.fieldType(i) match {
        case t2: TStruct => ssb.addStruct(t2, storeStruct)
        case t2: TArray =>
          val length: LocalRef[Int] = ssb.fb.newLocal[Int]
          Code(length := dec.invoke[Int]("readInt"),
            ssb.addArray(t2, sab => storeArray(sab, length)))
        case _ => storeType(t.fieldType(i), ssb)
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
    val dec = sab.fb.getArg[Decoder](2)

    val region: Code[Region] = sab.region

    var c = Code(sab.start(length, init = false),
      region.storeInt(sab.offset, length))
    if (!t.elementType.required)
      c = Code(c, dec.invoke[Region, Long, Int, Unit]("readBytes", region, sab.offset + 4L, (length + 7) >>> 3)
      )
    val getNonmissingValue = t.elementType match {
      case t2: TArray =>
        val l: LocalRef[Int] = sab.fb.newLocal[Int]
        Code(l := dec.invoke[Int]("readInt"),
          sab.addArray(t2, sab => storeArray(sab, l)))
      case t2: TStruct => sab.addStruct(t2, storeStruct)
      case _ => storeType(t.elementType, sab)
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

  def getArrayReader(t: TArray): () => AsmFunction2[Region, Decoder, Long] = {
    val fb = FunctionBuilder.functionBuilder[Region, Decoder, Long]
    val dec = fb.getArg[Decoder](2)
    val srvb = new StagedRegionValueBuilder(fb, t)
    val length: LocalRef[Int] = srvb.fb.newLocal[Int]
    fb.emit(
      Code(
        length := dec.invoke[Int]("readInt"),
        StagedDecoder.storeArray(srvb, length),
        srvb.returnStart()))
    fb.result()
  }

  def getStructReader(t: TStruct): () => AsmFunction2[Region, Decoder, Long] = {
    val fb = FunctionBuilder.functionBuilder[Region, Decoder, Long]
    val srvb = new StagedRegionValueBuilder(fb, t)
    fb.emit(StagedDecoder.storeStruct(srvb))
    fb.emit(srvb.returnStart())
    fb.result()
  }

  def getRVReader(t: Type): () => AsmFunction2[Region, Decoder, Long] = {
    t.fundamentalType match {
      case t2: TArray => getArrayReader(t2)
      case t2: TStruct => getStructReader(t2)
    }
  }
}
