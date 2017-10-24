package is.hail.annotations

import is.hail.asm4s._
import is.hail.asm4s.Code._
import is.hail.expr._
import is.hail.utils._
import org.objectweb.asm.tree.{AbstractInsnNode, InsnNode}
import org.objectweb.asm.Opcodes._

import scala.collection.generic.Growable
import scala.language.implicitConversions

class StagedEncoder {

}

object StagedDecoder {

  private def storeBinary(srvb: StagedRegionValueBuilder[Decoder]): Code[Unit] = {
    val dec = srvb.input
    val off = srvb.currentOffset
    val length = srvb.extraInt
    val boff = srvb.extraLong
    Code(
      length := dec.invoke[Int]("readInt"),
      srvb.region.align(4L),
      boff := srvb.region.allocate(length.toL + 4L),
      srvb.region.storeAddress(off, boff),
      srvb.region.storeInt32(boff, length),
      dec.invoke[Long, Long, Int, Unit]("readBytes",srvb.region.mem, boff + 4L, length)
    )
  }

  private def storeType(typ: Type, srvb: StagedRegionValueBuilder[Decoder]): Code[Unit] = {
    assert(!typ.isInstanceOf[TStruct])
    assert(!typ.isInstanceOf[TArray])
    typ.fundamentalType match {
      case TBoolean => srvb.region.storeByte(srvb.currentOffset, srvb.input.invoke[Byte]("readByte"))
      case TInt32 => srvb.addInt32(srvb.input.invoke[Int]("readInt"))
      case TInt64 => srvb.addInt64(srvb.input.invoke[Long]("readLong"))
      case TFloat32 => srvb.addFloat32(srvb.input.invoke[Float]("readFloat"))
      case TFloat64 => srvb.addFloat64(srvb.input.invoke[Double]("readDouble"))
      case TBinary => storeBinary(srvb)
    }
  }

  private def storeStruct(srvb: StagedRegionValueBuilder[Decoder]): Code[Unit] = {
    val t = srvb.rowType.asInstanceOf[TStruct]
    val fb = srvb.fb

    val codeDec: LocalRef[Decoder] = srvb.input
    val region: StagedMemoryBuffer = srvb.region

    var c = Code(
      srvb.start(init = false),
      codeDec.invoke[Long, Long, Int, Unit]("readBytes", region.mem, srvb.startOffset, (t.size + 7) >>> 3)
    )
    for (i <- 0 until t.size) {
      c = Code(c,
        region.loadBit(srvb.startOffset,const(i).toL).mux(
          _empty,
          t.fieldType(i) match {
            case t2: TStruct =>
              val initF = fb.newLocal[Boolean]
              Code(
                initF.store(false),
                srvb.addStruct(t2, storeStruct)
              )
            case t2: TArray =>
              val length: LocalRef[Int] = fb.newLocal[Int]
              Code(
                length := codeDec.invoke[Int]("readInt"),
                srvb.addArray(t2, sab => storeArray(sab, length))
              )
            case _ => storeType(t.fieldType(i), srvb)
          }
        ),
        srvb.advance()
      )
    }
    c
  }

  private def storeArray(srvb: StagedRegionValueBuilder[Decoder], length: LocalRef[Int]): Code[Unit] = {

    val t = srvb.rowType.asInstanceOf[TArray]
    val fb = srvb.fb

    val codeDec: LocalRef[Decoder] = srvb.input
    val region: StagedMemoryBuffer = srvb.region

    val c = Code(
      srvb.start(length, init = false),
      region.storeInt32(srvb.startOffset, length),
      codeDec.invoke[Long, Long, Int, Unit]("readBytes", region.mem, srvb.startOffset + 4L, (length + 7) >>> 3)
    )
    val d = t.elementType match {
      case t2: TArray =>
        val l: LocalRef[Int] = srvb.fb.newLocal[Int]
        whileLoop(srvb.idx > length,
          Code(
            region.loadBit(srvb.startOffset + 4L, srvb.idx.toL).mux(
              _empty,
              Code(
                l := codeDec.invoke[Int]("readInt"),
                srvb.addArray(t2, sab => storeArray(sab, l))
              )
            ),
            srvb.advance()
          )
        )
      case t2: TStruct =>
        val initF = fb.newLocal[Boolean]
        Code(
          initF.store(false),
          whileLoop(srvb.idx < length,
            Code(
              region.loadBit(srvb.startOffset + 4L, srvb.idx.toL).mux(
                _empty,
                srvb.addStruct(t2, storeStruct)
              ),
              srvb.advance()
            )
          )
        )
      case _ =>
        whileLoop(srvb.idx < length,
          Code(
            region.loadBit(srvb.startOffset + 4L, srvb.idx.toL).mux(
              _empty,
              storeType(t.elementType, srvb)
            ),
            srvb.advance()
          )
        )
    }
    Code(c,d)
  }

  def getArrayReaderSerializable(t: TArray): () => AsmFunction2[Decoder, MemoryBuffer, Long] = {
    val srvb = new StagedRegionValueBuilder[Decoder](FunctionBuilder.functionBuilder[Decoder, MemoryBuffer, Long]("is/hail/annotations/generated"), t)
    val length: LocalRef[Int] = srvb.fb.newLocal[Int]
    srvb.emit(
      Code(
        length := srvb.input.invoke[Int]("readInt"),
        StagedDecoder.storeArray(srvb, length)
      )
    )
    srvb.build()
    srvb.transform
  }

  def getStructReaderSerializable(t: TStruct): () => AsmFunction2[Decoder, MemoryBuffer, Long] = {
    val srvb = new StagedRegionValueBuilder[Decoder](FunctionBuilder.functionBuilder[Decoder, MemoryBuffer, Long]("is/hail/annotations/generated"), t)
    val initFields = srvb.fb.newLocal[Boolean]
    srvb.emit(initFields.store(false))
    srvb.emit(StagedDecoder.storeStruct(srvb))
    srvb.build()
    srvb.transform
  }

  def getRVReaderSerializable(t: Type): () => AsmFunction2[Decoder, MemoryBuffer, Long] = {
    t.fundamentalType match {
      case t2: TArray => getArrayReaderSerializable(t2)
      case t2: TStruct => getStructReaderSerializable(t2)
    }
  }

  def getArrayReader(t: TArray): ((Decoder, MemoryBuffer) => Long) = {

    val srvb = new StagedRegionValueBuilder[Decoder](FunctionBuilder.functionBuilder[Decoder, MemoryBuffer, Long]("is/hail/annotations/generated"), t)
    val length: LocalRef[Int] = srvb.fb.newLocal[Int]

    srvb.emit(
      Code(
        length := srvb.input.invoke[Int]("readInt"),
        StagedDecoder.storeArray(srvb, length)
      )
    )
    srvb.build()
    (dec: Decoder, r: MemoryBuffer) => srvb.transform()(dec, r)
  }

  def getStructReader(t: TStruct): ((Decoder, MemoryBuffer) => Long) = {
    val srvb = new StagedRegionValueBuilder[Decoder](FunctionBuilder.functionBuilder[Decoder, MemoryBuffer, Long]("is/hail/annotations/generated"), t)
    val initFields = srvb.fb.newLocal[Boolean]
    srvb.emit(initFields.store(false))
    srvb.emit(StagedDecoder.storeStruct(srvb))
    srvb.build()
    (dec: Decoder, r: MemoryBuffer) => srvb.transform()(dec, r)
  }

  def getRegionValueReader(t: Type): ((Decoder, MemoryBuffer) => Long) = {
    t.fundamentalType match {
      case t2: TArray => getArrayReader(t2)
      case t2: TStruct => getStructReader(t2)
    }
  }

}
