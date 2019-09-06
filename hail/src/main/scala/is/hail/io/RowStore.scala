package is.hail.io

import java.io._

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir
import is.hail.expr.ir.{EmitFunctionBuilder, EmitUtils, EstimableEmitter, MethodBuilderSelfLike, PruneDeadFields}
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.nativecode._
import is.hail.utils._
import is.hail.{HailContext, cxx}

case class PackCodecSpec(child: BufferSpec) extends CodecSpec {
  def makeCodecSpec2(pType: PType) = PackCodecSpec2(pType, child)
}

final case class PackCodecSpec2(eType: PType, child: BufferSpec) extends CodecSpec2 {
  def encodedType: Type = eType.virtualType

  def computeSubsetPType(requestedType: Type): PType = {
    assert(PruneDeadFields.isSupertype(requestedType, eType.virtualType))
    PType.canonical(requestedType)
  }

  def buildEncoder(t: PType, requestedType: PType): (OutputStream) => Encoder = {
    if (HailContext.isInitialized && HailContext.get.flags.get("cpp") != null && requestedType == t) {
      val e: NativeEncoderModule = cxx.PackEncoder.buildModule(t, child)
      (out: OutputStream) => new NativePackEncoder(out, e)
    } else {
      val f = EmitPackEncoder(t, requestedType)
      out: OutputStream => new CompiledEncoder(child.buildOutputBuffer(out), f)
    }
  }

  def buildDecoder(requestedType: Type): (PType, (InputStream) => Decoder) = {
    val rt = computeSubsetPType(requestedType)
    if (HailContext.isInitialized && HailContext.get.flags != null && HailContext.get.flags.get("cpp") != null) {
      val d: NativeDecoderModule = cxx.PackDecoder.buildModule(eType, rt, child)
      (rt, (in: InputStream) => new NativePackDecoder(in, d))
    } else {
      val f = EmitPackDecoder(eType, rt)
      (rt, (in: InputStream) => new CompiledDecoder(child.buildInputBuffer(in), f))
    }
  }

  def buildCodeInputBuffer(is: Code[InputStream]): Code[InputBuffer] = child.buildCodeInputBuffer(is)

  def buildCodeOutputBuffer(os: Code[OutputStream]): Code[OutputBuffer] = child.buildCodeOutputBuffer(os)

  def buildEmitDecoderF[T](requestedType: Type, fb: EmitFunctionBuilder[_]): (PType, StagedDecoderF[T]) = {
    val rt = computeSubsetPType(requestedType)
    val mb = EmitPackDecoder.buildMethod(eType, rt, fb)
    (rt, (region: Code[Region], buf: Code[InputBuffer]) => mb.invoke[T](region, buf))
  }

  def buildEmitEncoderF[T](t: PType, fb: EmitFunctionBuilder[_]): StagedEncoderF[T] = {
    val mb = EmitPackEncoder.buildMethod(t, eType, fb)
    (region: Code[Region], off: Code[T], buf: Code[OutputBuffer]) => mb.invoke[Unit](region, off, buf)
  }

  def buildNativeDecoderClass(
    requestedType: Type,
    inputStreamType: String,
    tub: cxx.TranslationUnitBuilder
  ): (PType, cxx.Class) = {
    val rt = computeSubsetPType(requestedType)
    (rt, cxx.PackDecoder(eType, rt, inputStreamType, child, tub))
  }

  def buildNativeEncoderClass(t: PType, tub: cxx.TranslationUnitBuilder): cxx.Class = cxx.PackEncoder(t, child, tub)
}

object ShowBuf {
  def apply(buf: Array[Byte], pos: Int, n: Int): Unit = {
    val sb = new StringBuilder()
    val len = if (n < 32) n else 32
    var j = 0
    while (j < len) {
      val x = (buf(pos + j).toInt & 0xff)
      if (x <= 0xf) sb.append(s" 0${ x.toHexString }") else sb.append(s" ${ x.toHexString }")
      if ((j & 0x7) == 0x7) sb.append("\n")
      j += 1
    }
    System.err.println(sb.toString())
  }

  def apply(addr: Long, n: Int): Unit = {
    val sb = new StringBuilder()
    val len = if (n < 32) n else 32
    var j = 0
    while (j < len) {
      val x = (Memory.loadByte(addr + j).toInt & 0xff)
      if (x <= 0xf) sb.append(s" 0${ x.toHexString }") else sb.append(s" ${ x.toHexString }")
      if ((j & 0x7) == 0x7) sb.append("\n")
      j += 1
    }
    System.err.println(sb.toString())
  }
}

object EmitPackDecoder {
  self =>

  type Emitter = EstimableEmitter[MethodBuilderSelfLike]

  def emitTypeSize(t: PType): Int = {
    t match {
      case t: PArray => 120 + emitTypeSize(t.elementType)
      case t: PStruct => 100
      case _ => 20
    }
  }

  def emitBinary(
    t: PBinary,
    mb: MethodBuilder,
    in: Code[InputBuffer],
    srvb: StagedRegionValueBuilder): Code[Unit] = {
    val length = mb.newLocal[Int]
    val boff = mb.newLocal[Long]

    Code(
      length := in.readInt(),
      boff := srvb.allocateBinary(length),
      in.readBytes(srvb.region, boff + const(4), length))
  }

  def emitBaseStruct(
    t: PBaseStruct,
    requestedType: PBaseStruct,
    mb: MethodBuilder,
    in: Code[InputBuffer],
    srvb: StagedRegionValueBuilder): Code[Unit] = {
    val region = srvb.region

    val moff = mb.newField[Long]

    val initCode = Code(
      srvb.start(init = true),
      moff := region.allocate(const(1), const(t.nMissingBytes)),
      in.readBytes(region, moff, t.nMissingBytes))
    val fieldEmitters = new Array[Emitter](t.size)

    assert(t.isInstanceOf[PTuple] || t.isInstanceOf[PStruct])

    var i = 0
    var j = 0
    while (i < t.size) {
      val f = t.fields(i)
      fieldEmitters(i) =
        if (t.isInstanceOf[PTuple] ||
          (j < requestedType.size && requestedType.fields(j).name == f.name)) {
          val rf = requestedType.fields(j)
          assert(f.typ.required == rf.typ.required)
          j += 1

          new Emitter {
            def emit(mbLike: MethodBuilderSelfLike): Code[Unit] = {
              val readElement = self.emit(f.typ, rf.typ, mbLike.mb, in, srvb)
              Code(
                if (f.typ.required)
                  readElement
                else {
                  region.loadBit(moff, const(t.missingIdx(f.index))).mux(
                    srvb.setMissing(),
                    readElement)
                },
                srvb.advance())
            }

            def estimatedSize: Int = emitTypeSize(f.typ)
          }
        } else {
          new Emitter {
            def emit(mbLike: MethodBuilderSelfLike): Code[Unit] = {
              val skipField = skip(f.typ, mbLike.mb, in, region)
              if (f.typ.required)
                skipField
              else {
                region.loadBit(moff, const(t.missingIdx(f.index))).mux(
                  Code._empty,
                  skipField)
              }
            }

            def estimatedSize: Int = emitTypeSize(f.typ)
          }
        }
      i += 1
    }
    assert(j == requestedType.size)

    Code(initCode,
      EmitUtils.wrapToMethod(fieldEmitters, new MethodBuilderSelfLike(mb)))
  }

  def emitArray(
    t: PArray,
    requestedType: PArray,
    mb: MethodBuilder,
    in: Code[InputBuffer],
    srvb: StagedRegionValueBuilder): Code[Unit] = {
    val length = mb.newLocal[Int]
    val i = mb.newLocal[Int]
    val aoff = mb.newLocal[Long]

    Code(
      length := in.readInt(),
      srvb.start(length, init = false),
      aoff := srvb.offset,
      srvb.region.storeInt(aoff, length),
      if (t.elementType.required)
        Code._empty
      else
        in.readBytes(srvb.region, aoff + const(4), (length + 7) >>> 3),
      i := 0,
      Code.whileLoop(
        i < length,
        Code({
          val readElement = emit(t.elementType, requestedType.elementType, mb, in, srvb)
          if (t.elementType.required)
            readElement
          else
            t.isElementDefined(srvb.region, aoff, i).mux(
              readElement,
              srvb.setMissing())
        },
          srvb.advance(),
          i := i + const(1))))
  }

  def skipBaseStruct(t: PBaseStruct, mb: MethodBuilder, in: Code[InputBuffer], region: Code[Region]): Code[Unit] = {
    val moff = mb.newField[Long]

    val fieldEmitters = t.fields.map { f =>
      new Emitter {
        def emit(mbLike: MethodBuilderSelfLike): Code[Unit] = {
          val skipField = skip(f.typ, mbLike.mb, in, region)
          if (f.typ.required)
            skipField
          else
            region.loadBit(moff, const(t.missingIdx(f.index))).mux(
              Code._empty,
              skipField)
        }

        def estimatedSize: Int = emitTypeSize(f.typ)
      }
    }

    Code(
      moff := region.allocate(const(1), const(t.nMissingBytes)),
      in.readBytes(region, moff, t.nMissingBytes),
      EmitUtils.wrapToMethod(fieldEmitters, new MethodBuilderSelfLike(mb)))
  }

  def skipArray(t: PArray,
    mb: MethodBuilder,
    in: Code[InputBuffer],
    region: Code[Region]): Code[Unit] = {
    val length = mb.newLocal[Int]
    val i = mb.newLocal[Int]

    if (t.elementType.required) {
      Code(
        length := in.readInt(),
        i := 0,
        Code.whileLoop(i < length,
          Code(
            skip(t.elementType, mb, in, region),
            i := i + const(1))))
    } else {
      val moff = mb.newLocal[Long]
      val nMissing = mb.newLocal[Int]
      Code(
        length := in.readInt(),
        nMissing := ((length + 7) >>> 3),
        moff := region.allocate(const(1), nMissing.toL),
        in.readBytes(region, moff, nMissing),
        i := 0,
        Code.whileLoop(i < length,
          region.loadBit(moff, i.toL).mux(
            Code._empty,
            skip(t.elementType, mb, in, region)),
          i := i + const(1)))
    }
  }

  def skipBinary(t: PType, mb: MethodBuilder, in: Code[InputBuffer]): Code[Unit] = {
    val length = mb.newLocal[Int]
    Code(
      length := in.readInt(),
      in.skipBytes(length))
  }

  def skip(t: PType, mb: MethodBuilder, in: Code[InputBuffer], region: Code[Region]): Code[Unit] = {
    t match {
      case t2: PBaseStruct =>
        skipBaseStruct(t2, mb, in, region)
      case t2: PArray =>
        skipArray(t2, mb, in, region)
      case _: PBoolean => in.skipBoolean()
      case _: PInt64 => in.skipLong()
      case _: PInt32 => in.skipInt()
      case _: PFloat32 => in.skipFloat()
      case _: PFloat64 => in.skipDouble()
      case t2: PBinary => skipBinary(t2, mb, in)
    }
  }

  def emit(
    t: PType,
    requestedType: PType,
    mb: MethodBuilder,
    in: Code[InputBuffer],
    srvb: StagedRegionValueBuilder): Code[Unit] = {
    t match {
      case t2: PBaseStruct =>
        val requestedType2 = requestedType.asInstanceOf[PBaseStruct]
        srvb.addBaseStruct(requestedType2, { srvb2 =>
          emitBaseStruct(t2, requestedType2, mb, in, srvb2)
        })
      case t2: PArray =>
        val requestedType2 = requestedType.asInstanceOf[PArray]
        srvb.addArray(requestedType2, { srvb2 =>
          emitArray(t2, requestedType2, mb, in, srvb2)
        })
      case _: PBoolean => srvb.addBoolean(in.readBoolean())
      case _: PInt64 => srvb.addLong(in.readLong())
      case _: PInt32 => srvb.addInt(in.readInt())
      case _: PFloat32 => srvb.addFloat(in.readFloat())
      case _: PFloat64 => srvb.addDouble(in.readDouble())
      case t2: PBinary => emitBinary(t2, mb, in, srvb)
    }
  }

  def decode(t: PType, rt: PType, mb: MethodBuilder): Code[_] = {
    val in = mb.getArg[InputBuffer](2)
    t.fundamentalType match {
      case _: PBoolean => in.load().readBoolean()
      case _: PInt32 => in.load().readInt()
      case _: PInt64 => in.load().readLong()
      case _: PFloat32 => in.load().readFloat()
      case _: PFloat64 => in.load().readDouble()
      case _ =>
        val srvb = new StagedRegionValueBuilder(mb, rt)
        val emit = t.fundamentalType match {
          case t2: PBinary => emitBinary(t2, mb, in, srvb)
          case t2: PBaseStruct => emitBaseStruct(t2, rt.fundamentalType.asInstanceOf[PBaseStruct], mb, in, srvb)
          case t2: PArray => emitArray(t2, rt.fundamentalType.asInstanceOf[PArray], mb, in, srvb)
        }
        Code(emit, srvb.end())
    }
  }

  def buildMethod(t: PType, rt: PType, fb: EmitFunctionBuilder[_]): ir.EmitMethodBuilder = {
    val mb = fb.newMethod(Array[TypeInfo[_]](typeInfo[Region], typeInfo[InputBuffer]), ir.typeToTypeInfo(rt))
    mb.emit(decode(t, rt, mb))
    mb
  }

  def apply(t: PType, requestedType: PType): () => AsmFunction2[Region, InputBuffer, Long] = {
    val fb = new Function2Builder[Region, InputBuffer, Long]
    val mb = fb.apply_method

    if (t.isPrimitive) {
      val srvb = new StagedRegionValueBuilder(mb, requestedType)
      mb.emit(Code(
        srvb.start(),
        srvb.addIRIntermediate(requestedType)(decode(t, requestedType, mb)),
        srvb.end()))
    } else {
      mb.emit(decode(t, requestedType, mb))
    }

    fb.result()
  }
}

case class NativeDecoderModule(
  modKey: String,
  modBinary: Array[Byte]) extends Serializable

final class NativePackDecoder(in: InputStream, module: NativeDecoderModule) extends Decoder {
  private[this] val st = new NativeStatus()
  private[this] val mod = new NativeModule(module.modKey, module.modBinary)
  private[this] val make_decoder = mod.findPtrFuncL1(st, "make_input_buffer")
  assert(st.ok, st.toString())
  private[this] val decode_row = mod.findLongFuncL2(st, "decode_row")
  assert(st.ok, st.toString())
  private[this] val decode_byte = mod.findLongFuncL1(st, "decode_byte")
  assert(st.ok, st.toString())
  private[this] val input = new ObjectArray(in)
  private[this] val decoder = new NativePtr(make_decoder, st, input.get())
  input.close()
  make_decoder.close()
  assert(st.ok, st.toString())

  def close(): Unit = {
    decoder.close()
    decode_row.close()
    decode_byte.close()
    // NativePtr's to objects with destructors using the module code must
    // *not* be close'd last, since the module will be dlclose'd before the
    // destructor is called.  One safe policy is to close everything in
    // reverse order, ending with the NativeModule
    mod.close()
    st.close()
    in.close()
  }

  def readByte(): Byte = decode_byte(st, decoder.get()).toByte

  def readRegionValue(region: Region): Long = decode_row(st, decoder.get(), region.get())

  def seek(offset: Long): Unit = ???
}

object EmitPackEncoder { self =>

  type Emitter = EstimableEmitter[MethodBuilderSelfLike]

  def emitTypeSize(t: PType): Int = {
    t match {
      case t: PArray => 120 + emitTypeSize(t.elementType)
      case t: PStruct => 100
      case _ => 20
    }
  }

  def emitStruct(t: PStruct, requestedType: PStruct, mb: MethodBuilder, region: Code[Region], off: Code[Long], out: Code[OutputBuffer]): Code[Unit] = {
    val writeMissingBytes =
      if (requestedType.size == t.size)
        out.writeBytes(region, off, t.nMissingBytes)
      else {
        var c: Code[Unit] = Code._empty[Unit]
        var j = 0
        var n = 0
        while (j < requestedType.size) {
          var b = const(0)
          var k = 0
          while (k < 8 && j < requestedType.size) {
            val rf = requestedType.fields(j)
            if (!rf.typ.required) {
              val i = t.fieldIdx(rf.name)
              b = b | (t.isFieldMissing(region, off, i).toI << k)
              k += 1
            }
            j += 1
          }
          if (k > 0) {
            c = Code(c, out.writeByte(b.toB))
            n += 1
          }
        }
        assert(n == requestedType.nMissingBytes)
        c
      }

    val foff = mb.newField[Long]

    val fieldEmitters = requestedType.fields.zipWithIndex.map { case (rf, j) =>
      val i = t.fieldIdx(rf.name)
      val f = t.fields(i)
      new Emitter {
        def emit(mbLike: MethodBuilderSelfLike): Code[Unit] = {
          val mb = mbLike.mb
          val region = mb.getArg[Region](1).load()

          t.isFieldDefined(region, foff, i).mux(
            self.emit(f.typ, rf.typ, mb, region, t.fieldOffset(foff, i), out),
            Code._empty[Unit])
        }

        def estimatedSize: Int = emitTypeSize(rf.typ)
      }
    }

    Code(
      writeMissingBytes,
      foff := off,
      EmitUtils.wrapToMethod(fieldEmitters, new MethodBuilderSelfLike(mb)),
      Code._empty[Unit])
  }

  def emitTuple(t: PTuple, requestedType: PTuple, mb: MethodBuilder, region: Code[Region], off: Code[Long], out: Code[OutputBuffer]): Code[Unit] = {
    val foff = mb.newField[Long]

    val writeMissingBytes = out.writeBytes(region, off, t.nMissingBytes)

    val fieldEmitters = t.types.zipWithIndex.map { case (ft, i) =>
      new Emitter {
        def emit(mbLike: MethodBuilderSelfLike): Code[Unit] = {
          val mb = mbLike.mb
          val region = mb.getArg[Region](1).load()

          t.isFieldDefined(region, foff, i).mux(
            self.emit(ft, requestedType.types(i), mb, region, t.fieldOffset(foff, i), out),
            Code._empty[Unit])
        }

        def estimatedSize: Int = emitTypeSize(ft)
      }
    }

    Code(
      writeMissingBytes,
      foff := off,
      EmitUtils.wrapToMethod(fieldEmitters, new MethodBuilderSelfLike(mb)),
      Code._empty[Unit])
  }

  def emitArray(t: PArray, requestedType: PArray, mb: MethodBuilder, region: Code[Region], aoff: Code[Long], out: Code[OutputBuffer]): Code[Unit] = {
    val length = region.loadInt(aoff)

    val writeLen = out.writeInt(length)
    val writeMissingBytes =
      if (!t.elementType.required) {
        val nMissingBytes = (length + 7) >>> 3
        out.writeBytes(region, aoff + const(4), nMissingBytes)
      } else
        Code._empty[Unit]

    val i = mb.newLocal[Int]("i")

    val writeElems = Code(
      i := 0,
      Code.whileLoop(
        i < length,
        Code(t.isElementDefined(region, aoff, i).mux(
          emit(t.elementType, requestedType.elementType, mb, region, t.elementOffset(aoff, length, i), out),
          Code._empty[Unit]),
          i := i + const(1))))

    Code(writeLen, writeMissingBytes, writeElems)
  }

  def writeBinary(mb: MethodBuilder, region: Code[Region], boff: Code[Long], out: Code[OutputBuffer]): Code[Unit] = {
    val length = region.loadInt(boff)
    Code(
      out.writeInt(length),
      out.writeBytes(region, boff + const(4), length))
  }

  def emit(t: PType, requestedType: PType, mb: MethodBuilder, region: Code[Region], off: Code[Long], out: Code[OutputBuffer]): Code[Unit] = {
    t.fundamentalType match {
      case t: PStruct => emitStruct(t, requestedType.fundamentalType.asInstanceOf[PStruct], mb, region, off, out)
      case t: PTuple => emitTuple(t, requestedType.fundamentalType.asInstanceOf[PTuple], mb, region, off, out)
      case t: PArray => emitArray(t, requestedType.fundamentalType.asInstanceOf[PArray], mb, region, Region.loadAddress(off), out)
      case _: PBoolean => out.writeBoolean(Region.loadBoolean(off))
      case _: PInt32 => out.writeInt(Region.loadInt(off))
      case _: PInt64 => out.writeLong(Region.loadLong(off))
      case _: PFloat32 => out.writeFloat(Region.loadFloat(off))
      case _: PFloat64 => out.writeDouble(Region.loadDouble(off))
      case _: PBinary => writeBinary(mb, region, Region.loadAddress(off), out)
    }
  }

  def encode(t: PType, rt: PType, v: Code[_], mb: MethodBuilder): Code[Unit] = {
    val region: Code[Region] = mb.getArg[Region](1)
    val out: Code[OutputBuffer] = mb.getArg[OutputBuffer](3)
    t.fundamentalType match {
      case _: PBoolean => out.writeBoolean(coerce[Boolean](v))
      case _: PInt32 => out.writeInt(coerce[Int](v))
      case _: PInt64 => out.writeLong(coerce[Long](v))
      case _: PFloat32 => out.writeFloat(coerce[Float](v))
      case _: PFloat64 => out.writeDouble(coerce[Double](v))
      case t: PArray => emitArray(t, rt.fundamentalType.asInstanceOf[PArray], mb, region, coerce[Long](v), out)
      case _: PBinary => writeBinary(mb, region, coerce[Long](v), out)
      case _ => emit(t, rt, mb, region, coerce[Long](v), out)
    }
  }

  def buildMethod(t: PType, rt: PType, fb: EmitFunctionBuilder[_]): ir.EmitMethodBuilder = {
    val mb = fb.newMethod(Array[TypeInfo[_]](typeInfo[Region], ir.typeToTypeInfo(rt), typeInfo[OutputBuffer]), typeInfo[Unit])
    mb.emit(encode(t, rt, mb.getArg(2)(ir.typeToTypeInfo(rt)), mb))
    mb
  }

  def apply(t: PType, requestedType: PType): () => AsmFunction3[Region, Long, OutputBuffer, Unit] = {
    val fb = new Function3Builder[Region, Long, OutputBuffer, Unit]
    val mb = fb.apply_method
    val offset = mb.getArg[Long](2).load()
    mb.emit(encode(t, requestedType, Region.getIRIntermediate(t)(offset), mb))
    fb.result()
  }
}

case class NativeEncoderModule(
  modKey: String,
  modBinary: Array[Byte]) extends Serializable

final class NativePackEncoder(out: OutputStream, module: NativeEncoderModule) extends Encoder {
  private[this] val st = new NativeStatus()
  private[this] val mod = new NativeModule(module.modKey, module.modBinary)
  private[this] val makeOutputBufferF = mod.findPtrFuncL1(st, "makeOutputBuffer")
  assert(st.ok, st.toString())
  private[this] val encodeByteF = mod.findLongFuncL2(st, "encode_byte")
  assert(st.ok, st.toString())
  private[this] val encodeRVF = mod.findLongFuncL2(st, "encode_row")
  assert(st.ok, st.toString())
  private[this] val flushF = mod.findLongFuncL1(st, "encoder_flush")
  assert(st.ok, st.toString())
  private[this] val closeF = mod.findLongFuncL1(st, "encoder_close")
  assert(st.ok, st.toString())

  private[this] val objArray = new ObjectArray(out)
  assert(st.ok, st.toString())
  val buf = new NativePtr(makeOutputBufferF, st, objArray.get())
  objArray.close()
  makeOutputBufferF.close()

  def flush(): Unit = {
    flushF(st, buf.get())
    assert(st.ok, st.toString())
    out.flush()
  }

  def close(): Unit = {
    closeF(st, buf.get())
    assert(st.ok, st.toString())
    buf.close()
    encodeByteF.close()
    encodeRVF.close()
    flushF.close()
    closeF.close()
    mod.close()
    st.close()
    out.close()
  }

  def writeRegionValue(region: Region, offset: Long): Unit = {
    encodeRVF(st, buf.get(), offset)
    assert(st.ok, st.toString())
  }

  def writeByte(b: Byte): Unit = {
    encodeByteF(st, buf.get(), b)
    assert(st.ok, st.toString())
  }

  def indexOffset(): Long = ???
}
