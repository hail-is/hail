package is.hail.cxx

import is.hail.expr.types.physical._
import is.hail.io.{BufferSpec, NativeEncoderModule}

object PackEncoder {

  def encodeBinary(output_buf_ptr: Expression, off: Expression): Code = {
    val len = Variable("len", "int", s"load_length($off)")
    s"""
       |${ len.define }
       |$output_buf_ptr->write_int($len);
       |$output_buf_ptr->write_bytes($off + 4, $len);""".stripMargin
  }

  def encodeArray(t: PArray, output_buf_ptr: Expression, off: Expression): Code = {
    val len = Variable("len", "int", s"load_length($off)")
    val i = Variable("i", "int", s"0")
    val copyLengthAndMissing = if (t.elementType.required)
      s"$output_buf_ptr->write_int($len);"
    else
      s"""
         |$output_buf_ptr->write_int($len);
         |$output_buf_ptr->write_bytes($off + 4, n_missing_bytes($len));""".stripMargin
    val eltOff = Variable("eoff",
      "char *",
      s"round_up_alignment(${ if (!t.elementType.required) s"$off + 4 + n_missing_bytes($len)" else s"$off + 4" }, ${ t.elementType.alignment })")
    val elt = t.elementType match {
      case (_: PBinary | _: PArray) => s"load_address($eltOff)"
      case _ => eltOff.toString
    }

    val writeElt = if (t.elementType.required)
      encode(t.elementType, output_buf_ptr, Expression(elt))
    else
      s"""
         |if (!load_bit($off + 4, $i)) {
         |  ${ encode(t.elementType, output_buf_ptr, Expression(elt)) };
         |}""".stripMargin

    s"""
       |${ len.define }
       |$copyLengthAndMissing
       |${ eltOff.define }
       |for (${ i.define } $i < $len; $i++) {
       |  $writeElt
       |  $eltOff += ${ t.elementByteSize };
       |}
      """.stripMargin
  }

  def encodeBaseStruct(t: PBaseStruct, output_buf_ptr: Expression, off: Expression): Code = {
    val nMissingBytes = t.nMissingBytes
    val storeFields: Array[Code] = Array.tabulate[Code](t.size) { idx =>
      val store = t.types(idx) match {
        case t2@(_: PArray | _: PBinary) =>
          encode(t2, output_buf_ptr, Expression(s"load_address($off + ${ t.byteOffsets(idx) })"))
        case t2 =>
          encode(t2, output_buf_ptr, Expression(s"$off + ${ t.byteOffsets(idx) }"))

      }
      if (t.fieldRequired(idx)) {
        store
      } else {
        s"""
           |if (!load_bit($off, ${ t.missingIdx(idx) })) {
           |  $store
           |}""".stripMargin
      }
    }
    s"""
       |$output_buf_ptr->write_bytes($off, $nMissingBytes);
       |${ storeFields.mkString("\n") }
      """.stripMargin
  }

  def encode(t: PType, output_buf_ptr: Expression, off: Expression): Code = t match {
    case _: PBoolean => s"$output_buf_ptr->write_byte(*($off) ? 1 : 0);"
    case _: PInt32 => s"$output_buf_ptr->write_int(load_int($off));"
    case _: PInt64 => s"$output_buf_ptr->write_long(load_long($off));"
    case _: PFloat32 => s"$output_buf_ptr->write_float(load_float($off));"
    case _: PFloat64 => s"$output_buf_ptr->write_double(load_double($off));"
    case _: PBinary => encodeBinary(output_buf_ptr, off)
    case t2: PArray => encodeArray(t2, output_buf_ptr, off)
    case t2: PBaseStruct => encodeBaseStruct(t2, output_buf_ptr, off)
  }

  def apply(t: PType, bufSpec: BufferSpec, tub: TranslationUnitBuilder): Class = {
    assert(t.isInstanceOf[PBaseStruct] || t.isInstanceOf[PArray])
    tub.include("hail/hail.h")
    tub.include("hail/Encoder.h")
    tub.include("hail/ObjectArray.h")
    tub.include("hail/Utils.h")
    tub.include("<cstdio>")
    tub.include("<memory>")

    val encBuilder = new ClassBuilder("Encoder", "NativeObj")

    val bufType = bufSpec.nativeOutputBufferType
    val buf = Variable("buf", s"std::shared_ptr<$bufType>")
    encBuilder.addPrivate(buf)

    encBuilder.addConstructor(s"${ encBuilder.name }(std::shared_ptr<OutputStream> os) : $buf(std::make_shared<$bufType>(os)) { }")

    val rowFB = FunctionBuilder("encode_row", Array("NativeStatus*" -> "st", "char const*" -> "row"), "void")
    rowFB += encode(t.fundamentalType, buf.ref, rowFB.getArg(1).ref)
    rowFB += "return;"
    encBuilder += rowFB.result()

    val byteFB = FunctionBuilder("encode_byte", Array("NativeStatus*" -> "st", "char" -> "b"), "void")
    byteFB += s"$buf->write_byte(${ byteFB.getArg(1) });"
    byteFB += "return;"
    encBuilder += byteFB.result()

    val flushFB = FunctionBuilder("flush", Array("NativeStatus*" -> "st"), "void")
    flushFB += s"$buf->flush();"
    flushFB += "return;"
    encBuilder += flushFB.result()

    val closeFB = FunctionBuilder("close", Array("NativeStatus*" -> "st"), "void")
    closeFB +=
      s"""
         |$buf->close();
         |return;""".stripMargin
    encBuilder += closeFB.result()

    encBuilder.result()
  }

  def buildModule(t: PType, bufSpec: BufferSpec): NativeEncoderModule = {
    assert(t.isInstanceOf[PBaseStruct] || t.isInstanceOf[PArray])
    val tub = new TranslationUnitBuilder()

    val encClass = apply(t, bufSpec, tub)
    tub += encClass

    val outBufFB = FunctionBuilder("makeOutputBuffer", Array("NativeStatus*" -> "st", "long" -> "objects"), "NativeObjPtr")
    outBufFB += "UpcallEnv up;"
    outBufFB += s"auto joutput_stream = reinterpret_cast<ObjectArray*>(${ outBufFB.getArg(1) })->at(0);"
    val bufType = bufSpec.nativeOutputBufferType
    outBufFB += s"return std::make_shared<$encClass>(std::make_shared<OutputStream>(up, joutput_stream));"
    tub += outBufFB.result()

    val rowFB = FunctionBuilder("encode_row", Array("NativeStatus*" -> "st", "long" -> "buf", "long" -> "row"), "long")
    rowFB += s"reinterpret_cast<$encClass *>(${ rowFB.getArg(1) })->encode_row(${ rowFB.getArg(0) }, reinterpret_cast<char *>(${ rowFB.getArg(2) }));"
    rowFB += "return 0;"
    tub += rowFB.result()

    val byteFB = FunctionBuilder("encode_byte", Array("NativeStatus*" -> "st", "long" -> "buf", "long" -> "b"), "long")
    byteFB += s"reinterpret_cast<$encClass *>(${ byteFB.getArg(1) })->encode_byte(${ byteFB.getArg(0) }, ${ byteFB.getArg(2) } & 0xff);"
    byteFB += "return 0;"
    tub += byteFB.result()

    val flushFB = FunctionBuilder("encoder_flush", Array("NativeStatus*" -> "st", "long" -> "buf"), "long")
    flushFB += s"reinterpret_cast<$encClass *>(${ flushFB.getArg(1) })->flush(${ flushFB.getArg(0) });"
    flushFB += "return 0;"
    tub += flushFB.result()

    val closeFB = FunctionBuilder("encoder_close", Array("NativeStatus*" -> "st", "long" -> "buf"), "long")
    closeFB += s"reinterpret_cast<$encClass *>(${ closeFB.getArg(1) })->close(${ closeFB.getArg(0) });"
    closeFB += "return 0;"
    tub += closeFB.result()

    val mod = tub.result().build("-O1 -llz4")

    NativeEncoderModule(mod.getKey, mod.getBinary)
  }
}