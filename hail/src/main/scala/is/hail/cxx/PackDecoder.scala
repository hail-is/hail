package is.hail.cxx

import java.io.PrintWriter

import is.hail.expr.types.physical._
import is.hail.io.{BufferSpec, NativeDecoderModule}

object PackDecoder {
  def skipBinary(input_buf_ptr: Expression): Code = {
    val len = Variable("skip_len", "int", s"$input_buf_ptr->read_int()")
    s"""
       |${ len.define }
       |$input_buf_ptr->skip_bytes($len);""".stripMargin
  }

  def skipArray(t: PArray, input_buf_ptr: Expression): Code = {
    val len = Variable("len", "int", s"$input_buf_ptr->read_int()")
    val i = Variable("i", "int", "0")
    if (t.elementType.required) {
      s"""
         |${ len.define }
         |for (${ i.define } $i < $len; $i++) {
         |  ${ skip(t.elementType, input_buf_ptr) }
         |}""".stripMargin
    } else {
      val missingBytes = ArrayVariable(s"missing", "char", s"n_missing_bytes($len)")
      s"""
         |${ len.define }
         |${ missingBytes.define }
         |$input_buf_ptr->read_bytes($missingBytes, n_missing_bytes($len));
         |for (${ i.define } $i < $len; $i++) {
         |  if (!load_bit($missingBytes, $i)) {
         |    ${ skip(t.elementType, input_buf_ptr) }
         |  }
         |}""".stripMargin
    }
  }

  def skipBaseStruct(t: PBaseStruct, input_buf_ptr: Expression): Code = {
    val missingBytes = ArrayVariable("missing", "char", s"${ t.nMissingBytes }")
    val skipFields = Array.tabulate[Code](t.size) { idx =>
      val fieldType = t.types(idx)
      if (fieldType.required)
        skip(fieldType, input_buf_ptr)
      else
        s"""
           |if (!load_bit($missingBytes, ${ t.missingIdx(idx) })) {
           |  ${ skip(fieldType, input_buf_ptr) }
           |}""".stripMargin
    }

    if (t.nMissingBytes > 0)
      s"""
         |${ missingBytes.define }
         |$input_buf_ptr->read_bytes($missingBytes, ${ t.nMissingBytes });
         |${ skipFields.mkString("\n") }""".stripMargin
    else
      skipFields.mkString("\n")
  }

  def skip(t: PType, input_buf_ptr: Expression): Code = t match {
    case t2: PArray => skipArray(t2, input_buf_ptr)
    case t2: PBaseStruct => skipBaseStruct(t2, input_buf_ptr)
    case _: PBinary => skipBinary(input_buf_ptr)
    case _: PBoolean => s"$input_buf_ptr->skip_boolean();"
    case _: PInt32 => s"$input_buf_ptr->skip_int();"
    case _: PInt64 => s"$input_buf_ptr->skip_long();"
    case _: PFloat32 => s"$input_buf_ptr->skip_float();"
    case _: PFloat64 => s"$input_buf_ptr->skip_double();"
  }

  def decodeBinary(input_buf_ptr: Expression, region: Expression, off: Expression, fb: FunctionBuilder): Code = {
    val len = Variable("len", "int", s"$input_buf_ptr->read_int()")
    val boff = Variable("boff", "char *", s"$region->allocate(${ PBinary.contentAlignment }, $len + 4)")
    s"""
       |${ len.define }
       |${ boff.define }
       |store_address($off, $boff);
       |store_int($boff, $len);
       |$input_buf_ptr->read_bytes($boff + 4, $len);""".stripMargin
  }

  def decodeArray(t: PArray, rt: PArray, input_buf_ptr: Expression, region: Expression, off: Expression, fb: FunctionBuilder): Code = {
    val len = Variable("len", "int", s"$input_buf_ptr->read_int()")
    val sab = new StagedContainerBuilder(fb, region.toString, rt)
    var decodeElt = decode(t.elementType, rt.elementType, input_buf_ptr, region, Expression(sab.eltOffset), fb)
    if (rt.elementType.required)
      decodeElt =
        s"""
           |if (!${ rt.cxxIsElementMissing(sab.end(), sab.idx) }) {
           |  $decodeElt
           |}
         """.stripMargin

    s"""
       |${ len.define }
       |${ sab.start(len, clearMissing = false) }
       |store_address($off, ${ sab.end() });
       |${ if (rt.elementType.required) "" else s"$input_buf_ptr->read_bytes(${ sab.end() } + 4, ${ rt.cxxNMissingBytes(s"$len") });" }
       |while (${ sab.idx } < $len) {
       |  $decodeElt
       |  ${ sab.advance() }
       |}
     """.stripMargin
  }

  def decodeBaseStruct(t: PBaseStruct, rt: PBaseStruct, input_buf_ptr: Expression, region: Expression, off: Expression, fb: FunctionBuilder): Code = {
    val ssb = new StagedBaseStructBuilder(fb, rt, off)

    def wrapMissing(missingBytes: Code, tidx: Int)(processField: Code): Code = {
      if (!t.fieldRequired(tidx)) {
        s"""
           |if (!${ t.cxxIsFieldMissing(missingBytes, tidx) }) {
           |  $processField
           |}""".stripMargin
      } else processField
    }

    if (t.size == rt.size) {
      assert((t.isInstanceOf[PTuple] && rt.isInstanceOf[PTuple]) || (t.asInstanceOf[PStruct].fieldNames sameElements t.asInstanceOf[PStruct].fieldNames))
      val decodeFields = Array.tabulate[Code](rt.size) { idx =>
        wrapMissing(s"$off", idx)(ssb.addField(idx, foff => decode(t.types(idx), rt.types(idx), input_buf_ptr, region, Expression(foff), fb)))
      }.mkString("\n")
      if (t.nMissingBytes > 0) {
        s"""
           |$input_buf_ptr->read_bytes($off, ${ t.nMissingBytes });
           |$decodeFields""".stripMargin
      } else
        decodeFields
    } else {
      val names = t.asInstanceOf[PStruct].fieldNames
      val rnames = rt.asInstanceOf[PStruct].fieldNames
      val t_missing_bytes = ArrayVariable(s"missing", "char", s"${ t.nMissingBytes }")

      var rtidx = 0
      val decodeFields = Array.tabulate[Code](t.size) { tidx =>
        val f = t.types(tidx)
        val skipField = rtidx >= rt.size || names(tidx) != rnames(rtidx)
        val processField = if (skipField)
          wrapMissing(s"$t_missing_bytes", tidx)(skip(f, input_buf_ptr))
        else
          s"""
             |${ wrapMissing(s"$t_missing_bytes", tidx)(ssb.addField(rtidx, foff => decode(f, rt.types(rtidx), input_buf_ptr, region, Expression(foff), fb))) }
             |${ if (f.required) "" else s" else { ${ ssb.setMissing(rtidx) } }" }
           """.stripMargin
        if (!skipField)
          rtidx += 1
        wrapMissing(s"$t_missing_bytes", tidx)(processField)
      }.mkString("\n")
      if (t.nMissingBytes > 0) {
        s"""
           |${ t_missing_bytes.define }
           |${ ssb.clearAllMissing() };
           |$input_buf_ptr->read_bytes($t_missing_bytes, ${ t.nMissingBytes });
           |$decodeFields""".stripMargin
      } else
        decodeFields
    }
  }

  def decode(t: PType, rt: PType, input_buf_ptr: Expression, region: Expression, off: Expression, fb: FunctionBuilder): Code = t match {
    case _: PBoolean => s"store_byte($off, $input_buf_ptr->read_byte());"
    case _: PInt32 => s"store_int($off, $input_buf_ptr->read_int());"
    case _: PInt64 => s"store_long($off, $input_buf_ptr->read_long());"
    case _: PFloat32 => s"store_float($off, $input_buf_ptr->read_float());"
    case _: PFloat64 => s"store_double($off, $input_buf_ptr->read_double());"
    case _: PBinary => decodeBinary(input_buf_ptr, region, off, fb)
    case t2: PArray => decodeArray(t2, rt.asInstanceOf[PArray], input_buf_ptr, region, off, fb)
    case t2: PBaseStruct => decodeBaseStruct(t2, rt.asInstanceOf[PBaseStruct], input_buf_ptr, region, off, fb)
  }

  def apply(t: PType, rt: PType, bufSpec: BufferSpec, tub: TranslationUnitBuilder): Class = {
    tub.include("hail/hail.h")
    tub.include("hail/Decoder.h")
    tub.include("hail/Region.h")
    tub.include("hail/Utils.h")
    tub.include("<cstdio>")
    tub.include("<memory>")

    val decoderBuilder = new ClassBuilder(genSym("Decoder"), "NativeObj")

    val bufType = bufSpec.nativeInputBufferType
    val buf = Variable("buf", s"std::shared_ptr<$bufType>")
    decoderBuilder.addPrivate(buf)

    decoderBuilder.addConstructor(s"${ decoderBuilder.name }(std::shared_ptr<InputStream> is) : $buf(std::make_shared<$bufType>(is)) { }")

    val rowFB = FunctionBuilder("decode_row", Array("NativeStatus*" -> "st", "Region *" -> "region"), "char *")
    val region = rowFB.getArg(1)
    val initialSize = rt match {
      case _: PArray | _: PBinary => 8
      case _ => rt.byteSize
    }
    val row = Variable("row", "char *", s"$region->allocate(${ rt.alignment }, $initialSize)")
    rowFB += row.define
    rowFB += decode(t.fundamentalType, rt.fundamentalType, buf.ref, region.ref, row.ref, rowFB)
    rowFB += (rt match {
      case _: PArray | _: PBinary => s"return load_address($row);"
      case _ => s"return $row;"
    })
    decoderBuilder += rowFB.result()

    val byteFB = FunctionBuilder("decode_byte", Array("NativeStatus*" -> "st"), "char")
    byteFB += s"return $buf->read_byte();"
    decoderBuilder += byteFB.result()

    decoderBuilder.result()
  }

  def buildModule(t: PType, rt: PType, bufSpec: BufferSpec): NativeDecoderModule = {
    assert(t.isInstanceOf[PBaseStruct] || t.isInstanceOf[PArray])
    val tub = new TranslationUnitBuilder()

    val decoder = apply(t, rt, bufSpec, tub)
    tub += decoder

    tub.include("hail/Decoder.h")
    tub.include("hail/ObjectArray.h")
    tub.include("<memory>")

    val inBufFB = FunctionBuilder("make_input_buffer", Array("NativeStatus*" -> "st", "long" -> "objects"), "NativeObjPtr")
    inBufFB += "UpcallEnv up;"
    inBufFB += s"auto jinput_stream = reinterpret_cast<ObjectArray*>(${ inBufFB.getArg(1) })->at(0);"
    inBufFB += s"return std::make_shared<$decoder>(std::make_shared<InputStream>(up, jinput_stream));"
    tub += inBufFB.result()

    val rowFB = FunctionBuilder("decode_row", Array("NativeStatus*" -> "st", "long" -> "buf", "long" -> "region"), "long")
    rowFB += s"return (long) reinterpret_cast<$decoder *>(${ rowFB.getArg(1) })->decode_row(${ rowFB.getArg(0) }, reinterpret_cast<Region *>(${ rowFB.getArg(2) }));"
    tub += rowFB.result()

    val byteFB = FunctionBuilder("decode_byte", Array("NativeStatus*" -> "st", "long" -> "buf"), "long")
    byteFB += s"return (long) reinterpret_cast<$decoder *>(${ byteFB.getArg(1) })->decode_byte(${ byteFB.getArg(0) });"
    tub += byteFB.result()

    val mod = tub.result().build("-O2 -llz4")

    NativeDecoderModule(mod.getKey, mod.getBinary)
  }

}