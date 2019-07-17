package is.hail.cxx

import is.hail.expr.types.physical._
import is.hail.io.{BufferSpec, NativeDecoderModule}

object PackDecoder {
  def skipBinary(input_buf_ptr: Expression, fb: FunctionBuilder): Code = {
    val len = fb.variable("skip_len", "int", s"$input_buf_ptr->read_int()")
    s"""
       |${ len.define }
       |$input_buf_ptr->skip_bytes($len);""".stripMargin
  }

  def skipArray(t: PArray, input_buf_ptr: Expression, fb: FunctionBuilder): Code = {
    val len = fb.variable("len", "int", s"$input_buf_ptr->read_int()")
    val i = fb.variable("i", "int", "0")
    if (t.elementType.required) {
      s"""
         |${ len.define }
         |for (${ i.define } $i < $len; $i++) {
         |  ${ skip(t.elementType, input_buf_ptr, fb) }
         |}""".stripMargin
    } else {
      val missingBytes = fb.arrayVariable(s"missing", "char", s"n_missing_bytes($len)")
      s"""
         |${ len.define }
         |${ missingBytes.define }
         |$input_buf_ptr->read_bytes($missingBytes, n_missing_bytes($len));
         |for (${ i.define } $i < $len; $i++) {
         |  if (!load_bit($missingBytes, $i)) {
         |    ${ skip(t.elementType, input_buf_ptr, fb) }
         |  }
         |}""".stripMargin
    }
  }

  def skipBaseStruct(t: PBaseStruct, input_buf_ptr: Expression, fb: FunctionBuilder): Code = {
    val missingBytes = fb.arrayVariable("missing", "char", s"${ t.nMissingBytes }")
    val skipFields = Array.tabulate[Code](t.size) { idx =>
      val fieldType = t.types(idx)
      if (fieldType.required)
        skip(fieldType, input_buf_ptr, fb)
      else
        s"""
           |if (!load_bit($missingBytes, ${ t.missingIdx(idx) })) {
           |  ${ skip(fieldType, input_buf_ptr, fb) }
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

  def skip(t: PType, input_buf_ptr: Expression, fb: FunctionBuilder): Code = t match {
    case t2: PArray => skipArray(t2, input_buf_ptr, fb)
    case t2: PBaseStruct => skipBaseStruct(t2, input_buf_ptr, fb)
    case _: PBinary => skipBinary(input_buf_ptr, fb)
    case _: PBoolean => s"$input_buf_ptr->skip_boolean();"
    case _: PInt32 => s"$input_buf_ptr->skip_int();"
    case _: PInt64 => s"$input_buf_ptr->skip_long();"
    case _: PFloat32 => s"$input_buf_ptr->skip_float();"
    case _: PFloat64 => s"$input_buf_ptr->skip_double();"
  }

  def decodeBinary(input_buf_ptr: Expression, region: Expression, off: Expression, fb: FunctionBuilder): Code = {
    val len = fb.variable("len", "int", s"$input_buf_ptr->read_int()")
    val boff = fb.variable("boff", "char *", s"$region->allocate(${ PBinary.contentAlignment }, $len + 4)")
    s"""
       |${ len.define }
       |${ boff.define }
       |store_address($off, $boff);
       |store_int($boff, $len);
       |$input_buf_ptr->read_bytes($boff + 4, $len);""".stripMargin
  }

  def decodeArray(t: PArray, rt: PArray, input_buf_ptr: Expression, region: Expression, off: Expression, fb: FunctionBuilder): Code = {
    val len = fb.variable("len", "int", s"$input_buf_ptr->read_int()")
    val sab = new StagedContainerBuilder(fb, region.toString, rt)
    var decodeElt = decode(t.elementType, rt.elementType, input_buf_ptr, region, Expression(sab.eltOffset), fb)
    if (!rt.elementType.required)
      decodeElt =
        s"""
           |if (!${ rt.cxxIsElementMissing(sab.end(), sab.idx) }) {
           |  $decodeElt
           |}
         """.stripMargin

    s"""
       |${ len.define }
       |${ sab.start(len.toString, clearMissing = false) }
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
      val t_missing_bytes = fb.arrayVariable(s"missing", "char", s"${ t.nMissingBytes }")

      var rtidx = 0
      val decodeFields = Array.tabulate[Code](t.size) { tidx =>
        val f = t.types(tidx)
        val skipField = rtidx >= rt.size || names(tidx) != rnames(rtidx)
        val processField = if (skipField)
          wrapMissing(s"$t_missing_bytes", tidx)(skip(f, input_buf_ptr, fb))
        else
          s"""
             |${ wrapMissing(s"$t_missing_bytes", tidx)(ssb.addField(rtidx, foff => decode(f, rt.types(rtidx), input_buf_ptr, region, Expression(foff), fb))) }
             |${ if (f.required) "" else s" else { ${ ssb.setMissing(rtidx) } }" }
           """.stripMargin
        if (!skipField)
          rtidx += 1
        processField
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

  def apply(t: PType, rt: PType, inputStreamType: String, bufSpec: BufferSpec, tub: TranslationUnitBuilder): Class = {
    tub.include("hail/hail.h")
    tub.include("hail/Decoder.h")
    tub.include("hail/Region.h")
    tub.include("hail/Utils.h")
    tub.include("<cstdio>")
    tub.include("<memory>")

    val decoderBuilder = tub.buildClass(tub.genSym("Decoder"), "NativeObj")

    val bufType = bufSpec.nativeInputBufferType(inputStreamType)
    val buf = decoderBuilder.variable("buf", s"std::shared_ptr<$bufType>")
    decoderBuilder += buf

    decoderBuilder += s"${ decoderBuilder.name }(std::shared_ptr<$inputStreamType> is) : $buf(std::make_shared<$bufType>(is)) { }"

    val (valueType, initialSize, returnVal) = rt match {
      case typ if typ.isPrimitive =>
        (typeToCXXType(rt), rt.byteSize, { r: Variable => s"*reinterpret_cast<${typeToCXXType(rt)} *>($r)" })
      case _: PArray | _: PBinary => ("char *", 8, { r: Variable => s"load_address($r)" })
      case _ => ("char *", rt.byteSize, { r: Variable => s"$r" })
    }

    val rowFB = decoderBuilder.buildMethod("decode_row", Array("Region *" -> "region"), valueType, const = true)
    val region = rowFB.getArg(0)
    val row = rowFB.variable("row", "char *", s"$region->allocate(${ rt.alignment }, $initialSize)")
    rowFB += row.define
    rowFB += decode(t.fundamentalType, rt.fundamentalType, buf.ref, region.ref, row.ref, rowFB)
    rowFB += s"return ${ returnVal(row) };"
    rowFB.end()

    val byteFB = decoderBuilder.buildMethod("decode_byte", Array(), "char", const = true)
    byteFB += s"return $buf->read_byte();"
    byteFB.end()

    decoderBuilder.end()
  }

  def buildModule(t: PType, rt: PType, bufSpec: BufferSpec): NativeDecoderModule = {
    assert(t.isInstanceOf[PBaseStruct] || t.isInstanceOf[PArray])
    val tub = new TranslationUnitBuilder()

    val decoder = apply(t, rt, "InputStream", bufSpec, tub)
    
    tub.include("hail/Decoder.h")
    tub.include("hail/ObjectArray.h")
    tub.include("<memory>")

    val inBufFB = tub.buildFunction("make_input_buffer", Array("NativeStatus*" -> "st", "long" -> "objects"), "NativeObjPtr")
    inBufFB += "UpcallEnv up;"
    inBufFB += s"auto jinput_stream = reinterpret_cast<ObjectArray*>(${ inBufFB.getArg(1) })->at(0);"
    inBufFB += s"return std::make_shared<$decoder>(std::make_shared<InputStream>(up, jinput_stream));"
    inBufFB.end()

    val rowFB = tub.buildFunction("decode_row", Array("NativeStatus*" -> "st", "long" -> "buf", "long" -> "region"), "long")
    rowFB += s"return (long) reinterpret_cast<$decoder *>(${ rowFB.getArg(1) })->decode_row(reinterpret_cast<ScalaRegion *>(${ rowFB.getArg(2) })->get_wrapped_region());"
    rowFB.end()

    val byteFB = tub.buildFunction("decode_byte", Array("NativeStatus*" -> "st", "long" -> "buf"), "long")
    byteFB += s"return (long) reinterpret_cast<$decoder *>(${ byteFB.getArg(1) })->decode_byte();"
    byteFB.end()

    val mod = tub.end().build("-O2")

    NativeDecoderModule(mod.getKey, mod.getBinary)
  }

}
