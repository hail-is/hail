package is.hail.cxx

import is.hail.expr.types.physical._

object NumpyType {
  // Follows numpy Array datatype protocol:
  // https://docs.scipy.org/doc/numpy/reference/arrays.interface.html
  // 1. < or > for Little/Big Endian or | for not relevant
  // 2. char for the basic element type
  // 3. int for number of bytes
  def apply(t: PType): String = {
    t match {
      case _: PInt32 => "<i4"
      case _: PInt64 => "<i8"
      case _: PFloat32 => "<f4"
      case _: PFloat64 => "<f8"
      case _: PBoolean => "<b1"
      case _ => throw new UnsupportedOperationException(s"Type not supported in npy conversion: $t")
    }
  }
}

object EncodeNDArray {

  // Encode for the .npy format. Spec found here:
  // https://www.numpy.org/devdocs/reference/generated/numpy.lib.format.html
  def npy(tub: TranslationUnitBuilder, t: PNDArray, output_buf_ptr: Expression, ndExpr: Expression): Code = {
    val nd = tub.variable("nd", "NDArray", ndExpr.toString)

    val numpyMagicStr = "\"\\x93NUMPY\""
    val magicStrLen = 6
    val npyFormatMajorVersion = "'\\x01'"
    val npyFormatMinorVersion = "'\\x00'"
    val formatVersionLen = 2

    val headerLenLen = 2
    val dtype = "\"" + NumpyType(t.elementType) + "\""
    val header = tub.variable("header", "std::string", s"npy_header($nd, $dtype)")
    val headerOffset = tub.variable("header_offset", "short", s"$header.length()")
    val headerPadding = tub.variable("header_padding", "short",
      s"64 - ($magicStrLen + $formatVersionLen + $headerLenLen + $headerOffset) % 64")
    val totalHeaderLen = tub.variable("header_len", "short", s"$headerOffset + $headerPadding")

    s"""
       | ${ nd.define }
       | ${ header.define }
       | ${ headerOffset.define }
       | ${ headerPadding.define }
       | ${ totalHeaderLen.define }
       |
       | $output_buf_ptr->write_bytes($numpyMagicStr, $magicStrLen);
       | $output_buf_ptr->write_byte($npyFormatMajorVersion);
       | $output_buf_ptr->write_byte($npyFormatMinorVersion);
       | $output_buf_ptr->write_bytes(reinterpret_cast<const char *>(&$totalHeaderLen), $headerLenLen);
       | $output_buf_ptr->write_bytes($header.c_str(), $header.length());
       | for (int i = 0; i < $headerPadding - 1; ++i) {
       |   $output_buf_ptr->write_byte(' ');
       | }
       | $output_buf_ptr->write_byte('\\n');
       |
       | ${ encodeData(tub, t, output_buf_ptr, nd) }
     """.stripMargin
  }

  private def encodeData(
    tub: TranslationUnitBuilder,
    t: PNDArray,
    output_buf_ptr: Expression,
    nd: Variable): Code = {

    val dims = Array.tabulate(t.nDims){ i => tub.variable(s"dim${i}_", "int") }

    val element = Expression(NDArrayEmitter.loadElement(nd, dims, t.elementType))
    val body = PackEncoder.encode(tub, t.elementType, output_buf_ptr, element)
    dims.zipWithIndex.foldRight(body){ case ((dimVar, dimIdx), innerLoops) =>
      s"""
         |${ dimVar.define }
         |for ($dimVar = 0; $dimVar < $nd.shape[$dimIdx]; ++$dimVar) {
         |  $innerLoops
         |}
         |""".stripMargin
    }
  }
}
