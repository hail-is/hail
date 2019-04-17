package is.hail.cxx

import is.hail.expr.types.physical.PNDArray

object EncodeNDArray {

  // Encode for the .npy format. Spec found here:
  // https://www.numpy.org/devdocs/reference/generated/numpy.lib.format.html
  def npy(tub: TranslationUnitBuilder, t: PNDArray, output_buf_ptr: Expression, nd: Expression): Code = {
    val numpyMagicStr = "93NUMPY"
    val npyFormatMajorVersion = "02"

    s"""
       | $output_buf_ptr->write_bytes($numpyMagicStr, 6);
       | $output_buf_ptr->write_byte($npyFormatMajorVersion);
       | ${ encodeHeader(tub, t, output_buf_ptr, nd) }
       | ${ encodeData(tub, t, output_buf_ptr, nd) }
     """.stripMargin
  }

  private def encodeHeader(
    tub: TranslationUnitBuilder,
    t: PNDArray,
    output_buf_ptr: Expression,
    nd: Expression): Code = {

    val header = 
  }

  private def encodeData(
    tub: TranslationUnitBuilder,
    t: PNDArray,
    output_buf_ptr: Expression,
    nd: Expression): Code = {

    val dims = Array.tabulate(t.nDims){ i => tub.variable(s"dim${i}_", "int") }
    val index = dims.zipWithIndex.foldRight("0"){ case ((idx, dim), linearIndex) =>
      s"$idx * $nd.strides[$dim] + $linearIndex"
    }

    val element = Expression(s"load_element<${ typeToCXXType(t.elementType) }>(load_index($nd, $index))")
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
