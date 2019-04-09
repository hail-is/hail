package is.hail.io.vcf

import htsjdk.variant.variantcontext.VariantContext
import is.hail.annotations._
import is.hail.expr.ir.IRParser
import is.hail.expr.types._
import is.hail.expr.types.virtual._
import is.hail.utils._

class BufferedLineIterator(bit: BufferedIterator[String]) extends htsjdk.tribble.readers.LineIterator {
  override def peek(): String = bit.head

  override def hasNext: Boolean = bit.hasNext

  override def next(): String = bit.next()

  override def remove() {
    throw new UnsupportedOperationException
  }
}

class HtsjdkRecordReader(
    val callFields: Set[String],
    val entryFloatTypeName: String = TFloat64()._toPretty) extends Serializable {

  import HtsjdkRecordReader._

  val entryFloatType: TNumeric = IRParser.parseType(entryFloatTypeName) match {
    case t32: TFloat32 => t32
    case t64: TFloat64 => t64
    case _ => fatal(
        s"""invalid floating point type:
           |  expected ${TFloat32()._toPretty} or ${TFloat64()._toPretty}, got ${entryFloatTypeName}"""
    )
  }
}
