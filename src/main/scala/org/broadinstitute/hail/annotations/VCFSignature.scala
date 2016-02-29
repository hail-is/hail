package org.broadinstitute.hail.annotations

import htsjdk.variant.vcf.{VCFInfoHeaderLine, VCFHeaderLineCount, VCFHeaderLineType}
import org.broadinstitute.hail.annotations.SignatureType.SignatureType
import org.broadinstitute.hail.expr

case class VCFSignature(dType: expr.Type, vcfType: String, number: String, description: String)
  extends Signature

object VCFSignature {
  val integerRegex = """(\d+)""".r

  def parse(line: VCFInfoHeaderLine): VCFSignature = {
    val vcfType = line.getType.toString
    val parsedType = line.getType match {
      case VCFHeaderLineType.Integer => SignatureType.Int
      case VCFHeaderLineType.Float => SignatureType.Double
      case VCFHeaderLineType.String => SignatureType.String
      case VCFHeaderLineType.Character => SignatureType.Character
      case VCFHeaderLineType.Flag => SignatureType.Boolean
    }
    val parsedCount = line.getCountType match {
      case VCFHeaderLineCount.A => "A"
      case VCFHeaderLineCount.G => "G"
      case VCFHeaderLineCount.R => "R"
      case VCFHeaderLineCount.INTEGER => line.getCount.toString
      case VCFHeaderLineCount.UNBOUNDED => "."
    }
    // FIXME "A" should produce array
    val scalaType = parsedCount match {
      case "A" | "R" | "G" => getArrayType(parsedType)
      case integerRegex(i) => if (i.toInt > 1) getArrayType(parsedType) else parsedType
      case _ => parsedType
    }
    val desc = line.getDescription

    new VCFSignature(scalaType, vcfType, parsedCount, desc)
  }

  def getArrayType(st: SignatureType): SignatureType = {
    st match {
      case SignatureType.Int => SignatureType.ArrayInt
      case SignatureType.Double => SignatureType.ArrayDouble
      case SignatureType.String => SignatureType.ArrayString
    }
  }
}
