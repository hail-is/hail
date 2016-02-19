package org.broadinstitute.hail.annotations

import htsjdk.variant.vcf.{VCFInfoHeaderLine, VCFHeaderLineCount, VCFHeaderLineType}

case class VCFSignature(typeOf: String, vcfType: String, number: String, description: String)
  extends AnnotationSignature

object VCFSignature {
  val integerRegex = """(\d+)""".r
  def parse(line: VCFInfoHeaderLine): AnnotationSignature = {
    val vcfType = line.getType.toString
    val parsedType = line.getType match {
      case VCFHeaderLineType.Integer => "Int"
      case VCFHeaderLineType.Float => "Double"
      case VCFHeaderLineType.String => "String"
      case VCFHeaderLineType.Character => "Character"
      case VCFHeaderLineType.Flag => "Boolean"
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
      case "A" | "R" | "G" => s"Array[$parsedType]"
      case integerRegex(i) => if (i.toInt > 1) s"Array[$parsedType]" else parsedType
      case _ => parsedType
    }
    val desc = line.getDescription

    new VCFSignature(scalaType, vcfType, parsedCount, desc)
  }
}
