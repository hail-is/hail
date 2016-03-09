package org.broadinstitute.hail.annotations

import htsjdk.variant.vcf.{VCFInfoHeaderLine, VCFHeaderLineCount, VCFHeaderLineType}
import org.broadinstitute.hail.expr

case class VCFSignature(dType: expr.Type, vcfType: String, number: String, description: String)
  extends Signature

object VCFSignature {
  val integerRegex = """(\d+)""".r

  def parse(line: VCFInfoHeaderLine): VCFSignature = {
    val vcfType = line.getType.toString
    val parsedType = line.getType match {
      case VCFHeaderLineType.Integer => expr.TInt
      case VCFHeaderLineType.Float => expr.TDouble
      case VCFHeaderLineType.String => expr.TString
      case VCFHeaderLineType.Character => expr.TChar
      case VCFHeaderLineType.Flag => expr.TBoolean
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

  def getArrayType(st: expr.Type): expr.Type = {
    st match {
      case expr.TInt => expr.TArray(expr.TInt)
      case expr.TDouble   => expr.TArray(expr.TDouble)
      case expr.TString => expr.TArray(expr.TString)
      case _ => throw new UnsupportedOperationException()
    }
  }
}
