package org.broadinstitute.hail.annotations

case class VCFSignature(vcfType: String, emitType: String, number: String,
  emitConversionIdentifier: String, description: String)
  extends AnnotationSignature {

  def emitUtilities: String = ""
}

object VCFSignature {

  val arrayRegex = """Array\[(\w+)\]""".r
  val setRegex = """Set\[(\w+)\]""".r
  def getConversionMethod(str: String): String = {
    str match {
      case arrayRegex(subType) => s"toArray$subType"
      case setRegex(subType) => s"toSet$subType"
      case _ => s"to$str"
    }
  }

  def vcfTypeToScala(str: String): String =
    str match {
      case "Flag" => "Boolean"
      case "Integer" => "Int"
      case "Float" => "Double"
      case "String" => "String"
      case "Character" => "Character"
      case "." => "String"
      case _ => throw new UnsupportedOperationException("unexpected annotation type")
    }

  def parse(number: String, vcfType: String, desc: String): AnnotationSignature = {
    val parsedType: String = vcfTypeToScala(vcfType)

    val scalaType: String = {
      if (number == "0" || number == "1") {
        parsedType
      }
      else if (number == "A" || number == "R" || number == "G") {
        s"Array[$parsedType]"
      }
      else
        throw new UnsupportedOperationException
    }
    val conversionMethod = getConversionMethod(scalaType)
    new VCFSignature(vcfType, scalaType, number, conversionMethod, desc)
  }
}