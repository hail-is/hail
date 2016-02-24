package org.broadinstitute.hail.annotations

import org.broadinstitute.hail.expr

abstract class AnnotationSignature {
  def typeOf: String

  def index: Int

  def remapIndex(newIndex: Int): AnnotationSignature

  def toExprType: expr.Type = typeOf match {
    case "Unit" => expr.TUnit
    case "Boolean" => expr.TBoolean
    case "Character" => expr.TChar
    case "Int" => expr.TInt
    case "Long" => expr.TLong
    case "Float" => expr.TFloat
    case "Double" => expr.TDouble
    case "Array[Int]" => expr.TArray(expr.TInt)
    case "Array[Double]" => expr.TArray(expr.TInt)
    case "Set[Int]" => expr.TSet(expr.TInt)
    case "Set[String]" => expr.TSet(expr.TString)
    case "String" => expr.TString
  }
}

case class SimpleSignature(typeOf: String, index: Int = -1) extends AnnotationSignature {
  def remapIndex(newIndex: Int): AnnotationSignature = this.copy(index = newIndex)
}
