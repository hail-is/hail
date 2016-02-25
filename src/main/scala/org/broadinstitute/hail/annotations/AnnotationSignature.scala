package org.broadinstitute.hail.annotations

import java.io.{DataInputStream, DataOutputStream}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.expr

abstract class AnnotationSignature {
  def typeOf: String

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
    case "Array[String]" => expr.TArray(expr.TString)
    case "Set[Int]" => expr.TSet(expr.TInt)
    case "Set[String]" => expr.TSet(expr.TString)
    case "String" => expr.TString
  }

  def parser(name: String, missing: Set[String]): (String) => Option[Any] = {
    typeOf match {
      case "Double" =>
        (v: String) =>
          try {
            someIf(!missing(v), v.toDouble)
          } catch {
            case e: java.lang.NumberFormatException =>
              fatal( s"""java.lang.NumberFormatException: tried to convert "$v" to Double in column "$name" """)
          }
      case "Int" =>
        (v: String) =>
          try {
            someIf(!missing(v), v.toInt)
          } catch {
            case e: java.lang.NumberFormatException =>
              fatal( s"""java.lang.NumberFormatException: tried to convert "$v" to Int in column "$name" """)
          }
      case "Boolean" =>
        (v: String) =>
          try {
            someIf(!missing(v), v.toBoolean)
          } catch {
            case e: java.lang.IllegalArgumentException =>
              fatal( s"""java.lang.IllegalArgumentException: tried to convert "$v" to Boolean in column "$name" """)
          }
      case "String" =>
        (v: String) =>
          someIf(!missing(v), v)
      case _ =>
        fatal(
          s"""Unrecognized type "$typeOf" in signature "$name".  Hail supports parsing the following types in annotations:
              |  - Double (floating point number)
              |  - Int  (integer)
              |  - Boolean
              |  - String
              |
               |  Note that the above types are case sensitive.
             """.stripMargin)
    }
  }
}

case class SimpleSignature(typeOf: String) extends AnnotationSignature
