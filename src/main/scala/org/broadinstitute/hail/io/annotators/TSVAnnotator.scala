package org.broadinstitute.hail.io.annotators

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr

import scala.collection.mutable

trait TSVAnnotator {
  def parseStringType(s: String): expr.Type = {
    s match {
      case "Float" => expr.TFloat
      case "Double" => expr.TDouble
      case "Int" => expr.TInt
      case "Long" => expr.TLong
      case "Boolean" => expr.TBoolean
      case "String" => expr.TString
      case _ => fatal(
        s"""Unrecognized type "$s".  Hail supports parsing the following types in annotations:
            |  - Float (4-byte floating point number)
            |  - Double (8-byte floating point number)
            |  - Int (4-byte integer)
            |  - Long (8-byte integer)
            |  - Boolean
            |  - String
            |
             |  Note that the above types are case sensitive.""".stripMargin)
    }
  }

  def buildParsers(missing: String,
    namesAndTypes: Array[(String, Option[expr.Type])]): Array[(mutable.ArrayBuilder[Annotation], String) => Unit] = {
    namesAndTypes.map {
      case (head, ot) =>
        ot match {
          case Some(t) => (ab: mutable.ArrayBuilder[Annotation], s: String) => {
            if (s == missing) {
              ab += null: Annotation
              ()
            }
            else
              try {
                ab += t.parse(s)
                ()
              } catch {
                case e: Exception =>
                  fatal(s"""${e.getClass.getName}: tried to convert "$s" to $t in column "$head" """)
              }
          }
          case None => (ab: mutable.ArrayBuilder[Annotation], s: String) => ()
        }
    }
  }
}
