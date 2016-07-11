package org.broadinstitute.hail.io.annotators

import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._

import scala.collection.mutable

trait TSVAnnotator {

  def buildParsers(missing: String,
    namesAndTypes: Array[(String, Option[Type])]): Array[(mutable.ArrayBuilder[Annotation], String) => Unit] = {
    namesAndTypes.map {
      case (head, ot) =>
        ot match {
          case Some(t) => (ab: mutable.ArrayBuilder[Annotation], s: String) => {
            if (s == missing) {
              ab += Annotation.empty
              ()
            } else {
              try {
                ab += t.asInstanceOf[Parsable].parse(s)
                ()
              } catch {
                case e: Exception =>
                  fatal(s"""${e.getClass.getName}: tried to convert "$s" to $t in column "$head" """)
              }
            }
          }
          case None => (ab: mutable.ArrayBuilder[Annotation], s: String) => ()
        }
    }
  }
}
