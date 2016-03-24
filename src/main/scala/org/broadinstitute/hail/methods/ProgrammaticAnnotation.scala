package org.broadinstitute.hail.methods

import org.broadinstitute.hail.expr
import org.broadinstitute.hail.Utils.fatal

object ProgrammaticAnnotation {
  def checkType(key: String, exprType: expr.Type): Unit = {
    exprType match {
      case expr.TArray(expr.TInt) => ()
      case expr.TArray(expr.TDouble) => ()
      case expr.TArray(expr.TString) => ()
      case expr.TString => ()
      case expr.TInt => ()
      case expr.TLong => ()
      case expr.TDouble => ()
      case expr.TFloat => ()
      case expr.TSet(expr.TInt) => ()
      case expr.TSet(expr.TString) => ()
      case expr.TBoolean => ()
      case expr.TChar => ()
      case _ =>
        fatal(s"parse error in field '$key'.  Programmatic annotations do not allow the type '$exprType'")
    }
  }
}