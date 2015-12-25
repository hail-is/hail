package org.broadinstitute.hail.methods

import java.io.Serializable

import scala.reflect.ClassTag


class EvaluatorWithTransformation[T, S](t: String, f: T => S)(implicit tct: ClassTag[T]) extends Serializable {
  @transient var p: Option[S] = None

  def typeCheck() {
    require(p.isEmpty)
    p = Some(f(Evaluator.eval[T](t)))
  }

  def eval(): S = p match {
    case null | None =>
      val v = f(Evaluator.eval[T](t))
      p = Some(v)
      v
    case Some(v) => v
  }
}

class Evaluator[T](t: String)(implicit tct: ClassTag[T])
  extends Serializable {
  @transient var p: Option[T] = None

  def typeCheck() {
    require(p.isEmpty)
    try {
      p = Some(Evaluator.eval[T](t))
    }
    catch {
      case e: scala.tools.reflect.ToolBoxError =>
        /* e.message looks like:
           reflective compilation has failed:

           ';' expected but '.' found. */
        org.broadinstitute.hail.Utils.fatal("parse error in condition: " + e.message) //FIXME not quite right
    }
  }

  def eval(): T = p match {
    case null | None =>
      val v = Evaluator.eval[T](t)
      p = Some(v)
      v
    case Some(v) => v
  }
}

object Evaluator {
  import scala.reflect.runtime.currentMirror
  import scala.tools.reflect.ToolBox

  val m: Map[String, String] = FilterTransformer.nameMap

  def eval[T](t: String): T = {
    //println(s"t = $t")
    val toolbox = currentMirror.mkToolBox()
    //val ast = toolbox.parse(t)
    val ast = new FilterTransformer(m).transform(toolbox.parse(t))
    toolbox.typeCheck(ast)
    toolbox.eval(ast).asInstanceOf[T]
  }
}