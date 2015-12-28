package org.broadinstitute.hail.methods

import java.io.Serializable

import scala.reflect.ClassTag
import scala.reflect.runtime.universe._

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
        org.broadinstitute.hail.Utils.fatal("parse error in condition: " + e.message) //FIXME not quite right / add to other evaluators?
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

class EvaluatorWithValueTransform[T, S](t: String, f: T => S)(implicit tct: ClassTag[T]) extends Serializable {
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

class EvaluatorWithTreeTransform[T](t: String, nameMap: Map[String, String])(implicit tct: ClassTag[T])
  extends Serializable {
  @transient var p: Option[T] = None

  def typeCheck() {
    require(p.isEmpty)
    p = Some(EvaluatorWithTreeTransform.eval[T](t, nameMap))
  }

  def eval(): T = p match {
    case null | None =>
      val v = EvaluatorWithTreeTransform.eval[T](t, nameMap)
      p = Some(v)
      v
    case Some(v) => v
  }
}

class EvaluatorWithValueAndTreeTransform[T, S](t: String, f: T => S, nameMap: Map[String, String])(implicit tct: ClassTag[T]) extends Serializable {
  @transient var p: Option[S] = None

  def typeCheck() {
    require(p.isEmpty)
    p = Some(f(EvaluatorWithTreeTransform.eval[T](t, nameMap)))
  }

  def eval(): S = p match {
    case null | None =>
      val v = f(EvaluatorWithTreeTransform.eval[T](t, nameMap))
      p = Some(v)
      v
    case Some(v) => v
  }
}

object Evaluator {
  import scala.reflect.runtime.currentMirror
  import scala.tools.reflect.ToolBox

  def eval[T](t: String): T = {
    //println(s"t = $t")
    val toolbox = currentMirror.mkToolBox()
    val ast = toolbox.parse(t)
    toolbox.typeCheck(ast)
    toolbox.eval(ast).asInstanceOf[T]
  }
}

object EvaluatorWithTreeTransform {
  import scala.reflect.runtime.currentMirror
  import scala.tools.reflect.ToolBox

  def eval[T](t: String, nameMap: Map[String, String]): T = {
    //println(s"t = $t")
    val toolbox = currentMirror.mkToolBox()
    val ast = new TreeTransformer(nameMap).transform(toolbox.parse(t))
    toolbox.typeCheck(ast)
    toolbox.eval(ast).asInstanceOf[T]
  }
}

class TreeTransformer(nameMap: Map[String, String]) extends Transformer {
  override def transform(t: Tree): Tree = t match {
    case Select(exp, TermName(n)) =>
      nameMap.get(n) match {
        case Some(newName) => Select(transform(exp), TermName(newName))
        case None => super.transform(t)
      }
    case _ => super.transform(t)
  }
}