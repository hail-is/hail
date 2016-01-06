package org.broadinstitute.hail.methods

import java.io.Serializable

import scala.reflect.ClassTag
import scala.reflect.runtime.universe._
import org.broadinstitute.hail.Utils._

class EvaluatorWithValueTransform[T, S](t: String, f: T => S, treeMap: (Tree) => Tree)(implicit tct: ClassTag[T]) extends Serializable {
  @transient var p: Option[S] = None

  def this(t: String, f: T => S)(implicit tct: ClassTag[T]) = this(t, f, Map.empty)
  
  def typeCheck() {
    require(p.isEmpty)
    try
      p = Some(f(Evaluator.eval[T](t, treeMap)))
    catch {
      case e: scala.tools.reflect.ToolBoxError =>
        fatal("parse error in condition:" + e.message.dropWhile(_ != ':').tail)
    }
  }

  def eval(): S = p match {
    case null | None =>
      val v = f(Evaluator.eval[T](t, treeMap))
      p = Some(v)
      v
    case Some(v) => v
  }
}

class Evaluator[T](t: String, treeMap: (Tree) => Tree)(implicit tct: ClassTag[T]) extends EvaluatorWithValueTransform[T,T](t, identity, treeMap) {

  def this(t: String)(implicit tct: ClassTag[T]) = this(t, Map.empty)
}

object Evaluator {
  import scala.reflect.runtime.currentMirror
  import scala.tools.reflect.ToolBox

  def eval[U](t: String): U = {
    // println(s"t = $t")
    val toolbox = currentMirror.mkToolBox()
    val ast = toolbox.parse(t)
    toolbox.typeCheck(ast)
    toolbox.eval(ast).asInstanceOf[U]
  }

  def eval[U](t: String, treeMap: (Tree) => Tree): U = {
    // println(s"t = $t")
    val toolbox = currentMirror.mkToolBox()
    val ast = treeMap(toolbox.parse(t))
    toolbox.typeCheck(ast)
    toolbox.eval(ast).asInstanceOf[U]
  }
}
