package org.broadinstitute.hail.methods

import java.io.Serializable

import scala.reflect.ClassTag
import scala.reflect.runtime.universe._

class Evaluator[T](t: String, treeMap: (Tree) => Tree)(implicit tct: ClassTag[T]) extends Serializable {
  @transient var p: Option[T] = None

  def this(t: String)(implicit tct: ClassTag[T]) = this(t, Map.empty)

  def typeCheck() {
    require(p.isEmpty)
    try
      p = Some(Evaluator.eval[T](t, treeMap))
    catch {
      case e: scala.tools.reflect.ToolBoxError =>
        org.broadinstitute.hail.Utils.fatal("parse error in condition: " + e.message.dropWhile(_ != ':').tail)
    }
  }

  def eval(): T = p match {
    case null | None =>
      val v = Evaluator.eval[T](t, treeMap)
      p = Some(v)
      v
    case Some(v) => v
  }
}

class EvaluatorWithValueTransform[T, S](t: String, f: T => S, treeMap: (Tree) => Tree)(implicit tct: ClassTag[T]) extends Serializable {
  @transient var p: Option[S] = None

  def this(t: String, f: T => S)(implicit tct: ClassTag[T]) = this(t, f, Map.empty)
  
  def typeCheck() {
    require(p.isEmpty)
    try
      p = Some(f(Evaluator.eval[T](t, treeMap)))
    catch {
      case e: scala.tools.reflect.ToolBoxError =>
        org.broadinstitute.hail.Utils.fatal("parse error in condition:" + e.message.dropWhile(_ != ':').tail)
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


object Evaluator {
  import scala.reflect.runtime.currentMirror
  import scala.tools.reflect.ToolBox

//  def apply[T](t: String): EvaluatorWithValueAndTreeTransform[T, T] = new EvaluatorWithValueAndTreeTransform[T, T](t, identity, None)
//  def apply[T, S](t: String, f: T => S): EvaluatorWithValueAndTreeTransform[T, S]
//  def apply[T](t: String, nameMap: Map[String, String])
//  def apply[T, S](t: String, f: T => S, nameMap: Map[String, String])

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

class SymbolRenamer(nameMap: Map[String, String]) extends Transformer with Serializable {
  override def transform(t: Tree): Tree = t match {
    case Select(exp, TermName(n)) =>
      nameMap.get(n) match {
        case Some(newName) => Select(transform(exp), TermName(newName))
        case None => super.transform(t)
      }
    case _ => super.transform(t)
  }
}