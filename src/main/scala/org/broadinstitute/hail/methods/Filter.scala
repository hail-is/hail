package org.broadinstitute.hail.methods

import scala.reflect.runtime.currentMirror
import scala.tools.reflect.ToolBox
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant.{Genotype, Variant}
import scala.reflect.ClassTag
import scala.language.implicitConversions

class FilterString(val s: String) extends AnyVal {
  def ~(t: String): Boolean = s.r.findFirstIn(t).isDefined
  def !~(t: String): Boolean = !this.~(t)
}

object FilterUtils {
  implicit def toFilterString(s: String): FilterString = new FilterString(s)
}

class ConditionPredicate[T](sym: String,
  condition: String)(implicit tct: ClassTag[T]) extends Serializable {
  @transient var p: (T) => Boolean = null

  def compile(typeCheck: Boolean) {
    if (p == null) {
      printTime {
        val toolbox = currentMirror.mkToolBox()
        val ast = toolbox.parse("(" + sym + ": " + tct.runtimeClass.getCanonicalName + ") => { " +
          "import org.broadinstitute.hail.driver.FilterUtils._; " + condition + " }: Boolean")
        if (typeCheck)
          toolbox.typeCheck(ast)
        p = toolbox.eval(ast).asInstanceOf[(T) => Boolean]
      }
    }
  }

  def apply(v: T): Boolean = {
    compile(false)
    val r = p(v)
    // println("v = " + v + " " + r)
    r
  }
}

class GenotypeConditionPredicate[T](condition: String) extends Serializable {
  @transient var p: (Variant, Int, Genotype) => Boolean = null

  def compile(typeCheck: Boolean) {
    if (p == null) {
      printTime {
        val toolbox = currentMirror.mkToolBox()
        val ast = toolbox.parse("(v: org.broadinstitute.hail.variant.Variant, " +
          "s: Int, " +
          "g: org.broadinstitute.hail.variant.Genotype) => { " +
          "import org.broadinstitute.hail.driver.FilterUtils._; " +
          condition + " }: Boolean")
        if (typeCheck)
          toolbox.typeCheck(ast)
        p = toolbox.eval(ast).asInstanceOf[(Variant, Int, Genotype) => Boolean]
      }
    }
  }

  def apply(v: Variant, s: Int, g: Genotype): Boolean = {
    compile(false)
    val r = p(v, s, g)
    // println("v = " + v + " " + r)
    r
  }
}
