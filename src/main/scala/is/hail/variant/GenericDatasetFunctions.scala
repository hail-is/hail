package is.hail.variant

import is.hail.annotations._
import is.hail.expr.{EvalContext, Parser, TGenotype, Type}
import is.hail.methods.Filter
import is.hail.sparkextras.OrderedKey
import is.hail.utils._

import scala.collection.mutable

class GenericDatasetFunctions[RPK, RK](private val vsm: VariantSampleMatrix[RPK, RK, Annotation]) {
  implicit val kOk: OrderedKey[RPK, RK] = vsm.kOk

  def annotateGenotypesExpr(expr: String): VariantSampleMatrix[RPK, RK, Annotation] = {
    val symTab = Map(
      "v" -> (0, vsm.vSignature),
      "va" -> (1, vsm.vaSignature),
      "s" -> (2, vsm.sSignature),
      "sa" -> (3, vsm.saSignature),
      "g" -> (4, vsm.genotypeSignature),
      "global" -> (5, vsm.globalSignature))

    val ec = EvalContext(symTab)
    ec.set(5, vsm.globalAnnotation)

    val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, Some(Annotation.GENOTYPE_HEAD))

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
    val finalType = (paths, types).zipped.foldLeft(vsm.genotypeSignature) { case (gsig, (ids, signature)) =>
      val (s, i) = gsig.insert(signature, ids)
      inserterBuilder += i
      s
    }
    val inserters = inserterBuilder.result()

    info(
      s"""Modified the genotype schema with annotateGenotypesExpr.
         |  Original: ${ vsm.genotypeSignature.toPrettyString(compact = true) }
         |  New: ${ finalType.toPrettyString(compact = true) }""".stripMargin)

    vsm.mapValuesWithAll(
      (v: Annotation, va: Annotation, s: Annotation, sa: Annotation, g: Annotation) => {
        ec.setAll(v, va, s, sa, g)
        f().zip(inserters)
          .foldLeft(g) { case (ga, (a, inserter)) =>
            inserter(ga, a)
          }
      }).copy(genotypeSignature = finalType, isGenericGenotype = true)
  }

  def exportGenotypes(path: String, expr: String, typeFile: Boolean, printMissing: Boolean = false) {
    val localPrintMissing = printMissing
    val filterF: Annotation => Boolean = g => g != null || localPrintMissing

    vsm.exportGenotypes(path, expr, typeFile, filterF)
  }

  /**
    *
    * @param filterExpr filter expression involving v (Variant), va (variant annotations), s (sample),
    * sa (sample annotations), and g (genotype annotation), which returns a boolean value
    * @param keep keep genotypes where filterExpr evaluates to true
    */
  def filterGenotypes(filterExpr: String, keep: Boolean = true): VariantSampleMatrix[RPK, RK, Annotation] = {

    val symTab = Map(
      "v" -> (0, vsm.vSignature),
      "va" -> (1, vsm.vaSignature),
      "s" -> (2, vsm.sSignature),
      "sa" -> (3, vsm.saSignature),
      "g" -> (4, vsm.genotypeSignature),
      "global" -> (5, vsm.globalSignature))


    val ec = EvalContext(symTab)
    ec.set(5, vsm.globalAnnotation)
    val f: () => java.lang.Boolean = Parser.parseTypedExpr[java.lang.Boolean](filterExpr, ec)

    val localKeep = keep
    vsm.mapValuesWithAll(
      (v: Annotation, va: Annotation, s: Annotation, sa: Annotation, g: Annotation) => {
        ec.setAll(v, va, s, sa, g)

        if (Filter.boxedKeepThis(f(), localKeep))
          g
        else
          null
      })
  }

  def queryGA(code: String): (Type, Querier) = {

    val st = Map(Annotation.GENOTYPE_HEAD -> (0, vsm.genotypeSignature))
    val ec = EvalContext(st)
    val a = ec.a

    val (t, f) = Parser.parseExpr(code, ec)

    val f2: Annotation => Any = { annotation =>
      a(0) = annotation
      f()
    }

    (t, f2)
  }
}
