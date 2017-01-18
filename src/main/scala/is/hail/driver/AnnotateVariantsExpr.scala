package is.hail.driver

import is.hail.utils._
import is.hail.annotations._
import is.hail.expr._
import is.hail.methods.Aggregators
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object AnnotateVariantsExpr extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Annotation expression")
    var condition: String = _
  }

  def newOptions = new Options

  def name = "annotatevariants expr"

  def description = "Annotate variants programatically"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val localGlobalAnnotation = vds.globalAnnotation

    val cond = options.condition

    val ec = Aggregators.variantEC(vds)
    val (paths, types, f) = Parser.parseAnnotationExprs(cond, ec, Some(Annotation.VARIANT_HEAD))

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
    val finalType = (paths, types).zipped.foldLeft(vds.vaSignature) { case (vas, (ids, signature)) =>
      val (s, i) = vas.insert(signature, ids)
      inserterBuilder += i
      s
    }
    val inserters = inserterBuilder.result()

    val aggregateOption = Aggregators.buildVariantAggregations(vds, ec)

    val annotated = vds.mapAnnotations { case (v, va, gs) =>
      ec.setAll(localGlobalAnnotation, v, va)

      aggregateOption.foreach(f => f(v, va, gs))
      f().zip(inserters)
        .foldLeft(va) { case (va, (v, inserter)) =>
          inserter(va, v)
        }
    }.copy(vaSignature = finalType)
    state.copy(vds = annotated)
  }
}
