package is.hail.driver

import is.hail.utils._
import is.hail.annotations._
import is.hail.expr._
import is.hail.methods.Aggregators
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object AnnotateAllelesExpr extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Annotation expression")
    var condition: String = _

    @Args4jOption(required = false, name = "--propagate-gq", usage = "Propagate GQ instead of computing from PL when splitting alleles")
    var propagateGQ: Boolean = false

  }

  def newOptions = new Options

  def name = "annotatealleles expr"

  def description = "Annotate alleles programatically"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val cond = options.condition
    val propagateGQ = options.propagateGQ
    val isDosage = vds.isDosage

    val (vas2, insertIndex) = vds.vaSignature.insert(TInt, "aIndex")
    val (vas3, insertSplit) = vas2.insert(TBoolean, "wasSplit")
    val localGlobalAnnotation = vds.globalAnnotation

    val aggregationST = Map(
      "global" -> (0, vds.globalSignature),
      "v" -> (1, TVariant),
      "va" -> (2, vas3),
      "g" -> (3, TGenotype),
      "s" -> (4, TSample),
      "sa" -> (5, vds.saSignature))
    val ec = EvalContext(Map(
      "global" -> (0, vds.globalSignature),
      "v" -> (1, TVariant),
      "va" -> (2, vas3),
      "gs" -> (3, TAggregable(TGenotype, aggregationST))))

    val (paths, types, f) = Parser.parseAnnotationExprs(cond, ec, Some(Annotation.VARIANT_HEAD))

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
    val finalType = (paths, types).zipped.foldLeft(vds.vaSignature) { case (vas, (ids, signature)) =>
      val (s, i) = vas.insert(TArray(signature), ids)
      inserterBuilder += i
      s
    }
    val inserters = inserterBuilder.result()

    val aggregateOption = Aggregators.buildVariantAggregations(vds, ec)

    val annotated = vds.mapAnnotations { case (v, va, gs) =>

      val annotations = SplitMulti.split(v, va, gs,
        propagateGQ = propagateGQ,
        compress = true,
        keepStar = true,
        isDosage = isDosage,
        insertSplitAnnots = { (va, index, wasSplit) =>
          insertSplit(insertIndex(va, Some(index)), Some(wasSplit))
        })
        .map({
          case (v,(va,gs)) =>
            ec.setAll(localGlobalAnnotation, v, va)
            aggregateOption.foreach(f => f(v, va, gs))
            f()
        }).toArray

      inserters.zipWithIndex.foldLeft(va){
        case (va,(inserter, i)) =>
          inserter(va, Some(annotations.map(_(i).getOrElse(Annotation.empty)).toArray[Any]: IndexedSeq[Any]))
      }

    }.copy(vaSignature = finalType)
    state.copy(vds = annotated)
  }
}
