package is.hail.driver

import is.hail.annotations._
import is.hail.expr._
import is.hail.methods.Aggregators
import is.hail.variant.Variant
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object AnnotateGlobalExprByVariant extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Annotation expression")
    var condition: String = _
  }

  def newOptions = new Options

  def name = "annotateglobal exprbyvariant"

  def description = "Use the Hail Expression Language to compute new annotations from existing global annotations, as well as perform variant aggregations."

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val localGlobalAnnotation = vds.globalAnnotation

    val cond = options.condition

    val aggregationST = Map(
      "global" -> (0, vds.globalSignature),
      "v" -> (1, TVariant),
      "va" -> (2, vds.vaSignature))
    val ec = EvalContext(Map(
      "global" -> (0, vds.globalSignature),
      "variants" -> (1, TAggregable(TVariant, aggregationST))))

    val (paths, types, f) = Parser.parseAnnotationExprs(cond, ec, Some(Annotation.GLOBAL_HEAD))

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]

    val finalType = (paths, types).zipped.foldLeft(vds.globalSignature) { case (v, (ids, signature)) =>
      val (s, i) = v.insert(signature, ids)
      inserterBuilder += i
      s
    }

    val inserters = inserterBuilder.result()

    val (zVal, seqOp, combOp, resOp) = Aggregators.makeFunctions[(Variant, Annotation)](ec, { case (ec, (v, va)) =>
      ec.setAll(localGlobalAnnotation, v, va)
    })

    val result = vds.variantsAndAnnotations
      .treeAggregate(zVal)(seqOp, combOp, depth = HailConfiguration.treeAggDepth(vds.nPartitions))
    resOp(result)

    ec.setAll(localGlobalAnnotation)
    val ga = inserters
      .zip(f())
      .foldLeft(vds.globalAnnotation) { case (a, (ins, res)) =>
        ins(a, res)
      }

    state.copy(vds = vds.copy(
      globalAnnotation = ga,
      globalSignature = finalType)
    )
  }
}

object AnnotateGlobalExprBySample extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Annotation expression")
    var condition: String = _
  }

  def newOptions = new Options

  def name = "annotateglobal exprbysample"

  def description = "Use the Hail Expression Language to compute new annotations from existing global annotations, as well as perform sample aggregations."

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val localGlobalAnnotation = vds.globalAnnotation

    val cond = options.condition

    val aggregationST = Map(
      "global" -> (0, vds.globalSignature),
      "s" -> (1, TVariant),
      "sa" -> (2, vds.saSignature))
    val ec = EvalContext(Map(
      "global" -> (0, vds.globalSignature),
      "samples" -> (1, TAggregable(TSample, aggregationST))))

    val (paths, types, f) = Parser.parseAnnotationExprs(cond, ec, Option(Annotation.GLOBAL_HEAD))

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]

    val finalType = (paths, types).zipped.foldLeft(vds.globalSignature) { case (v, (ids, signature)) =>
      val (s, i) = v.insert(signature, ids)
      inserterBuilder += i
      s
    }

    val inserters = inserterBuilder.result()

    val (zVal, seqOp, combOp, resOp) = Aggregators.makeFunctions[(String, Annotation)](ec, { case (ec, (s, sa)) =>
      ec.setAll(localGlobalAnnotation, s, sa)
    })

    val result = vds.sampleIdsAndAnnotations
      .aggregate(zVal)(seqOp, combOp)
    resOp(result)

    ec.setAll(localGlobalAnnotation)
    val ga = inserters
      .zip(f())
      .foldLeft(vds.globalAnnotation) { case (a, (ins, res)) =>
        ins(a, res)
      }

    state.copy(vds = vds.copy(
      globalAnnotation = ga,
      globalSignature = finalType))
  }
}

