package org.broadinstitute.hail.driver.keytable

import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.driver.{Command, State}
import org.broadinstitute.hail.expr.{EvalContext, Parser}
import org.broadinstitute.hail.utils._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object AnnotateKeyTable extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-n", aliases = Array("--name"),
      usage = "Key table name")
    var name: String = _

    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Annotation expression")
    var condition: String = _
  }

  def newOptions = new Options

  def name = "annotatekeytable expr"

  def description = "Annotate key table programatically"

  def supportsMultiallelic = true

  def requiresVDS = true

  override def hidden = true

  def run(state: State, options: Options): State = {
    val kt = state.ktEnv.get(options.name) match {
      case Some(newKT) =>
        newKT
      case None =>
        fatal("no such key table $name in environment")
    }

    val cond = options.condition

    val symTab = Map(
      kt.keyIdentifier -> (0, kt.keySignature),
      kt.valueIdentifier -> (1, kt.valueSignature),
      "global" -> (2, state.vds.globalSignature)
    )

    val ec = EvalContext(symTab)

    ec.set(2, state.vds.globalAnnotation)

    val (parseTypes, fns) = Parser.parseAnnotationArgs(cond, ec, kt.valueIdentifier)

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
    val finalType = parseTypes.foldLeft(kt.valueSignature) { case (kas, (ids, signature)) =>
      val (s, i) = kas.insert(signature, ids)
      inserterBuilder += i
      s
    }
    val inserters = inserterBuilder.result()

    val annotated = kt.mapAnnotations { case (k, ka) =>
      ec.setAll(k, ka)

      fns.zip(inserters)
        .foldLeft(ka) { case (a, (fn, inserter)) =>
          inserter(a, fn())
        }
    }.copy(valueSignature = finalType)

    state.copy(ktEnv = state.ktEnv + (options.name -> kt))

    state
  }
}
