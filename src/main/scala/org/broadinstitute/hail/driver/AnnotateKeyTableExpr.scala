package org.broadinstitute.hail.driver

import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr.{EvalContext, Parser, TStruct}
import org.broadinstitute.hail.utils._
import org.kohsuke.args4j.{Option => Args4jOption}
import org.broadinstitute.hail.keytable.KeyTable

import scala.collection.mutable

object AnnotateKeyTableExpr extends Command {
  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--cond"),
    usage = "Boolean expression for annotating", metaVar = "EXPR")
    var condition: String = _

    @Args4jOption(required = true, name = "-n", aliases = Array("--name"),
      usage = "Name of source key table")
    var name: String = _

    @Args4jOption(required = false, name = "-d", aliases = Array("--dest"),
      usage = "Name of destination key table (can be same as source)")
    var dest: String = _
  }

  def newOptions = new Options

  def name = "annotatekeytable expr"

  def description = "Annotate key table using an expression"

  def supportsMultiallelic = true

  def requiresVDS = false

  override def hidden = true

  def run(state: State, options: Options): State = {
    val cond = options.condition
    val dest = options.dest

    val kt = state.ktEnv.get(options.name) match {
      case Some(newKT) =>
        newKT
      case None =>
        fatal("no such key table $name in environment")
    }

    val symTab = kt.fields.zipWithIndex.map{case (fd, i) => (fd.name, (i, fd.`type`))}.toMap
    val ec = EvalContext(symTab)

    val (parseTypes, fns) = Parser.parseAnnotationArgs(cond, ec, None)

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]

    val finalValueSignature = parseTypes.foldLeft(kt.valueSignature) { case (vs, (ids, signature)) =>
      val (s: TStruct, i) = vs.insert(signature, ids)
      inserterBuilder += i
      s
    }

    val inserters = inserterBuilder.result()
    val nKeys = kt.nKeys

    val annotated = kt.mapAnnotations{ case (k, v) =>
      KeyTable.setEvalContext(ec, k, v, nKeys)

      fns.zip(inserters)
        .foldLeft(v) { case (va, (fn, inserter)) =>
          inserter(va, Option(fn()))
        }
    }.copy(valueSignature = finalValueSignature)

    state.copy(ktEnv = state.ktEnv + ( dest -> annotated))
  }
}
