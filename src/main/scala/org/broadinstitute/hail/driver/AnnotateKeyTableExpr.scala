package org.broadinstitute.hail.driver

import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr.{EvalContext, Parser, TStruct, Type}
import org.broadinstitute.hail.utils._
import org.kohsuke.args4j.{Option => Args4jOption}
import org.broadinstitute.hail.keytable.KeyTable

import scala.collection.mutable

object AnnotateKeyTableExpr extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-n", aliases = Array("--name"),
      usage = "Name of source key table")
    var name: String = _

    @Args4jOption(required = false, name = "-d", aliases = Array("--dest"),
      usage = "Name of destination key table (can be same as source)")
    var dest: String = _

    @Args4jOption(required = false, name = "-c", aliases = Array("--cond"),
      usage = "Named expression for adding fields to the table", metaVar = "EXPR")
    var condition: String = _

    @Args4jOption(required = false, name = "-k", aliases = Array("--key-names"),
      usage = "Names of key in new table (default is existing key names)", metaVar = "EXPR")
    var keyNames: String = _
  }

  def newOptions = new Options

  def name = "annotatekeytable expr"

  def description = "Annotate key table using an expression"

  def supportsMultiallelic = true

  def requiresVDS = false

  override def hidden = true

  def run(state: State, options: Options): State = {
    val cond = options.condition
    val name = options.name
    val dest = if (options.dest != null) options.dest else name

    val kt = state.ktEnv.get(name) match {
      case Some(newKT) =>
        newKT
      case None =>
        fatal("no such key table $name in environment")
    }

    val ec = EvalContext(kt.fields.map(fd => (fd.name, fd.`type`)): _*)

    val (parseTypes, fns) =
      if (cond != null)
        Parser.parseAnnotationArgs(cond, ec, None)
      else
        (Array.empty[(List[String], Type)], Array.empty[() => Any])

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]

    val finalSignature = parseTypes.foldLeft(kt.signature) { case (vs, (ids, signature)) =>
      val (s: TStruct, i) = vs.insert(signature, ids)
      inserterBuilder += i
      s
    }

    val inserters = inserterBuilder.result()

    val keyNames = if (options.keyNames != null) Parser.parseIdentifierList(options.keyNames) else kt.keyNames.toArray

    val nFields = kt.nFields

    val f: Annotation => Annotation = { a =>
      KeyTable.setEvalContext(ec, a, nFields)

      fns.zip(inserters)
        .foldLeft(a) { case (a1, (fn, inserter)) =>
          inserter(a1, Option(fn()))
        }
    }

    state.copy(ktEnv = state.ktEnv + (dest -> KeyTable(kt.mapAnnotations(f), finalSignature, keyNames)))
  }
}
