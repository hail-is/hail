package org.broadinstitute.hail.driver

import org.broadinstitute.hail.expr._
import org.json4s.jackson.JsonMethods._

object ShowGlobals extends Command {

  class Options extends BaseOptions

  def newOptions = new Options

  def name = "showglobals"

  def description = "Show all global annotations"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val json = vds.globalSignature.toJSON(vds.globalAnnotation)
    println(
      if (vds.globalSignature == TStruct.empty)
        "Global annotations: `global' = Empty"
      else
        "Global annotations: `global' = " + pretty(render(json)))

    state
  }
}
