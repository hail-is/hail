package is.hail.driver

import is.hail.annotations.Annotation
import is.hail.utils._

object Typecheck extends Command {

  class Options extends BaseOptions

  def newOptions = new Options

  def name = "typecheck"

  def description = "check if all sample, variant, and global annotations agree with stored schema"

  def supportsMultiallelic = true

  def requiresVDS = true

  override def hidden: Boolean = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if (!vds.globalSignature.typeCheck(vds.globalAnnotation))
      warn(
        s"""found violation in global annotation
            |Schema: ${ vds.globalSignature.toPrettyString() }
            |
            |Annotation: ${ Annotation.printAnnotation(vds.globalAnnotation) }""".stripMargin)

    vds.sampleIdsAndAnnotations.find { case (_, sa) => !vds.saSignature.typeCheck(sa) }
      .foreach { case (s, sa) =>
        warn(
          s"""found violation in sample annotations for sample $s
              |Schema: ${ vds.saSignature.toPrettyString() }
              |
              |Annotation: ${ Annotation.printAnnotation(sa) }""".stripMargin)
      }

    val vaSignature = vds.vaSignature
    vds.variantsAndAnnotations.find { case (_, va) => !vaSignature.typeCheck(va) }
      .foreach { case (v, va) =>
        warn(
          s"""found violation in variant annotations for variant $v
              |Schema: ${ vaSignature.toPrettyString() }
              |
              |Annotation: ${ Annotation.printAnnotation(va) }""".stripMargin)
      }

    state
  }
}
