package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object AnnotateVariants extends Command {

  def newOptions: Options = throw new UnsupportedOperationException

  def name = "annotatevariants"

  def description = "Annotate variants in current dataset"

  override def lookup(args: Array[String]): (Command, Array[String]) = {
    val errorString =
      """parse error: expect one of the following:
        |  annotatevariants expr <args>
        |  annotatevariants tsv <args>
        |  annotatevariants vcf <args>
        |  annotatevariants vds <args>
        |  annotatevariants bed <args>
        |  annotatevariants intervals <args>
      """.stripMargin
    fatalIf(args.isEmpty, errorString)
    args.head match {
      case "expr" => (AnnotateVariantsExpr, args.tail)
      case "bed" => (AnnotateVariantsBed, args.tail)
      case "tsv" => (AnnotateVariantsTSV, args.tail)
      case "vcf" => (AnnotateVariantsVCF, args.tail)
      case "vds" => (AnnotateVariantsVDS, args.tail)
      case "intervals" => (AnnotateVariantsIList, args.tail)
      case _ => fatal(errorString)
    }
  }

  override def runCommand(state: State, options: Options): State = throw new UnsupportedOperationException

  override def run(state: State, args: Array[String]): State = {
    val (c, newArgs) = lookup(args)
    c.run(state, newArgs)
  }

  override def run(state: State, options: Options): State = throw new UnsupportedOperationException

}
