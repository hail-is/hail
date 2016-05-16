package org.broadinstitute.hail.driver.example

import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.driver._

/* Example command to count the number of non-ref alternate allele calls in cases vs controls per variant.
   Case/control status is stored in `sa.case: Boolean'.  The result is stored in variant annotations as `va.nCase'
   and `va.nControl'.  */
object CaseControlCount extends Command {

  class Options extends BaseOptions

  def newOptions = new Options

  def name = "example/casecontrolcount"

  def description = "Count number of alternate alleles in cases vs controls"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    // query sa.case, must be boolean
    val qCase = vds.querySA("sa.case")._2

    // insert nCase, then nControl
    val (tmpVAS, insertCase) = vds.vaSignature.insert(TInt, "nCase")
    val (newVAS, insertControl) = tmpVAS.insert(TInt, "nControl")

    state.copy(vds =
      vds.mapAnnotationsWithAggregate((0, 0), newVAS)({ case ((nCase, nControl), v, va, s, sa, g) =>
        val isCase = qCase(sa)
        (g.nNonRefAlleles, isCase) match {
          case (Some(n), Some(true)) => (nCase + n, nControl)
          case (Some(n), Some(false)) => (nCase, nControl + n)
          case _ => (nCase, nControl)
        }
      }, { case ((nCase1, nControl1), (nCase2, nControl2)) =>
        (nCase1 + nCase2, nControl1 + nControl2)
      }, { case (va, (nCase, nControl)) =>
        // same order as signature insertion above
        insertControl(insertCase(va, Some(nCase)),
          Some(nControl))
      }))
  }
}
