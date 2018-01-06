package is.hail.methods

import is.hail.utils._
import is.hail.variant.{Locus, MatrixTable}

object ImputeSexPlink {
  def apply(in: MatrixTable,
    mafThreshold: Double = 0.0,
    includePar: Boolean = false,
    fFemaleThreshold: Double = 0.2,
    fMaleThreshold: Double = 0.8,
    popFrequencyExpr: Option[String] = None): MatrixTable = {
    var vsm = in

    val gr = vsm.genomeReference

    val xIntervals = IntervalTree(gr.locus.ordering,
      gr.xContigs.map(contig => Interval(Locus(contig, 0), Locus(contig, gr.contigLength(contig)))).toArray)
    vsm = FilterIntervals(vsm, xIntervals, keep = true)

    if (!includePar)
      vsm = FilterIntervals(vsm, IntervalTree(gr.locus.ordering, gr.par), keep = false)

    vsm = vsm.annotateVariantsExpr(
      s"va = ${ popFrequencyExpr.getOrElse("gs.map(g => g.GT.nNonRefAlleles).sum().toFloat64() / gs.filter(g => isDefined(g.GT)).count() / 2") }")

    val resultSA = vsm
      .filterVariantsExpr(s"va > $mafThreshold")
      .annotateSamplesExpr(s"""sa =
let ib = gs.map(g => g.GT).inbreeding(g => va) and
    isFemale = if (ib.Fstat < $fFemaleThreshold) true else if (ib.Fstat > $fMaleThreshold) false else NA: Boolean
 in merge({ isFemale: isFemale }, ib)""")
      .samplesKT()

    in.annotateSamplesTable(resultSA, root = "sa.imputesex")
  }
}
