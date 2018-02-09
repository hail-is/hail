package is.hail.methods

import is.hail.HailContext
import is.hail.expr.types._
import is.hail.table.Table
import is.hail.utils._
import is.hail.variant.CopyState._
import is.hail.variant.GenotypeType._
import is.hail.variant._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

case class MendelError(contig: String, position: Int, alleles: IndexedSeq[String], trio: CompleteTrio, code: Int,
  gtKid: GenotypeType, gtDad: GenotypeType, gtMom: GenotypeType) {
  val ref = alleles(0)
  val alt = alleles(1)

  def gtString(gt: GenotypeType): String =
    if (gt == HomRef)
      ref + "/" + ref
    else if (gt == Het)
      ref + "/" + alt
    else if (gt == HomVar)
      alt + "/" + alt
    else
      "./."

  def implicatedSamplesWithCounts: Iterator[(String, (Long, Long))] = {
    if (code == 2 || code == 1) Iterator(trio.kid, trio.knownDad, trio.knownMom)
    else if (code == 6 || code == 3 || code == 11 || code == 12) Iterator(trio.kid, trio.knownDad)
    else if (code == 4 || code == 7 || code == 9 || code == 10) Iterator(trio.kid, trio.knownMom)
    else Iterator(trio.kid)
  }
    .map((_, (1L, if (AltAlleleMethods.isSNP(alleles(0), alleles(1))) 1L else 0L)))

  def toLineMendel(sampleIds: IndexedSeq[String]): String = {
    val t = trio
    val sb = new StringBuilder()
    sb.append(t.fam.getOrElse("0"))
    sb.append('\t')
    sb.append(t.kid)
    sb.append('\t')
    sb.append(contig)
    sb.append('\t')
    sb.append(Variant.variantID(contig, position, alleles))
    sb.append('\t')
    sb.append(code)
    sb.append('\t')
    sb.append(errorString)
    sb.result()
  }

  def errorString: String =
    s"${ gtString(gtDad) } x ${ gtString(gtMom) } -> ${ gtString(gtKid) }"
}

object MendelErrors {

  def getCode(probandGt: Int, motherGt: Int, fatherGt: Int, copyState: CopyState): Int =
    (fatherGt, motherGt, probandGt, copyState) match {
      case (0, 0, 1, Auto) => 2 // Kid is Het
      case (2, 2, 1, Auto) => 1
      case (0, 0, 2, Auto) => 5 // Kid is HomVar
      case (0, _, 2, Auto) => 3
      case (_, 0, 2, Auto) => 4
      case (2, 2, 0, Auto) => 8 // Kid is HomRef
      case (2, _, 0, Auto) => 6
      case (_, 2, 0, Auto) => 7
      case (_, 2, 0, HemiX) => 9 // Kid is hemizygous
      case (_, 0, 2, HemiX) => 10
      case (2, _, 0, HemiY) => 11
      case (0, _, 2, HemiY) => 12
      case _ => 0 // No error
    }

  def apply(vds: MatrixTable, preTrios: IndexedSeq[CompleteTrio]): MendelErrors = {
    vds.requireUniqueSamples("mendel_errors")
    vds.requireRowKeyVariant("mendel_errors")

    val grLocal = vds.genomeReference

    val trios = preTrios.filter(_.sex.isDefined)
    val nSamplesDiscarded = preTrios.size - trios.size

    if (nSamplesDiscarded > 0)
      warn(s"$nSamplesDiscarded ${ plural(nSamplesDiscarded, "sample") } discarded from .fam: sex of child is missing.")

    val trioMatrix = vds.trioMatrix(Pedigree(trios), completeTrios = true)
    val fullRowType = trioMatrix.rvRowType
    val nTrios = trioMatrix.numCols

    val sc = vds.sparkContext
    val triosBc = sc.broadcast(trios)
    // all trios have defined sex, see filter above
    val trioSexBc = sc.broadcast(trios.map(_.sex.get))

    val localRowType = vds.rowType
    new MendelErrors(vds.hc, vds.rowKeyStruct, trios, vds.stringSampleIds,
      trioMatrix.rvd.mapPartitions { it =>
        val view = new HardcallTrioGenotypeView(fullRowType, "GT")
        val variantView = new RegionValueVariant(localRowType)
        it.flatMap { rv =>
          view.setRegion(rv)
          variantView.setRegion(rv)
          Iterator.range(0, nTrios).flatMap { i =>
            view.setGenotype(i)
            val probandGt = if (view.hasProbandGT) view.getProbandGT else -1
            val motherGt = if (view.hasMotherGT) view.getMotherGT else -1
            val fatherGt = if (view.hasFatherGT) view.getFatherGT else -1
            val code = getCode(probandGt, motherGt, fatherGt, grLocal.copyState(trioSexBc.value(i), Locus(variantView.contig(), variantView.position())))
            if (code != 0)
              Some(MendelError(variantView.contig(), variantView.position(), variantView.alleles(),
                triosBc.value(i), code,
                GenotypeType(probandGt), GenotypeType(fatherGt), GenotypeType(motherGt)))
            else
              None
          }
        }

      }.cache()
    )
  }
}

case class MendelErrors(hc: HailContext, vSig: TStruct, trios: IndexedSeq[CompleteTrio],
  sampleIds: IndexedSeq[String],
  mendelErrors: RDD[MendelError]) {

  private val sc = mendelErrors.sparkContext
  private val trioFam = trios.iterator.flatMap(t => t.fam.map(f => (t.kid, f))).toMap
  private val nuclearFams = Pedigree.nuclearFams(trios)

  def nErrorPerVariant: RDD[(Row, Int)] = {
    mendelErrors
      .map(me => Row(Locus(me.contig, me.position), me.alleles))
      .countByValueRDD()
  }

  def nErrorPerNuclearFamily: RDD[((String, String), (Int, Int))] = {
    val parentsRDD = sc.parallelize(nuclearFams.keys.toSeq)
    mendelErrors
      .map(me => ((me.trio.knownDad, me.trio.knownMom), (1, if (AltAlleleMethods.isSNP(me.ref, me.alt)) 1 else 0)))
      .union(parentsRDD.map((_, (0, 0))))
      .reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2))
  }

  def nErrorPerIndiv: RDD[(String, (Long, Long))] = {
    val indivRDD = sc.parallelize(trios.flatMap(t => Iterator(t.kid, t.knownDad, t.knownMom)).distinct)
    mendelErrors
      .flatMap(_.implicatedSamplesWithCounts)
      .union(indivRDD.map((_, (0L, 0L))))
      .reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2))
  }

  def mendelKT(): Table = {
    val signature = TStruct(
      "fam_id" -> TString(),
      "s" -> TString(),
      "locus" -> vSig.fieldType(0),
      "alleles" -> vSig.fieldType(1),
      "code" -> TInt32(),
      "error" -> TString())

    val rdd = mendelErrors.map { e => Row(e.trio.fam.orNull, e.trio.kid, Locus(e.contig, e.position),
      e.alleles, e.code, e.errorString) }

    Table(hc, rdd, signature, Array("s", "locus", "alleles"))
  }

  def fMendelKT(): Table = {
    val signature = TStruct(
      "fam_id" -> TString(),
      "pat_id" -> TString(),
      "mat_id" -> TString(),
      "children" -> TInt32(),
      "errors" -> TInt32(),
      "snp_errors" -> TInt32()
    )

    val trioFamBc = sc.broadcast(trioFam)
    val nuclearFamsBc = sc.broadcast(nuclearFams)

    val rdd = nErrorPerNuclearFamily.map { case ((dad, mom), (n, nSNP)) =>
      val kids = nuclearFamsBc.value.get((dad, mom))

      Row(kids.flatMap(x => trioFamBc.value.get(x.head)).orNull, dad, mom, kids.map(_.length).getOrElse(0), n, nSNP)
    }

    Table(hc, rdd, signature, Array("pat_id", "mat_id"))
  }

  def iMendelKT(): Table = {

    val signature = TStruct(
      "fam_id" -> TString(),
      "s" -> TString(),
      "errors" -> TInt64(),
      "snp_errors" -> TInt64()
    )

    val trioFamBc = sc.broadcast(trios.iterator.flatMap { t =>
      t.fam.toArray.flatMap { f =>
        Iterator(t.kid -> f) ++ Iterator(t.dad, t.mom).flatten.map(x => x -> f)
      }
    }.toMap)

    val rdd = nErrorPerIndiv.map { case (s, (n, nSNP)) => Row(trioFamBc.value.getOrElse(s, null), s, n, nSNP) }

    Table(hc, rdd, signature, Array("s"))
  }

  def lMendelKT(): Table = {
    val signature = TStruct(
      "locus" -> vSig.fieldType(0),
      "alleles" -> vSig.fieldType(1),
      "errors" -> TInt32()
    )

    val rdd = nErrorPerVariant.map { case (v, l) => Row(v.getAs[Locus](0), v.getAs[IndexedSeq[String]](1), l.toInt) }

    Table(hc, rdd, signature, Array("locus", "alleles"))
  }
}
