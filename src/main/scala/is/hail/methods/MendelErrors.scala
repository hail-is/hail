package is.hail.methods

import is.hail.HailContext
import is.hail.annotations.Annotation
import is.hail.expr.{TInt32, TString, TStruct, TVariant, Type}
import is.hail.keytable.KeyTable
import is.hail.utils.{MultiArray2, _}
import is.hail.variant.CopyState._
import is.hail.variant.GenotypeType._
import is.hail.variant._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

import scala.collection.mutable

case class MendelError(variant: Variant, trio: CompleteTrio, code: Int,
  gtKid: GenotypeType, gtDad: GenotypeType, gtMom: GenotypeType) {

  def gtString(v: Variant, gt: GenotypeType): String =
    if (gt == HomRef)
      v.ref + "/" + v.ref
    else if (gt == Het)
      v.ref + "/" + v.alt
    else if (gt == HomVar)
      v.alt + "/" + v.alt
    else
      "./."

  def implicatedSamplesWithCounts: Iterator[(String, (Int, Int))] = {
    if (code == 2 || code == 1) Iterator(trio.kid, trio.knownDad, trio.knownMom)
    else if (code == 6 || code == 3 || code == 11 || code == 12) Iterator(trio.kid, trio.knownDad)
    else if (code == 4 || code == 7 || code == 9 || code == 10) Iterator(trio.kid, trio.knownMom)
    else Iterator(trio.kid)
  }
    .map((_, (1, if (variant.altAllele.isSNP) 1 else 0)))

  def toLineMendel(sampleIds: IndexedSeq[String]): String = {
    val v = variant
    val t = trio
    t.fam.getOrElse("0") + "\t" + t.kid + "\t" + v.contig + "\t" +
      v.contig + ":" + v.start + ":" + v.ref + ":" + v.alt + "\t" + code + "\t" + errorString
  }

  def errorString = gtString(variant, gtDad) + " x " + gtString(variant, gtMom) + " -> " + gtString(variant, gtKid)
}

object MendelErrors {

  def getCode(gts: IndexedSeq[GenotypeType], copyState: CopyState): Int = {
    (gts(1), gts(2), gts(0), copyState) match {
      // (gtDad, gtMom, gtKid)
      case (HomRef, HomRef, Het, Auto) => 2 // Kid is Het
      case (HomVar, HomVar, Het, Auto) => 1
      case (HomRef, HomRef, HomVar, Auto) => 5 // Kid is HomVar
      case (HomRef, _, HomVar, Auto) => 3
      case (_, HomRef, HomVar, Auto) => 4
      case (HomVar, HomVar, HomRef, Auto) => 8 // Kid is HomRef
      case (HomVar, _, HomRef, Auto) => 6
      case (_, HomVar, HomRef, Auto) => 7
      case (_, HomVar, HomRef, HemiX) => 9 // Kid is hemizygous
      case (_, HomRef, HomVar, HemiX) => 10
      case (HomVar, _, HomRef, HemiY) => 11
      case (HomRef, _, HomVar, HemiY) => 12
      case _ => 0 // No error
    }
  }

  def apply(vds: VariantDataset, preTrios: IndexedSeq[CompleteTrio]): MendelErrors = {
    vds.requireUniqueSamples("mendel_errors")

    val trios = preTrios.filter(_.sex.isDefined)
    val nSamplesDiscarded = preTrios.size - trios.size

    if (nSamplesDiscarded > 0)
      warn(s"$nSamplesDiscarded ${ plural(nSamplesDiscarded, "sample") } discarded from .fam: sex of child is missing.")

    val sampleTrioRoles = mutable.Map.empty[String, List[(Int, Int)]]
    trios.zipWithIndex.foreach { case (t, ti) =>
      sampleTrioRoles += (t.kid -> sampleTrioRoles.getOrElse(t.kid, List.empty[(Int, Int)]).::(ti, 0))
      sampleTrioRoles += (t.knownDad -> sampleTrioRoles.getOrElse(t.knownDad, List.empty[(Int, Int)]).::(ti, 1))
      sampleTrioRoles += (t.knownMom -> sampleTrioRoles.getOrElse(t.knownMom, List.empty[(Int, Int)]).::(ti, 2))
    }

    val sc = vds.sparkContext
    val sampleTrioRolesBc = sc.broadcast(sampleTrioRoles)
    val triosBc = sc.broadcast(trios)
    // all trios have defined sex, see filter above
    val trioSexBc = sc.broadcast(trios.map(_.sex.get))

    val zeroVal: MultiArray2[GenotypeType] = MultiArray2.fill(trios.length, 3)(NoCall)

    def seqOp(a: MultiArray2[GenotypeType], s: Annotation, g: Genotype): MultiArray2[GenotypeType] = {
      sampleTrioRolesBc.value.get(s.asInstanceOf[String]).foreach(l => l.foreach { case (ti, ri) => a.update(ti, ri, Genotype.gtType(g)) })
      a
    }

    def mergeOp(a: MultiArray2[GenotypeType], b: MultiArray2[GenotypeType]): MultiArray2[GenotypeType] = {
      for ((i, j) <- a.indices)
        if (b(i, j) != NoCall)
          a(i, j) = b(i, j)
      a
    }

    new MendelErrors(vds.hc, vds.vSignature, trios, vds.stringSampleIds,
      vds
        .aggregateByVariantWithKeys(zeroVal)(
          (a, v, s, g) => seqOp(a, s, g),
          mergeOp)
        .flatMap { case (v, a) =>
          a.rows.flatMap { case (row) => val code = getCode(row, v.copyState(trioSexBc.value(row.i)))
            if (code != 0)
              Some(MendelError(v, triosBc.value(row.i), code, row(0), row(1), row(2)))
            else
              None
          }
        }
        .cache()
    )
  }
}

case class MendelErrors(hc: HailContext, vSig: Type, trios: IndexedSeq[CompleteTrio],
  sampleIds: IndexedSeq[String],
  mendelErrors: RDD[MendelError]) {

  private val sc = mendelErrors.sparkContext
  private val trioFam = trios.iterator.flatMap(t => t.fam.map(f => (t.kid, f))).toMap
  private val nuclearFams = Pedigree.nuclearFams(trios)

  def nErrorPerVariant: RDD[(Variant, Int)] = {
    mendelErrors
      .map(_.variant)
      .countByValueRDD()
  }

  def nErrorPerNuclearFamily: RDD[((String, String), (Int, Int))] = {
    val parentsRDD = sc.parallelize(nuclearFams.keys.toSeq)
    mendelErrors
      .map(me => ((me.trio.knownDad, me.trio.knownMom), (1, if (me.variant.altAllele.isSNP) 1 else 0)))
      .union(parentsRDD.map((_, (0, 0))))
      .reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2))
  }

  def nErrorPerIndiv: RDD[(String, (Int, Int))] = {
    val indivRDD = sc.parallelize(trios.flatMap(t => Iterator(t.kid, t.knownDad, t.knownMom)).distinct)
    mendelErrors
      .flatMap(_.implicatedSamplesWithCounts)
      .union(indivRDD.map((_, (0, 0))))
      .reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2))
  }

  def mendelKT(): KeyTable = {
    val signature = TStruct(
      "fid" -> TString(),
      "s" -> TString(),
      "v" -> vSig,
      "code" -> TInt32(),
      "error" -> TString())

    val rdd = mendelErrors.map { e => Row(e.trio.fam.orNull, e.trio.kid, e.variant, e.code, e.errorString) }

    KeyTable(hc, rdd, signature, Array("s", "v"))
  }

  def fMendelKT(): KeyTable = {
    val signature = TStruct(
      "fid" -> TString(),
      "father" -> TString(),
      "mother" -> TString(),
      "nChildren" -> TInt32(),
      "nErrors" -> TInt32(),
      "nSNP" -> TInt32()
    )

    val trioFamBc = sc.broadcast(trioFam)
    val nuclearFamsBc = sc.broadcast(nuclearFams)

    val rdd = nErrorPerNuclearFamily.map { case ((dad, mom), (n, nSNP)) =>
      val kids = nuclearFamsBc.value.get((dad, mom))

      Row(kids.flatMap(x => trioFamBc.value.get(x.head)).orNull, dad, mom, kids.map(_.length).getOrElse(0), n, nSNP)
    }

    KeyTable(hc, rdd, signature, Array("fid"))
  }

  def iMendelKT(): KeyTable = {

    val signature = TStruct(
      "fid" -> TString(),
      "s" -> TString(),
      "nError" -> TInt32(),
      "nSNP" -> TInt32()
    )

    val trioFamBc = sc.broadcast(trios.iterator.flatMap { t =>
      t.fam.toArray.flatMap { f =>
        Iterator(t.kid -> f) ++ Iterator(t.dad, t.mom).flatten.map(x => x -> f)
      }
    }.toMap)

    val rdd = nErrorPerIndiv.map { case (s, (n, nSNP)) => Row(trioFamBc.value.getOrElse(s, null), s, n, nSNP) }

    KeyTable(hc, rdd, signature, Array("s"))
  }

  def lMendelKT(): KeyTable = {
    val signature = TStruct(
<<<<<<< HEAD
      "v" -> TVariant(GenomeReference.GRCh37),
      "nError" -> TInt32()
=======
      "v" -> vSig,
      "nError" -> TInt32
>>>>>>> Change hard coded GRCh37 to default reference
    )

    val rdd = nErrorPerVariant.map { case (v, l) => Row(v, l.toInt) }

    KeyTable(hc, rdd, signature, Array("v"))
  }
}
