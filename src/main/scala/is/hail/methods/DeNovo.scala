package is.hail.methods

import is.hail.keytable.KeyTable
import is.hail.variant._
import is.hail.expr._
import is.hail.utils._
import is.hail.annotations._

import scala.collection.mutable

object DeNovo {

  def keytableDefaultFields: Array[(String, Type)] = Array(
    "variant" -> TVariant,
    "probandID" -> TString,
    "fatherID" -> TString,
    "motherID" -> TString,
    "isFemale" -> TBoolean,
    "isCase" -> TBoolean,
    "validationLikelihood" -> TDouble,
    "probandGt" -> TGenotype,
    "motherGt" -> TGenotype,
    "fatherGt" -> TGenotype,
    "pDeNovo" -> TDouble)


  private val PRIOR = 1.0 / 30000000

  private val MIN_PRIOR = 100.0 / 30000000

  def callAutosomal(kid: CompleteGenotype, dad: CompleteGenotype, mom: CompleteGenotype, isSNP: Boolean, prior: Double,
    minPDeNovo: Double, nAltAlleles: Int, minDpRatio: Double, maxParentAB: Double): Option[(String, Double)] = {

    if (dad == null || mom == null ||
      !(kid.gt == 1 && dad.gt == 0 && mom.gt == 0) ||
      kid.dp.toDouble / (dad.dp + mom.dp) < minDpRatio ||
      (dad.ad(0) == 0) && (dad.ad(1) == 0) ||
      (mom.ad(0) == 0) && (mom.ad(1) == 0) ||
      dad.ad(1).toDouble / (dad.ad(0) + dad.ad(1)) >= maxParentAB ||
      mom.ad(1).toDouble / (mom.ad(0) + mom.ad(1)) >= maxParentAB)
      return None

    val kidP = kid.pl.map(x => math.pow(10, -x / 10d))
    val dadP = dad.pl.map(x => math.pow(10, -x / 10d))
    val momP = mom.pl.map(x => math.pow(10, -x / 10d))

    val kidSum = kidP.sum
    val dadSum = dadP.sum
    val momSum = momP.sum

    (0 until 3).foreach { i =>
      kidP(i) = kidP(i) / kidSum
      dadP(i) = dadP(i) / dadSum
      momP(i) = momP(i) / momSum
    }

    val pDeNovoData = dadP(0) * momP(0) * kidP(1) * PRIOR

    val pDataOneHet = (dadP(1) * momP(0) + dadP(0) * momP(1)) * kidP(1)
    val pOneParentHet = 1 - math.pow(1 - prior, 4)
    val pMissedHetInParent = pDataOneHet * pOneParentHet

    val pTrueDeNovo = pDeNovoData / (pDeNovoData + pMissedHetInParent)

    val kidAdRatio = kid.ad(1).toDouble / (kid.ad(0) + kid.ad(1))

    val kidDp = kid.dp
    val dpRatio = kidDp.toDouble / (mom.dp + dad.dp)

    // Below is the core calling algorithm
    if (pTrueDeNovo < minPDeNovo)
      None
    else if (!isSNP) {
      if ((pTrueDeNovo > 0.99) && (kidAdRatio > 0.3) && (nAltAlleles == 1))
        Some("HIGH", pTrueDeNovo)
      else if ((pTrueDeNovo > 0.5) && (kidAdRatio > 0.3) && (nAltAlleles <= 5))
        Some("MEDIUM", pTrueDeNovo)
      else if ((pTrueDeNovo > 0.05) && (kidAdRatio > 0.20))
        Some("LOW", pTrueDeNovo)
      else None
    } else {
      if ((pTrueDeNovo > 0.99) && (kidAdRatio > 0.3) && (dpRatio > 0.2) ||
        ((pTrueDeNovo > 0.99) && (kidAdRatio > 0.3) && (nAltAlleles == 1)) ||
        ((pTrueDeNovo > 0.5) && (kidAdRatio >= 0.3) && (nAltAlleles < 10) && (kidDp >= 10))
      )
        Some("HIGH", pTrueDeNovo)
      else if ((pTrueDeNovo > 0.5) && (kidAdRatio > 0.3) ||
        ((pTrueDeNovo > 0.5) && (nAltAlleles == 1))
      )
        Some("MEDIUM", pTrueDeNovo)
      else if ((pTrueDeNovo > 0.05) && (kidAdRatio > 0.20))
        Some("LOW", pTrueDeNovo)
      else None
    }
  }

  def callHemizygous(kid: CompleteGenotype, parent: CompleteGenotype, isSNP: Boolean, prior: Double,
    minPDeNovo: Double, nAltAlleles: Int, minDpRatio: Double, maxParentAB: Double): Option[(String, Double)] = {

    if (parent == null ||
      !(kid.gt == 2 && parent.gt == 0) ||
      kid.dp.toDouble / parent.dp < minDpRatio ||
      (parent.ad(0) == 0) && (parent.ad(1) == 0) ||
      parent.ad(1).toDouble / (parent.ad(0) + parent.ad(1)) >= maxParentAB)
      return None

    val kidP = kid.pl.map(x => math.pow(10, -x / 10d))
    val parentP = parent.pl.map(x => math.pow(10, -x / 10d))

    val kidSum = kidP.sum
    val parentSum = parentP.sum

    (0 until 3).foreach { i =>
      kidP(i) = kidP(i) / kidSum
      parentP(i) = parentP(i) / parentSum
    }

    val pDeNovoData = parentP(0) * kidP(2) * PRIOR

    val pDataOneHet = (parentP(1) + parentP(2)) * kidP(2)
    val pOneParentHet = 1 - math.pow(1 - prior, 4)
    val pMissedHetInParent = pDataOneHet * pOneParentHet

    val pTrueDeNovo = pDeNovoData / (pDeNovoData + pMissedHetInParent)

    val kidAdRatio = kid.ad(1).toDouble / (kid.ad(0) + kid.ad(1))

    val kidDp = kid.dp
    val dpRatio = kidDp.toDouble / parent.dp

    // Below is the core calling algorithm
    if (pTrueDeNovo < minPDeNovo || kidAdRatio <= 0.95)
      None
    else if (!isSNP) {
      if ((pTrueDeNovo > 0.99) && (kidAdRatio > 0.3) && (nAltAlleles == 1))
        Some("HIGH", pTrueDeNovo)
      else if ((pTrueDeNovo > 0.5) && (kidAdRatio > 0.3) && (nAltAlleles <= 5))
        Some("MEDIUM", pTrueDeNovo)
      else if ((pTrueDeNovo > 0.05) && (kidAdRatio > 0.20))
        Some("LOW", pTrueDeNovo)
      else None
    } else {
      if ((pTrueDeNovo > 0.99) && (kidAdRatio > 0.3) && (dpRatio > 0.2) ||
        ((pTrueDeNovo > 0.99) && (kidAdRatio > 0.3) && (nAltAlleles == 1)) ||
        ((pTrueDeNovo > 0.5) && (kidAdRatio >= 0.3) && (nAltAlleles < 10) && (kidDp >= 10)))
        Some("HIGH", pTrueDeNovo)
      else if ((pTrueDeNovo > 0.5) && (kidAdRatio > 0.3) ||
        ((pTrueDeNovo > 0.5) && (kidAdRatio > minDpRatio) && (nAltAlleles == 1)))
        Some("MEDIUM", pTrueDeNovo)
      else if ((pTrueDeNovo > 0.05) && (kidAdRatio > 0.20))
        Some("LOW", pTrueDeNovo)
      else None
    }
  }


  def call(vds: VariantDataset, famFile: String,
    referenceAFExpr: String,
    extraFieldsExpr: Option[String] = None,
    minGQ: Int = 20,
    minPDeNovo: Double = 0.05,
    maxParentAB: Double = 0.05,
    minChildAB: Double = 0.20,
    minDepthRatio: Double = 0.10): KeyTable = {
    require(vds.wasSplit)

    val ped = Pedigree.read(famFile, vds.hadoopConf, vds.sampleIds)

    val (popFrequencyT, popFrequencyF) = vds.queryVA(referenceAFExpr)
    if (popFrequencyT != TDouble)
      fatal(s"population frequency should be a Double, but got `$popFrequencyT'")

    val popFreqQuery: (Annotation) => Option[Double] =
      (a: Annotation) => Option(popFrequencyF(a)).map(_.asInstanceOf[Double])

    val additionalOutput = extraFieldsExpr.map { cond =>
      val symTab = Map(
        "v" -> (0, TVariant),
        "va" -> (1, vds.vaSignature),
        "global" -> (2, vds.globalSignature),
        "proband" -> (3, TString),
        "father" -> (4, TString),
        "mother" -> (5, TString),
        "probandGt" -> (6, TGenotype),
        "fatherGt" -> (7, TGenotype),
        "motherGt" -> (8, TGenotype),
        "probandAnnot" -> (9, vds.saSignature),
        "fatherAnnot" -> (10, vds.saSignature),
        "motherAnnot" -> (11, vds.saSignature)
      )

      val ec = EvalContext(symTab)

      val (names, types, fs) = Parser.parseNamedExprs(cond, ec)
      val defaultFields = keytableDefaultFields.map(_._1).toSet
      val badNames = names.toSet.intersect(defaultFields)
      if (badNames.nonEmpty)
        fatal(s"additional fields may not intersect with the default namespace.  Problem fields: [ ${
          badNames.mkString(", ")
        } ]")
      (ec, names.zip(types), fs)
    }

    val schema = TStruct(keytableDefaultFields ++ additionalOutput.map(o => o._2).getOrElse(Array.empty): _*)
    val nFields = schema.size
    val nDefaultFields = keytableDefaultFields.length

    val trios = ped.completeTrios.filter(_.sex.isDefined)
    val nSamplesDiscarded = ped.trios.length - trios.length
    val nTrios = trios.size

    info(s"Calling de novo events for $nTrios trios")

    if (nSamplesDiscarded > 0)
      warn(s"$nSamplesDiscarded ${ plural(nSamplesDiscarded, "sample") } discarded from .fam: missing from data set.")
    val sampleTrioRoles = mutable.Map.empty[String, List[(Int, Int)]]

    // need a map from Sample position(int) to (int, int)
    trios.zipWithIndex.foreach { case (t, ti) =>
      sampleTrioRoles += (t.kid -> ((ti, 0) :: sampleTrioRoles.getOrElse(t.kid, List.empty[(Int, Int)])))
      sampleTrioRoles += (t.dad -> ((ti, 1) :: sampleTrioRoles.getOrElse(t.dad, List.empty[(Int, Int)])))
      sampleTrioRoles += (t.mom -> ((ti, 2) :: sampleTrioRoles.getOrElse(t.mom, List.empty[(Int, Int)])))
    }

    val idMapping = vds.sampleIds.zipWithIndex.toMap

    val sc = vds.sparkContext
    val trioIndexBc = sc.broadcast(trios.map(t => (idMapping(t.kid), idMapping(t.dad), idMapping(t.mom))))
    val sampleTrioRolesBc = sc.broadcast(vds.sampleIds.map(sampleTrioRoles.getOrElse(_, Nil)).toArray)
    val triosBc = sc.broadcast(trios)
    val trioSexBc = sc.broadcast(trios.map(_.sex.get).toArray)

    val localGlobal = vds.globalAnnotation
    val localAnnotationsBc = vds.sampleAnnotationsBc

    val rdd = vds.rdd.mapPartitions { iter =>
      val arr = MultiArray2.fill[CompleteGenotype](trios.length, 3)(null)

      iter.flatMap { case (v, (va, gs)) =>

        var totalAlleles = 0
        var nAltAlleles = 0

        var ii = 0
        while (ii < nTrios) {
          var jj = 0
          while (jj < 3) {
            arr.update(ii, jj, null)
            jj += 1
          }
          ii += 1
        }

        var i = 0
        gs.foreach { g =>

          g.toCompleteGenotype.foreach { cg =>
            val roles = sampleTrioRolesBc.value(i)
            roles.foreach { case (ri, ci) => arr.update(ri, ci, cg) }

            nAltAlleles += cg.gt
            totalAlleles += 2
          }
          i += 1
        }

        // correct for the observed genotype
        val computedFrequency = (nAltAlleles.toDouble - 1) / totalAlleles.toDouble

        val popFrequency = popFreqQuery(va).getOrElse(0d)
        if (popFrequency < 0 || popFrequency > 1)
          fatal(
            s"""invalid population frequency value `$popFrequency' for variant $v
                  Population prior must fall between 0 and 1.""".stripMargin)

        val frequency = math.max(math.max(computedFrequency, popFrequency), MIN_PRIOR)

        (0 until nTrios).flatMap { t =>
          val kid = arr(t, 0)
          val dad = arr(t, 1)
          val mom = arr(t, 2)

          val isSNP = v.altAllele.isSNP
          val annotation =
            if (kid == null || kid.gt == 0 || kid.gq <= minGQ ||
              (kid.ad(0) == 0 && kid.ad(1) == 0) ||
              kid.ad(1).toDouble / (kid.ad(0) + kid.ad(1)) <= minChildAB)
              None
            else if (v.isAutosomal || v.inXPar)
              callAutosomal(kid, dad, mom, isSNP, frequency,
                minPDeNovo, nAltAlleles, minDepthRatio, maxParentAB)
            else {
              val sex = trioSexBc.value(t)
              if (v.inXNonPar) {
                if (sex == Sex.Female)
                  callAutosomal(kid, dad, mom, isSNP, frequency,
                    minPDeNovo, nAltAlleles, minDepthRatio, maxParentAB)
                else
                  callHemizygous(kid, mom, isSNP, frequency, minPDeNovo, nAltAlleles, minDepthRatio, maxParentAB)
              } else if (v.inYNonPar) {
                if (sex == Sex.Female)
                  None
                else
                  callHemizygous(kid, dad, isSNP, frequency, minPDeNovo, nAltAlleles, minDepthRatio, maxParentAB)
              } else if (v.isMitochondrial)
                callHemizygous(kid, mom, isSNP, frequency, minPDeNovo, nAltAlleles, minDepthRatio, maxParentAB)
              else None
            }

          annotation.map { case (str, p) =>

            val defaults: Array[Annotation] = Array(
              v,
              triosBc.value(t).kid,
              triosBc.value(t).dad,
              triosBc.value(t).mom,
              triosBc.value(t).sex.map(s => s == Sex.Female).orNull,
              triosBc.value(t).pheno.map(p => p == Phenotype.Case).orNull,
              str,
              kid.toGenotype,
              if (dad != null) dad.toGenotype else null,
              if (mom != null) mom.toGenotype else null,
              p)

            val fullRow = additionalOutput match {
              case Some((ec, _, fs)) =>
                val trio = triosBc.value(t)
                val kidId = trio.kid
                val dadId = trio.dad
                val momId = trio.mom

                val (kidIndex, dadIndex, momIndex) = trioIndexBc.value(t)

                ec.set(0, v)
                ec.set(1, va)
                ec.set(2, localGlobal)
                ec.set(3, kidId)
                ec.set(4, dadId)
                ec.set(5, momId)
                ec.set(6, if (dad != null) dad.toGenotype else null)
                ec.set(7, if (mom != null) mom.toGenotype else null)
                ec.set(8, localAnnotationsBc.value(kidIndex))
                ec.set(9, localAnnotationsBc.value(dadIndex))
                ec.set(10, localAnnotationsBc.value(momIndex))
                ec.set(11, p)

                val results = fs()
                val combined = defaults ++ results
                assert(combined.length == nFields)
                combined
              case None => defaults
            }
            Annotation.fromSeq(fullRow)
          }
        }.iterator
      }
    }.cache()

    KeyTable(vds.hc, rdd, schema, Array.empty)
  }
}