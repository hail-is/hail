package is.hail.methods

import is.hail.keytable.KeyTable
import is.hail.variant._
import is.hail.expr._
import is.hail.utils._
import is.hail.annotations._
import org.apache.spark.sql.Row

import scala.collection.mutable

object DeNovo {

  def isCompleteGenotype(g: Genotype): Boolean = {
    Genotype.unboxedGT(g) >= 0 && Genotype.unboxedAD(g) != null && Genotype.unboxedDP(g) >= 0 && Genotype.unboxedGQ(g) >= 0 && Genotype.unboxedPL(g) != null
  }

  def schema(variantType: Type): TStruct = TStruct(
    "variant" -> variantType,
    "proband" -> TString(),
    "father" -> TString(),
    "mother" -> TString(),
    "isFemale" -> TBoolean(),
    "confidence" -> TString(),
    "probandGt" -> TGenotype(),
    "motherGt" -> TGenotype(),
    "fatherGt" -> TGenotype(),
    "pDeNovo" -> TFloat64())

  private val PRIOR = 1.0 / 30000000

  private val MIN_PRIOR = 100.0 / 30000000

  def callAutosomal(kid: Genotype, dad: Genotype, mom: Genotype, isSNP: Boolean, prior: Double,
    minPDeNovo: Double, nAltAlleles: Int, minDpRatio: Double, maxParentAB: Double): Option[(String, Double)] = {

    if (dad == null || mom == null ||
      !(Genotype.unboxedGT(kid) == 1 && Genotype.unboxedGT(dad) == 0 && Genotype.unboxedGT(mom) == 0) ||
      Genotype.unboxedGT(kid).toDouble / (Genotype.unboxedGT(dad) + Genotype.unboxedGT(mom)) < minDpRatio ||
      (Genotype.unboxedAD(dad)(0) == 0) && (Genotype.unboxedAD(dad)(1) == 0) ||
      (Genotype.unboxedAD(mom)(0) == 0) && (Genotype.unboxedAD(mom)(1) == 0) ||
      Genotype.unboxedAD(dad)(1).toDouble / (Genotype.unboxedAD(dad)(0) + Genotype.unboxedAD(dad)(1)) >= maxParentAB ||
      Genotype.unboxedAD(mom)(1).toDouble / (Genotype.unboxedAD(mom)(0) + Genotype.unboxedAD(mom)(1)) >= maxParentAB)
      return None

    val kidP = Genotype.unboxedPL(kid).map(x => math.pow(10, -x / 10d))
    val dadP = Genotype.unboxedPL(dad).map(x => math.pow(10, -x / 10d))
    val momP = Genotype.unboxedPL(mom).map(x => math.pow(10, -x / 10d))

    val kidSum = kidP.sum
    val dadSum = dadP.sum
    val momSum = momP.sum

    (0 until 3).foreach { i =>
      kidP(i) = kidP(i) / kidSum
      dadP(i) = dadP(i) / dadSum
      momP(i) = momP(i) / momSum
    }

    val pDataGivenDeNovo = dadP(0) * momP(0) * kidP(1) * PRIOR
    val pHetInParent = 1 - math.pow(1 - prior, 4)
    val pDataGivenMissedHet = (dadP(1) * momP(0) + dadP(0) * momP(1)) * kidP(1) * pHetInParent

    val pDeNovo = pDataGivenDeNovo / (pDataGivenDeNovo + pDataGivenMissedHet)

    val kidAdRatio = Genotype.unboxedAD(kid)(1).toDouble / (Genotype.unboxedAD(kid)(0) + Genotype.unboxedAD(kid)(1))

    val kidDp = Genotype.unboxedDP(kid)
    val dpRatio = kidDp.toDouble / (Genotype.unboxedDP(mom) + Genotype.unboxedDP(dad))

    // Core calling algorithm
    if (pDeNovo < minPDeNovo)
      None
    else if (dpRatio < minDpRatio)
      None
    else if (!isSNP) {
      if ((pDeNovo > 0.99) && (kidAdRatio > 0.3) && (nAltAlleles == 1))
        Some("HIGH", pDeNovo)
      else if ((pDeNovo > 0.5) && (kidAdRatio > 0.3) && (nAltAlleles <= 5))
        Some("MEDIUM", pDeNovo)
      else if ((pDeNovo > 0.05) && (kidAdRatio > 0.20))
        Some("LOW", pDeNovo)
      else None
    } else {
      if ((pDeNovo > 0.99) && (kidAdRatio > 0.3) && (dpRatio > 0.2) ||
        ((pDeNovo > 0.99) && (kidAdRatio > 0.3) && (nAltAlleles == 1)) ||
        ((pDeNovo > 0.5) && (kidAdRatio >= 0.3) && (nAltAlleles < 10) && (kidDp >= 10))
      )
        Some("HIGH", pDeNovo)
      else if ((pDeNovo > 0.5) && (kidAdRatio > 0.3) ||
        ((pDeNovo > 0.5) && (nAltAlleles == 1))
      )
        Some("MEDIUM", pDeNovo)
      else if ((pDeNovo > 0.05) && (kidAdRatio > 0.20))
        Some("LOW", pDeNovo)
      else None
    }
  }

  def callHemizygous(kid: Genotype, parent: Genotype, isSNP: Boolean, prior: Double,
    minPDeNovo: Double, nAltAlleles: Int, minDpRatio: Double, maxParentAB: Double): Option[(String, Double)] = {

    if (parent == null ||
      !(Genotype.unboxedGT(kid) == 2 && Genotype.unboxedGT(parent) == 0) ||
      Genotype.unboxedDP(kid).toDouble / Genotype.unboxedDP(parent) < minDpRatio ||
      (Genotype.unboxedAD(parent)(0) == 0) && (Genotype.unboxedAD(parent)(1) == 0) ||
      Genotype.unboxedAD(parent)(1).toDouble / (Genotype.unboxedAD(parent)(0) + Genotype.unboxedAD(parent)(1)) >= maxParentAB)
      return None

    val kidP = Genotype.unboxedPL(kid).map(x => math.pow(10, -x / 10d))
    val parentP = Genotype.unboxedPL(parent).map(x => math.pow(10, -x / 10d))

    val kidSum = kidP.sum
    val parentSum = parentP.sum

    (0 until 3).foreach { i =>
      kidP(i) = kidP(i) / kidSum
      parentP(i) = parentP(i) / parentSum
    }

    val pDataGivenDeNovo = parentP(0) * kidP(2) * PRIOR
    // seems wrong (should be exponent of 2?) but agrees with the Python code
    val pHetInParent = 1 - math.pow(1 - prior, 4)
    val pDataGivenMissedHet = (parentP(1) + parentP(2)) * kidP(2) * pHetInParent

    val pDeNovo = pDataGivenDeNovo / (pDataGivenDeNovo + pDataGivenMissedHet)

    val kidAdRatio = Genotype.unboxedAD(kid)(1).toDouble / (Genotype.unboxedAD(kid)(0) + Genotype.unboxedAD(kid)(1))

    val kidDp = Genotype.unboxedDP(kid)
    val dpRatio = kidDp.toDouble / Genotype.unboxedDP(parent)

    // Core calling algorithm
    if (pDeNovo < minPDeNovo || kidAdRatio <= 0.95)
      None
    else if (!isSNP) {
      if ((pDeNovo > 0.99) && (kidAdRatio > 0.3) && (nAltAlleles == 1))
        Some("HIGH", pDeNovo)
      else if ((pDeNovo > 0.5) && (kidAdRatio > 0.3) && (nAltAlleles <= 5))
        Some("MEDIUM", pDeNovo)
      else if ((pDeNovo > 0.05) && (kidAdRatio > 0.20))
        Some("LOW", pDeNovo)
      else None
    } else {
      if ((pDeNovo > 0.99) && (kidAdRatio > 0.3) && (dpRatio > 0.2) ||
        ((pDeNovo > 0.99) && (kidAdRatio > 0.3) && (nAltAlleles == 1)) ||
        ((pDeNovo > 0.5) && (kidAdRatio >= 0.3) && (nAltAlleles < 10) && (kidDp >= 10)))
        Some("HIGH", pDeNovo)
      else if ((pDeNovo > 0.5) && (kidAdRatio > 0.3) ||
        ((pDeNovo > 0.5) && (kidAdRatio > minDpRatio) && (nAltAlleles == 1)))
        Some("MEDIUM", pDeNovo)
      else if ((pDeNovo > 0.05) && (kidAdRatio > 0.20))
        Some("LOW", pDeNovo)
      else None
    }
  }


  def apply(vds: VariantDataset, ped: Pedigree,
    referenceAFExpr: String,
    minGQ: Int = 20,
    minP: Double = 0.05,
    maxParentAB: Double = 0.05,
    minChildAB: Double = 0.20,
    minDepthRatio: Double = 0.10): KeyTable = {
    require(vds.wasSplit)

    val (popFrequencyT, popFrequencyF) = vds.queryVA(referenceAFExpr)
    if (!popFrequencyT.isOfType(TFloat64()))
      fatal(s"population frequency should be a Double, but got `$popFrequencyT'")

    // it's okay to cast null to 0.0 here because missing is treated as 0.0
    val popFreqQuery: (Annotation) => Double = popFrequencyF(_).asInstanceOf[Double]

    val trios = ped.filterTo(vds.stringSampleIds.toSet).completeTrios
    val nSamplesDiscarded = ped.trios.length - trios.length
    val nTrios = trios.size

    info(s"Calling de novo events for $nTrios complete trios")

    val sampleTrioRoles = mutable.Map.empty[String, List[(Int, Int)]]

    trios.zipWithIndex.foreach { case (t, ti) =>
      sampleTrioRoles += (t.kid -> ((ti, 0) :: sampleTrioRoles.getOrElse(t.kid, List.empty[(Int, Int)])))
      sampleTrioRoles += (t.knownDad -> ((ti, 1) :: sampleTrioRoles.getOrElse(t.knownDad, List.empty[(Int, Int)])))
      sampleTrioRoles += (t.knownMom -> ((ti, 2) :: sampleTrioRoles.getOrElse(t.knownMom, List.empty[(Int, Int)])))
    }

    val idMapping = vds.stringSampleIds.zipWithIndex.toMap

    val sc = vds.sparkContext
    val trioIndexBc = sc.broadcast(trios.map(t => (idMapping(t.kid), idMapping(t.knownDad), idMapping(t.knownMom))))
    val sampleTrioRolesBc = sc.broadcast(vds.stringSampleIds.map(sampleTrioRoles.getOrElse(_, Nil)).toArray)
    val triosBc = sc.broadcast(trios)
    val trioSexBc = sc.broadcast(trios.map(_.sex.orNull).toArray)

    val rdd = vds.typedRDD[Locus, Variant, Genotype].mapPartitions { iter =>
      val arr = MultiArray2.fill[Genotype](trios.length, 3)(null)

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
          if (DeNovo.isCompleteGenotype(g)) {

            val roles = sampleTrioRolesBc.value(i)
            roles.foreach { case (ri, ci) => arr.update(ri, ci, g) }

            nAltAlleles += Genotype.unboxedGT(g)
            totalAlleles += 2
          }
          i += 1
        }

        // correct for the observed genotype
        val computedFrequency = (nAltAlleles.toDouble - 1) / totalAlleles.toDouble

        val popFrequency = popFreqQuery(va)
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
            if (kid == null || Genotype.unboxedGT(kid) == 0 || Genotype.unboxedGQ(kid) <= minGQ ||
              (Genotype.unboxedAD(kid)(0) == 0 && Genotype.unboxedAD(kid)(1) == 0) ||
              Genotype.unboxedAD(kid)(1).toDouble / (Genotype.unboxedAD(kid)(0) + Genotype.unboxedAD(kid)(1)) <= minChildAB)
              None
            else if (v.isAutosomal || v.inXPar)
              callAutosomal(kid, dad, mom, isSNP, frequency,
                minP, nAltAlleles, minDepthRatio, maxParentAB)
            else {
              val sex = trioSexBc.value(t)
              if (v.inXNonPar) {
                if (sex == Sex.Female)
                  callAutosomal(kid, dad, mom, isSNP, frequency,
                    minP, nAltAlleles, minDepthRatio, maxParentAB)
                else if (sex == Sex.Male)
                  callHemizygous(kid, mom, isSNP, frequency, minP, nAltAlleles, minDepthRatio, maxParentAB)
                else None
              } else if (v.inYNonPar) {
                if (sex == Sex.Female)
                  None
                else if (sex == Sex.Male)
                  callHemizygous(kid, dad, isSNP, frequency, minP, nAltAlleles, minDepthRatio, maxParentAB)
                else None
              } else if (v.isMitochondrial)
                callHemizygous(kid, mom, isSNP, frequency, minP, nAltAlleles, minDepthRatio, maxParentAB)
              else None
            }

          annotation.map { case (confidence, p) =>
            Row(
              v,
              triosBc.value(t).kid,
              triosBc.value(t).knownDad,
              triosBc.value(t).knownMom,
              triosBc.value(t).sex.map(s => (s == Sex.Female): Annotation).orNull,
              confidence,
              kid,
              dad,
              mom,
              p)
          }
        }.iterator
      }
    }.persist()

    KeyTable(vds.hc, rdd, schema(vds.vSignature), Array.empty)
  }
}
