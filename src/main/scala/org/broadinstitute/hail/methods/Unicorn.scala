/* Implements the core UNICORN model */

package org.broadinstitute.hail.methods

import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant._

class Unicorn() {
  def name = "UNICORN"

  type Stage1Dist = Map[Variant,(Double,Double)]
  type cases = Seq[(Int,(Double,Double))]

  def alleleCountAnnotate(vds : VariantDataset, refName : String = "refCount", altName : String = "altCount") : VariantDataset = {
    val (t1,i1) = vds.vaSignature.insert(TInt,refName)
    val (t2,i2) = t1.insert(TInt,altName)
    vds.mapAnnotations((v : Variant, va : Annotation, geno) => {
      var nRef : Int = 0
      var nAlt : Int = 0
      geno foreach { (g : Genotype) =>
        g.nNonRefAlleles match {
          case None => Unit
          case Some(n) => {
            nRef += 2 - n
            nAlt += n
          }
        }
      }
      i2(i1(va,Some(nRef)),Some(nAlt))
    }).copy(vaSignature=t2)
  }

  def getP (va : Annotation, qref : Querier, qalt : Querier) : Double = {
    val ref : Double = qref(va) match { case Some(x : Int) => x.toDouble 
                               case _ => 0.0 }
    val alt : Double = qalt(va) match { case Some(x : Int) => x.toDouble 
                               case _ => 0.0 }
    alt / Math.max(ref + alt,1.0)
  }

  def betweenVarStep (va : Annotation, getters : (Querier,Querier), p : Double, n : Double) : Double = {
    val (qref,qalt) = getters
    val pi = getP(va,qref,qalt)
    n * (pi - p) * (pi - p)
  }

  // Counts number of alleles in cluster
  def alleleN(va : Annotation, getters : (Querier,Querier)) : Int = {
    val (qref,qalt) = getters
    val ref : Int = qref(va) match { case Some(x : Int) => x
                               case _ => 0 }
    val alt : Int = qalt(va) match { case Some(x : Int) => x
                               case _ => 0 }
    (ref + alt) / 2
  }

  def fstAnnotate(vds : VariantDataset, subpops : Seq[VariantDataset]) : VariantDataset = {
    val (t,i) = vds.vaSignature.insert(TDouble,"Fst")
    val (_,qglobref) = vds.queryVA("va.globalRefCount")
    val (_,qglobalt) = vds.queryVA("va.globalAltCount")
    val clusterGetters = subpops map (x => {
      val (_,qref) = x.queryVA("va.refCount")
      val (_,qalt) = x.queryVA("va.altCount")
      (qref,qalt)
    })
    val clusters = subpops map (x => x.variantsAndAnnotations.collect().toMap)
    vds.mapAnnotations((v : Variant, va : Annotation, geno) => {
      val p = getP(va,qglobref,qglobalt)
      val totalVar = p * (1.0 - p)
      val n = (0 until subpops.size) map ( (i : Int) => alleleN(clusters(i)(v), clusterGetters(i)) )
      val nsum = n.sum.toDouble
      val betweenVar = ((0 until subpops.size) map
        ( (i : Int) => betweenVarStep (clusters(i)(v), clusterGetters(i), p, n(i) / nsum) )
        ).sum
      val fst = if (totalVar <= 0.0) 1.0 else (betweenVar / totalVar)
      i(va,Some(fst))
    }).copy(vaSignature=t)
  }
  
  // Calculates priors for Stage 1, then updates to get posteriors (a,b)
  def calcHyperparams(vds : VariantDataset, fst : Map[Variant,Double]) : Stage1Dist = {
    val (_,qref) = vds.queryVA("va.refCount")
    val (_,qalt) = vds.queryVA("va.altCount")
    val (_,qglobref) = vds.queryVA("va.globalRefCount")
    val (_,qglobalt) = vds.queryVA("va.globalAltCount")
    vds.mapWithAll ((v, va, _, _, _) => {
      val p = getP(va,qglobref,qglobalt)
      val (a,b) = if (fst(v) == 0.0) (0.5,0.5) else (p * (1 - fst(v)) / fst(v), (1 - p) * (1 - fst(v)) / fst(v))
      val ref = qref(va) match { case Some(n : Int) => n
                                 case _ => 0 }
      val alt = qalt(va) match { case Some(n : Int) => n
                                 case _ => 0 }
      (v,(a + alt, b + ref))
    }).collect().toMap
  }

  def clusterWidePriors(data : VariantDataset, clusts : Seq[Set[String]]) : Seq[Stage1Dist] = {
    var vds = alleleCountAnnotate(data,refName = "globalRefCount",altName = "globalAltCount")
    var subvds : Array[VariantDataset] = Array.tabulate(clusts.size)((i : Int) => vds.filterSamples((name : String, A : Annotation) => clusts(i) contains name) )
    subvds = subvds map (g => alleleCountAnnotate(g))
    vds = fstAnnotate(vds,subvds)

    val (_,qfst) = vds.queryVA("va.Fst")
    val fst = ( vds.mapWithAll ((v, va, _, _, _) => qfst(va) match {
      case Some(x : Double) => (v,x)
      case _ => (v,0.0)
      } ) ).collect().toMap
    
    var posteriors : Seq[Stage1Dist] = subvds map (x => calcHyperparams(x,fst))
    posteriors
  }

  // Uses MCMC algorithm to compute posteriors on within-cluster GP
  def fitGP(data : VariantDataset, priors : Seq[Stage1Dist]) : Unit = {
    // Read in data & priors
    // For each variant:
    //   Generate Gaussian prior for allele frequency (Tracy 39-44)
    //   Generate priors for phi
    //   Make initial pull for phi and S
    //   For each MCMC iteration (Tracy uses 6000):
    //     d

    Unit
  }

  def betaMeanVariance(beta : (Double, Double)) : (Double,Double) = {
    val (a,b) = beta
    val mean = a / (a + b)
    val variance = (a * b) / ((a + b) * (a + b) * (a + b + 1))
    (mean,variance)
  }

  def nulldist(samples : cases, model : Seq[Stage1Dist]) : Map[Variant,(Double,Double)] = {
    val n = samples.size.toDouble
    val variants = model(0) mapValues (x => (0.0,0.0))
    val modelParam : Seq[Map[Variant,(Double,Double)]] = model map ((m : Stage1Dist) => m mapValues betaMeanVariance)
    val ybar = (samples foldLeft variants) ((acc,loc) => { 
      val (clust,(pc1,pc2)) = loc
      (acc.keys map ((k : Variant) => {
         val (mean,variance) : (Double,Double) = modelParam(clust)(k)
         val (aggregatemean,aggregatevariance) = acc(k)
         val meanincrement = mean * 2 / n
         val varianceincrement = ( (2 * mean * (1 - mean)) + (2 * variance)) / (n * n)
         (k,(aggregatemean + meanincrement,aggregatevariance + varianceincrement)) } )).toMap 
    } )
    ybar 
  }
  
  def chisqtop(x : Double) : Double = Gamma.regularizedGammaQ(0.5, x / 2)

  // Assumes samples and vds describe the same dataset, and ref / alt alleles
  // are the same in the cases as in the UNICORN controls 
  // Assumes null dist is accurate and factors in missingness
  def test(samples : cases, vds_samples : VariantDataset, nulldist : Map[Variant,(Double,Double)]) : Map[Variant,Option[Double]] = {
    val vds = alleleCountAnnotate(vds_samples)
    //val (_, refQuery) = vds.queryVA("va.refCount")
    val (_, altQuery) = vds.queryVA("va.altCount")
    val vam = vds.variantsAndAnnotations.collect().toMap
    (vam.keys map ((v : Variant) => {
      val va : Annotation = vam(v)
      val altCount = altQuery(va) match { case Some(n : Int) => n.toDouble
                                          case _ => 0.0 }
      val Y = altCount / samples.size
      if (! (nulldist contains v)) None else Some ( {
        val (Ybar,Ybarvar) = nulldist(v)
        val chisq = (Y - Ybar) * (Y - Ybar) / Ybarvar
        chisqtop(chisq)
        } )
    } )).toMap
  }

  def apply(data : VariantDataset, clusts : Seq[Set[String]]) : Seq[Stage1Dist] = {
    val stage1prior : Seq[Stage1Dist] = clusterWidePriors(data,clusts)
    // Stage 2 goes here!
    stage1prior
  }
}
