package org.broadinstitute.hail.methods

/* Implements adaptive PCA
 *
 * Adaptive PCA is a hierarchical clustering method for genotypes.  At each
 * step, PCA is used to move genotypes into a lower-dimensional space which 
 * captures most of the variance of the original.  Then, a stopcondition is
 * evaluated. If the genotypes meet the stopcondition, they are viewed as a
 * homogenous population and clustering ends.   If the stopcondition is not
 * met, a clustering algorithm is used to split the genotypes into k clust-
 * ters and adaptive PCA is run again on each of the sub-clusters.
 *
 * In this implementation, the clustering algorithm is Ward's algorithm and
 * clustering stops when the eigenvalues of the first two principal compon-
 * ents exceed p percent of the sum of eigenvalues.
 *
 * Note that this class includes a hacky definition of a tree, because real
 * binary trees aren't in Scala's standard library as far as I can tell.
 */

import org.broadinstitute.hail.variant.VariantDataset

class AdaptivePCA(k : Int) {
  def name = "AdaptivePCA"
    
  abstract class Tree[A]
  case class Leaf[A](v : A) extends Tree[A]
  case class Node[A](left : Tree[A],v : A,right : Tree[A]) extends Tree[A]

  def leaves[A](T : Tree[A]) : Set[A] = {
    T match {
      case Leaf(v) => Set(v)
      case Node(l,_,r) => leaves(l) ++ leaves(r)
    }
  }

  /*def stopcondition(evalues : Seq[Double]) : Boolean = {
    ((evalues(0) + evalues(1)) / evalues.sum) > threshold
  }*/

  def apply (vds : VariantDataset, iterations : Int) : Tree[IndexedSeq[String]] = {
    val sampleIds = vds.sampleIds
    if (vds.nSamples <= k | iterations == 0) Leaf(sampleIds) else {
    val PCA = new SamplePCA(k,false,true)
    val (scores, None, Some(evalues)) = PCA(vds)
    val W = new Ward()
    val D_base = W.distMat(scores)
    val D = D_base map ((S : Seq[Double]) => (S map Math.sqrt))
    val clusts = W(D,2).toSeq
    /* Note- the next line is not good Hail style, it's forced by 
     * the fact that SamplePCA returns a plain 2d array w/o IDs
     * attached.
     */
    val idclusts = clusts map ((S : Set[Int]) => S map ((i : Int) => sampleIds(i)))
    def p (i : Int) (name : String, A : Any) : Boolean = idclusts(i) contains name
    Node(apply(vds.filterSamples(p(0)),iterations-1),sampleIds,apply(vds.filterSamples(p(1)),iterations-1))
  } }
}
