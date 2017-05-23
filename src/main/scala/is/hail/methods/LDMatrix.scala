package is.hail.methods

import is.hail.utils._
import is.hail.stats.ToNormalizedIndexedRowMatrix
import is.hail.variant.{Variant, VariantDataset}
import org.apache.spark.Partitioner
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.reflect.classTag


object LDMatrix {

  /**
    * Computes the LD matrix for the given VDS.
    * @param vds VDS on which to compute Pearson correlation between pairs of variants.
    * @return An LDMatrix.
    */
  def apply(vds : VariantDataset): LDMatrix = {
    val nSamples = vds.nSamples
    val originalVariants = vds.variants.collect()
    assert(originalVariants.isSorted, "Array of variants is not sorted. This is a bug.")

    val normalizedIRM = ToNormalizedIndexedRowMatrix(vds)
    normalizedIRM.rows.persist()
    val variantKeptIndices = normalizedIRM.rows.map{case IndexedRow(idx, _) => idx.toInt}.collect()
    assert(variantKeptIndices.isSorted, "Array of kept variants is not sorted. This is a bug.")
    val variantsKept = variantKeptIndices.map(idx => originalVariants(idx))

    val nVariantsKept = variantsKept.length
    val nVariantsDropped = originalVariants.length - nVariantsKept

    info(s"Computing LD Matrix with ${variantsKept.length} variants using $nSamples samples. $nVariantsDropped variants were dropped.")

    //The indices can be expected to be correct from the zip since the VDS is backed by an OrderedRDD of variants.
    val normalizedFilteredRows = normalizedIRM.rows.zipWithIndex()
      .map {case (IndexedRow(_, data), idx) => IndexedRow(idx, data)}

    val normalizedBlockMatrix = new IndexedRowMatrix(normalizedFilteredRows).toBlockMatrixDense()
    normalizedBlockMatrix.persist(StorageLevel.MEMORY_AND_DISK)
    normalizedBlockMatrix.blocks.count()
    normalizedIRM.rows.unpersist()

    val indexedRowMatrix = (normalizedBlockMatrix multiply normalizedBlockMatrix.transpose)
      .toIndexedRowMatrix()

    val nSamplesInverse = 1.0 / nSamples
    val scaledIndexedRowMatrix = new IndexedRowMatrix(indexedRowMatrix.rows.
      map{case IndexedRow(idx, vals) => IndexedRow(idx, vals.map(d => d * nSamplesInverse))})

    LDMatrix(scaledIndexedRowMatrix, variantsKept, vds.nSamples)
  }

  def apply2(vds: VariantDataset): LDMatrix = {
    val nSamples = vds.nSamples
    //TODO This line sucks, also need to collect variantsKeptArray.
    val bpvs = vds.rdd.flatMap{ case(v, (_, gs)) => LDPrune.toBitPackedVector(gs.hardCallIterator, nSamples)}.zipWithIndex().map{ case(a, b) => (b, a)}

    // Chunk out BPVs with block id's also, use grid partitioner to identify destinations, then map destinations over to
    // send vectors there, and then use Jackie method to smash individual BPV's together.

    val groupSize = 5 * nSamples / vds.rdd.getNumPartitions

    val grouped = bpvs.map{case(vIdx, bpv) => (vIdx / groupSize, (vIdx, bpv))}.groupByKey()
    val numberOfGroups = Math.ceil(grouped.count() / groupSize.toDouble).toInt

    val groupDestinations = simulateLDMatrix(numberOfGroups)

    grouped.flatMap{ case(groupID: Long, itr: Iterable[(Long, LDPrune.BitPackedVector)]) =>
      //Get the set of places this group is needed.
      val destinations = groupDestinations.getOrElse(groupID.toInt, Set.empty)
      destinations.map(dest => (dest, itr))
    }


    //At some point, end up with RDD of blocks
    val computedBlocks: RDD[((Int, Int), Matrix)] = ???

    //Realize I've only computed upper triangle of blocks, so need to transpose to get rest of it.
    val allBlocks = computedBlocks.flatMap{ case((i, j), mat) =>
      if (i == j) List(((i, j), mat)) else List(((i, j), mat), ((j, i), mat.transpose))
    }

    //Construct block matrix and convert to IndexedRowMatrix



    //Make LD Matrix
    ???
  }

  //Create a Map from chunk number to Set of partitions where this chunk is needed.
  private def simulateLDMatrix(numberOfGroups: Int): Map[Int, Set[Int]] = {
    // Generate all possible combos
    val range = (0 until numberOfGroups)
    val combos: Set[(Int, Int)] = range.combinations(2).map(seq => (seq(0), seq(1))).toSet ++ (range zip range).toSet

    // Now, use grid partitioner to assign these block coordinates to partitions


    ???
  }

  /**
    * Gets a GridPartitioner instance from spark mllib using reflection, since the constructor is private.
    */
  private def sneakyGridPartitioner(nRowBlocks: Int, nColBlocks: Int, suggestedNumPartitions: Int): Partitioner = {
    val intClass = classTag[Int].runtimeClass
    val gpObjectClass = Class.forName("org.apache.spark.mllib.linalg.distributed.GridPartitioner$")
    val gpApply = gpObjectClass.getMethod("apply", intClass, intClass, intClass)

    try {
      gpApply.setAccessible(true)
      gpApply.invoke(gpObjectClass.getField("MODULE$").get(null), nRowBlocks: java.lang.Integer,
        nColBlocks: java.lang.Integer, suggestedNumPartitions: java.lang.Integer).asInstanceOf[Partitioner]
    } finally {
      gpApply.setAccessible(false)
    }
  }
}

/**
  *
  * @param matrix Spark IndexedRowMatrix. Entry (i, j) encodes the r value between variants i and j.
  * @param variants Array of variants indexing the rows and columns of the matrix.
  * @param nSamples Number of samples used to compute this matrix.
  */
case class LDMatrix(matrix: IndexedRowMatrix, variants: Array[Variant], nSamples: Int) {
  def toLocalMatrix(): Matrix = {
    matrix.toBlockMatrixDense().toLocalMatrix()
  }
}
