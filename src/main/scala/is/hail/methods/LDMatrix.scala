package is.hail.methods

import is.hail.utils._
import is.hail.stats.ToNormalizedIndexedRowMatrix
import is.hail.variant.{Variant, VariantDataset}
import org.apache.spark.Partitioner
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow, IndexedRowMatrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.reflect.classTag


object LDMatrix {


  /*def apply(vds : VariantDataset): LDMatrix = {
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
  }*/

  //TODO Try to replace longs with Int's where possible
  //TODO Variant Index doesn't have to be carried all the way down maybe.
  /**
    * Computes the LD matrix for the given VDS.
    * @param vds VDS on which to compute Pearson correlation between pairs of variants.
    * @param groupSize Size of groups to make in LD matrix computation.
    * @return An LDMatrix.
    */
  def apply(vds: VariantDataset, groupSize: Int): LDMatrix = {
    val nSamples = vds.nSamples

    val bitPackedOpts = vds.rdd.map { case (v, (_, gs)) => LDPrune.toBitPackedVector(gs.hardCallIterator, nSamples)}
    val filterArray = bitPackedOpts.map(!_.isEmpty).collect()
    val originalVariants = vds.variants.collect()
    val variantsKeptArray = originalVariants.zip(filterArray).filter(_._2).map(_._1)
    val bpvs = bitPackedOpts.flatMap(x => x).zipWithIndex().map { case (a, b) => (b, a) }
    val nVariantsDropped = originalVariants.length - variantsKeptArray.length

    info(s"Computing LD Matrix with ${variantsKeptArray.length} variants using $nSamples samples. $nVariantsDropped variants were dropped.")

    //(GroupID, Iterable(VariantID, BPV))
    val grouped: RDD[(Long, Iterable[(Long, LDPrune.BitPackedVector)])] = bpvs.map { case (vIdx, bpv) => (vIdx / groupSize, (vIdx, bpv)) }.groupByKey()
    val numberOfGroups = Math.ceil(grouped.count() / groupSize.toDouble).toInt

    val numBlocksOnDimension = math.ceil(variantsKeptArray.length.toDouble / groupSize).toInt
    val partitioner = sneakyGridPartitioner(numBlocksOnDimension, numBlocksOnDimension, vds.rdd.getNumPartitions)

    val (groupDestinations, partitionToPairsMap) = simulateLDMatrix(numberOfGroups, partitioner)
    val partitionToPairsMapBc = vds.hc.sc.broadcast(partitionToPairsMap)

    //(PartitionID, (GroupID, Iterable(VariantID, BPV)))
    val destinationRDD: RDD[(Int, (Long, Iterable[(Long, LDPrune.BitPackedVector)]))] = grouped.flatMap { case (groupID: Long, itr: Iterable[(Long, LDPrune.BitPackedVector)]) =>
      //Get the set of places this group is needed.
      val destinations = groupDestinations.getOrElse(groupID.toInt, Set.empty)
      destinations.map(dest => (dest, (groupID, itr)))
    }

    //Now with destination RDD created, I have to send groups to their partitions and do the local
    //matrix computation.

    //Consider reduceByKey vs groupByKey
    val computedBlocks: RDD[((Int, Int), Matrix)] = destinationRDD.groupByKey(partitioner).flatMap { case (partitionID, groups: Iterable[(Long, Iterable[(Long, LDPrune.BitPackedVector)])]) =>
      val groupMap = groups.toMap
      val blocksToConstruct = partitionToPairsMapBc.value.get(partitionID).get

      val blocks = blocksToConstruct.map { case (group1, group2) =>
        // 1. Get variants that make up group 1 and group 2, sorted by variant number
        // 2. Since they're sorted, it should be safe to build matrix where entry i, j is computeR(g1Vars(i), g2Vars(j))

        val g1Variants = groupMap.get(group1).get.map(_._2).toArray
        val g2Variants = groupMap.get(group2).get.map(_._2).toArray

        //TODO verify that g1 and g2 Variants are definitely sorted (they should be).
        val dataLength = g1Variants.length * g2Variants.length

        val data = new Array[Double](dataLength)

        var i = 0
        while (i < g2Variants.length) {
          var j = 0
          while (j < g1Variants.length) {
            data(j + i * g2Variants.length) = LDPrune.computeR(g1Variants(j), g2Variants(i))
            j += 1
          }
          i += 1
        }

        ((group1, group2), new DenseMatrix(g1Variants.length, g2Variants.length, data))
      }

      blocks
    }

    val allBlocks = computedBlocks.flatMap { case ((i, j), mat) =>
      if (i == j) List(((i, j), mat)) else List(((i, j), mat), ((j, i), mat.transpose))
     }

    val blockMatrix: BlockMatrix = new BlockMatrix(allBlocks, groupSize, groupSize)
    val irm = blockMatrix.toIndexedRowMatrix()

    LDMatrix(irm, variantsKeptArray, nSamples)

  }

  /**
    * Computes two maps. The first map specifies which partitions a given group is needed on. The second map
    * specifies which pairs must be computed by a given partition.
    * @param numberOfGroups The total number of groups.
    * @param partitioner The partitioner used to divide up the blocks that need to be computed.
    * @return
    */
  private def simulateLDMatrix(numberOfGroups: Int, partitioner: Partitioner): (Map[Int, Set[Int]], Map[Int, Set[(Int, Int)]]) = {
    // Generate all possible combos
    val range = (0 until numberOfGroups)
    val combos: Set[(Int, Int)] = range.combinations(2).map(seq => (seq(0), seq(1))).toSet ++ (range zip range).toSet

    // Now, use grid partitioner to assign these block coordinates to partitions
    val groupToPartitionSet = combos.flatMap{pair =>
      val partition = partitioner.getPartition(pair)
      List((pair._1, partition), (pair._2, partition))
    }
    val groupToPartitionsMap = groupToPartitionSet.groupBy{ case(group, partition) => group}.map{ case(group, set) => (group, set.map(pair => pair._2))}

    //And also get the table that specifies which pairs make up a partition.
    val partitionToPairMap: Map[Int, Set[(Int, Int)]] = combos.groupBy(partitioner.getPartition)

    return (groupToPartitionsMap, partitionToPairMap)
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
