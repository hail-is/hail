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

  //TODO Try to replace longs with Int's where possible
  //TODO Variant Index doesn't have to be carried all the way down maybe.
  def apply2(vds: VariantDataset, groupSize: Int, numBlocksOnDimension: Int): LDMatrix = {
    val nSamples = vds.nSamples

    val bitPackedOpts = vds.rdd.map { case (v, (_, gs)) => LDPrune.toBitPackedVector(gs.hardCallIterator, nSamples)}
    val filterArray = bitPackedOpts.map(!_.isEmpty).collect()
    val originalVariants = vds.variants.collect()
    val variantsKeptArray = originalVariants.zip(filterArray).filter(_._2).map(_._1)
    val bpvs = bitPackedOpts.flatMap(x => x).zipWithIndex().map { case (a, b) => (b, a) }

    // Chunk out BPVs with block id's also, use grid partitioner to identify destinations, then map destinations over to
    // send vectors there, and then use Jackie method to smash individual BPV's together.

    //(GroupID, Iterable(VariantID, BPV))
    val grouped: RDD[(Long, Iterable[(Long, LDPrune.BitPackedVector)])] = bpvs.map { case (vIdx, bpv) => (vIdx / groupSize, (vIdx, bpv)) }.groupByKey()
    val numberOfGroups = Math.ceil(grouped.count() / groupSize.toDouble).toInt

    val partitioner = sneakyGridPartitioner(numBlocksOnDimension, numBlocksOnDimension, vds.rdd.getNumPartitions)

    val (groupDestinations: Map[Int, Set[Int]], partitionToPairsMap: Map[Int, Set[(Int, Int)]]) = simulateLDMatrix(numberOfGroups, partitioner)
    val partitionToPairsMapBc = vds.hc.sc.broadcast(partitionToPairsMap)

    //(PartitionID, (GroupID, Iterable(VariantID, BPV)))
    val destinationRDD: RDD[(Int, (Long, Iterable[(Long, LDPrune.BitPackedVector)]))] = grouped.flatMap { case (groupID: Long, itr: Iterable[(Long, LDPrune.BitPackedVector)]) =>
      //Get the set of places this group is needed.
      val destinations = groupDestinations.getOrElse(groupID.toInt, Set.empty)
      destinations.map(dest => (dest, (groupID, itr)))
    }

    //Now with destination RDD created, I have to send groups to their partitions and do the local
    //matrix computation.

    //Steps:
    // 1. GroupBy to get everything associated with its partition number
    // 2. Make it such that any BPV group can be accessed in constant time by its groupID.
    // 3. Using the task list, construct the specified blocks.

    //Consider reduceByKey vs groupByKey
    val computedBlocks: RDD[((Int, Int), Matrix)] = destinationRDD.groupByKey(partitioner).flatMap { case (partitionID, groups: Iterable[(Long, Iterable[(Long, LDPrune.BitPackedVector)])]) =>
      val groupMap = groups.toMap
      val blocksToConstruct = partitionToPairsMapBc.value.get(partitionID).get

      val blocks = blocksToConstruct.map { case (group1, group2) =>
        // 1. Get variants that make up group 1 and group 2, sorted by variant number
        // 2. Since they're sorted, it should be safe to build matrix where entry i, j is computeR(g1Vars(i), g2Vars(j))

        val g1Variants = groupMap.get(group1).get.map(_._2).toArray
        val g2Variants = groupMap.get(group2).get.map(_._2).toArray

        //TODO verify that g1 and g2 Variants are definitely sorted.

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


    //Realize I've only computed upper triangle of blocks, so need to transpose to get rest of it.
    val allBlocks = computedBlocks.flatMap { case ((i, j), mat) =>
      if (i == j) List(((i, j), mat)) else List(((i, j), mat), ((j, i), mat.transpose))
     }

    //Construct block matrix and convert to IndexedRowMatrix

    val blockMatrix: BlockMatrix = new BlockMatrix(allBlocks, groupSize, groupSize)
    val irm = blockMatrix.toIndexedRowMatrix()

    //Make LD Matrix
    LDMatrix(irm, variantsKeptArray, nSamples)

  }

  //Create a Map from group number to Set of partitions where this chunk is needed.
  private def simulateLDMatrix(numberOfGroups: Int, partitioner: Partitioner): (Map[Int, Set[Int]], Map[Int, Set[(Int, Int)]]) = {
    // Generate all possible combos
    val range = (0 until numberOfGroups)
    val combos: Set[(Int, Int)] = range.combinations(2).map(seq => (seq(0), seq(1))).toSet ++ (range zip range).toSet

    // Now, use grid partitioner to assign these block coordinates to partitions
    //TODO: Don't get partition twice for no reason.
    val groupToPartitionSet = combos.flatMap(pair => List((pair._1, partitioner.getPartition(pair)), (pair._2, partitioner.getPartition(pair))))
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
