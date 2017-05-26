package is.hail.methods

import is.hail.utils._
import is.hail.variant.{Variant, VariantDataset}
import org.apache.spark.Partitioner
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRowMatrix}
import org.apache.spark.rdd.RDD

import scala.reflect.classTag


object LDMatrix {

  /**
    * Computes the LD matrix for the given VDS.
    * @param vds VDS on which to compute Pearson correlation between pairs of variants.
    * @param blockSize Size of blocks to make in LD matrix computation.
    * @return An LDMatrix.
    */
  def apply(vds: VariantDataset, blockSize: Int): LDMatrix = {
    /*
    This works by first breaking the set of all variants into "groups". Then simulateLDMatrix determines the partitions
    where each group is needed. Finally, the groups are sent to their partitions, where they are combined in pairs with other
    groups to form the "blocks" of a BlockMatrix. The dimensions of the blocks are the same as the sizes of the groups that
    form them. The ids of the two groups that formed a block specify the block's coordinates in the block matrix.
     */

    val nSamples = vds.nSamples

    val bitPackedOpts = vds.rdd.map { case (_, (_, gs)) => LDPrune.toBitPackedVector(gs.hardCallIterator, nSamples)}
    val filterArray = bitPackedOpts.map(!_.isEmpty).collect()
    val originalVariants = vds.variants.collect()
    val variantsKeptArray = originalVariants.zip(filterArray).filter(_._2).map(_._1)
    val nVariantsDropped = originalVariants.length - variantsKeptArray.length

    val bpvs = bitPackedOpts.flatMap(x => x).zipWithIndex()

    info(s"Computing LD Matrix with ${variantsKeptArray.length} variants using $nSamples samples. $nVariantsDropped variants were dropped.")

    val trueBlockSize = math.min(blockSize, variantsKeptArray.length)

    val grouped = bpvs.map { case (bpv, vIdx) => (vIdx / trueBlockSize, (vIdx, bpv)) }
      .groupByKey()
      .map{ case (groupID, it) => (groupID, it.toArray.sortBy{ case (vIdx, bpv) => vIdx }.map(_._2)) }

    val numberOfGroups = Math.ceil(grouped.count() / trueBlockSize.toDouble).toInt

    val numBlocksOnDimension = math.ceil(variantsKeptArray.length.toDouble / trueBlockSize).toInt
    val partitioner = reflectGridPartitioner(numBlocksOnDimension, numBlocksOnDimension, vds.rdd.getNumPartitions)

    val (groupDestinations, partitionToPairsMap) = simulateLDMatrix(numberOfGroups, partitioner)
    val partitionToPairsMapBc = vds.hc.sc.broadcast(partitionToPairsMap)

    val destinationRDD: RDD[(Int, (Long, Array[LDPrune.BitPackedVector]))] = grouped.flatMap { case (groupID: Long, arr: Array[LDPrune.BitPackedVector]) =>
      val destinations = groupDestinations.getOrElse(groupID.toInt, Set.empty)
      destinations.map(dest => (dest, (groupID, arr)))
    }

    val computedBlocks: RDD[((Int, Int), Matrix)] = destinationRDD.groupByKey(partitioner).flatMap { case (partitionID, groups) =>
      val groupMap = groups.toMap
      val blocksToConstruct = partitionToPairsMapBc.value.get(partitionID).get

      blocksToConstruct.map { case (group1, group2) =>
        // These variants are in correct sorted order from time of construction of grouped RDD.
        // Since they're sorted, it is safe to build matrix where entry i, j is computeR(g1Vars(i), g2Vars(j))
        val g1Variants = groupMap.get(group1).get
        val g2Variants = groupMap.get(group2).get

        val matrixEntries = new Array[Double](g1Variants.length * g2Variants.length)

        var i = 0
        while (i < g2Variants.length) {
          var j = 0
          while (j < g1Variants.length) {
            matrixEntries(j + i * g2Variants.length) = LDPrune.computeR(g1Variants(j), g2Variants(i))
            j += 1
          }
          i += 1
        }

        ((group1, group2), new DenseMatrix(g1Variants.length, g2Variants.length, matrixEntries))
      }
    }

    val allBlocks = computedBlocks.flatMap { case ((i, j), mat) =>
      if (i == j) List(((i, j), mat)) else List(((i, j), mat), ((j, i), mat.transpose))
    }

    val blockMatrix: BlockMatrix = new BlockMatrix(allBlocks, trueBlockSize, trueBlockSize)

    LDMatrix(blockMatrix.toIndexedRowMatrix(), variantsKeptArray, nSamples)
  }

  /**
    * Computes two maps. The first map specifies which partitions a given group is needed on. The second map
    * specifies which blocks must be computed by a given partition.
    * @param numberOfGroups The total number of groups.
    * @param partitioner The partitioner used to divide up the blocks that need to be computed.
    * @return (map from groups to set of partitions, map from partition to set of blocks to compute)
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
    * Gets a GridPartitioner instance from spark MLLib using reflection, since the constructor is private.
    * This should probably go on some kind of MLLib Utils object.
    */
  def reflectGridPartitioner(nRowBlocks: Int, nColBlocks: Int, suggestedNumPartitions: Int): Partitioner = {
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
