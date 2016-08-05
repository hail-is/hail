package org.apache.spark.sql.execution.datasources.parquet

import java.net.URI
import java.util.concurrent.TimeUnit
import java.util.logging.{Logger => JLogger}
import java.util.{List => JList}

import org.apache.hadoop.fs._
import org.apache.hadoop.io.Writable
import org.apache.hadoop.mapred.FileInputFormat
import org.apache.hadoop.mapreduce._
import org.apache.hadoop.util.StopWatch
import org.apache.parquet.hadoop._
import org.apache.parquet.{Log => ApacheParquetLog}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.{RDD, SqlNewHadoopPartition, SqlNewHadoopRDD}
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.execution.datasources.PartitionSpec
import org.apache.spark.sql.sources._
import org.apache.spark.sql.types.StructType
import org.apache.spark.util.{SerializableConfiguration, Utils}
import org.apache.spark.{Partition => SparkPartition}

import scala.collection.JavaConversions._

class PartitionedParquetRelation(paths: Array[String],
  maybeDataSchema: Option[StructType],
  maybePartitionSpec: Option[PartitionSpec],
  userDefinedPartitionColumns: Option[StructType],
  parameters: Map[String, String])(
  sqlContext: SQLContext) extends ParquetRelation(paths, maybeDataSchema, maybePartitionSpec,
  userDefinedPartitionColumns, parameters)(sqlContext) {
  override def buildScan(
    requiredColumns: Array[String],
    filters: Array[Filter],
    inputFiles: Array[FileStatus],
    broadcastedConf: Broadcast[SerializableConfiguration]): RDD[Row] = {
    val useMetadataCache = sqlContext.getConf(SQLConf.PARQUET_CACHE_METADATA)
    val parquetFilterPushDown = sqlContext.conf.parquetFilterPushDown
    val assumeBinaryIsString = sqlContext.conf.isParquetBinaryAsString
    val assumeInt96IsTimestamp = sqlContext.conf.isParquetINT96AsTimestamp
    val followParquetFormatSpec = sqlContext.conf.followParquetFormatSpec

    // Parquet row group size. We will use this value as the value for
    // mapreduce.input.fileinputformat.split.minsize and mapred.min.split.size if the value
    // of these flags are smaller than the parquet row group size.
    val parquetBlockSize = ParquetOutputFormat.getLongBlockSize(broadcastedConf.value.value)

    // Create the function to set variable Parquet confs at both driver and executor side.
    val initLocalJobFuncOpt =
      ParquetRelation.initializeLocalJobFunc(
        requiredColumns,
        filters,
        dataSchema,
        parquetBlockSize,
        useMetadataCache,
        parquetFilterPushDown,
        assumeBinaryIsString,
        assumeInt96IsTimestamp,
        followParquetFormatSpec) _

    val setInputPaths =
      ParquetRelation.initializeDriverSideJobFunc(inputFiles, parquetBlockSize) _

    Utils.withDummyCallSite(sqlContext.sparkContext) {
      new SqlNewHadoopRDD(
        sc = sqlContext.sparkContext,
        broadcastedConf = broadcastedConf,
        initDriverSideJobFuncOpt = Some(setInputPaths),
        initLocalJobFuncOpt = Some(initLocalJobFuncOpt),
        inputFormatClass = classOf[PartitionedParquetInputFormat[InternalRow]],
        valueClass = classOf[InternalRow]) {

        val cacheMetadata = useMetadataCache

        @transient val cachedStatuses = inputFiles.map { f =>
          // In order to encode the authority of a Path containing special characters such as '/'
          // (which does happen in some S3N credentials), we need to use the string returned by the
          // URI of the path to create a new Path.
          val pathWithEscapedAuthority = escapePathUserInfo(f.getPath)
          new FileStatus(
            f.getLen, f.isDirectory, f.getReplication, f.getBlockSize, f.getModificationTime,
            f.getAccessTime, f.getPermission, f.getOwner, f.getGroup, pathWithEscapedAuthority)
        }.toSeq

        private def escapePathUserInfo(path: Path): Path = {
          val uri = path.toUri
          new Path(new URI(
            uri.getScheme, uri.getRawUserInfo, uri.getHost, uri.getPort, uri.getPath,
            uri.getQuery, uri.getFragment))
        }

        // Overridden so we can inject our own cached files statuses.
        override def getPartitions: Array[SparkPartition] = {
          val inputFormat = new PartitionedParquetInputFormat[InternalRow] {
            override def listStatus(jobContext: JobContext): JList[FileStatus] = {
              if (cacheMetadata) cachedStatuses else super.listStatus(jobContext)
            }
          }

          val jobContext = newJobContext(getConf(isDriverSide = true), jobId)
          val rawSplits = inputFormat.getSplits(jobContext)

          Array.tabulate[SparkPartition](rawSplits.size) { i =>
            new SqlNewHadoopPartition(id, i, rawSplits(i).asInstanceOf[InputSplit with Writable])
          }
        }
      }.asInstanceOf[RDD[Row]] // type erasure hack to pass RDD[InternalRow] as RDD[Row]
    }
  }
}

class PartitionedParquetInputFormat[T] extends ParquetInputFormat[T] {

  val partRegex = "part-r-(\\d+)-.*\\.parquet.*".r

  def getPartNumber(fname: String): Int = {
    fname match {
      case partRegex(i) => i.toInt
      case _ => throw new PathIOException("no match")
    }
  }

  override def getSplits(job: JobContext): JList[InputSplit] = {
    val sw: StopWatch = new StopWatch().start
    val splits: JList[InputSplit] = new java.util.ArrayList[InputSplit]
    val files: JList[FileStatus] = listStatus(job)
    import scala.collection.JavaConverters._

    val sorted = files.asScala.toArray.sortBy(fs => getPartNumber(fs.getPath.getName)).toList
    for (file <- sorted) {
      val path: Path = file.getPath
      val length: Long = file.getLen
      if (length != 0) {
        var blkLocations: Array[BlockLocation] = null
        if (file.isInstanceOf[LocatedFileStatus]) {
          blkLocations = (file.asInstanceOf[LocatedFileStatus]).getBlockLocations
        }
        else {
          val fs: FileSystem = path.getFileSystem(job.getConfiguration)
          blkLocations = fs.getFileBlockLocations(file, 0, length)
        }

        splits.add(makeSplit(path, 0, length, blkLocations(0).getHosts, blkLocations(0).getCachedHosts))
      }
      else {
        splits.add(makeSplit(path, 0, length, new Array[String](0)))
      }
    }
    job.getConfiguration.setLong(FileInputFormat.NUM_INPUT_FILES, files.size)
    sw.stop
    if (FileInputFormat.LOG.isDebugEnabled) {
      FileInputFormat.LOG.debug("Total # of splits generated by getSplits: " + splits.size + ", TimeTaken: " + sw.now(TimeUnit.MILLISECONDS))
    }
    splits
  }
}