package org.broadinstitute.hail.variant

import java.util.ArrayList

import htsjdk.samtools.SAMSequenceDictionary
import org.kududb.ColumnSchema.ColumnSchemaBuilder
import org.kududb.{ColumnSchema, Schema, Type}
import org.kududb.client.{CreateTableOptions, KuduClient}

import scala.collection.JavaConversions._

object KuduUtils {

  def normalizeContig(contig: String): String = {
    "%2s".format(contig.replace("chr", ""))
  }

  def buildSchema(): Schema = {
    val cols = new ArrayList[ColumnSchema]
    cols.add(new ColumnSchemaBuilder("contig_norm", Type.STRING).key(true).build)
    cols.add(new ColumnSchemaBuilder("start", Type.INT32).key(true).build)
    cols.add(new ColumnSchemaBuilder("ref", Type.STRING).key(true).build)
    cols.add(new ColumnSchemaBuilder("alt", Type.STRING).key(true).build)
    cols.add(new ColumnSchemaBuilder("sample_group", Type.STRING).key(true).build)
    cols.add(new ColumnSchemaBuilder("contig", Type.STRING).build)
    cols.add(new ColumnSchemaBuilder("annotations", Type.BINARY).nullable(true).build)
    cols.add(new ColumnSchemaBuilder("genotypes_byte_len", Type.INT32).nullable(true).build)
    cols.add(new ColumnSchemaBuilder("genotypes", Type.BINARY).nullable(true).build)
    new Schema(cols)
  }

  def createTableOptions(schema: Schema, seqDict: SAMSequenceDictionary, rowsPerPartition: Int): CreateTableOptions = {
    val rangePartitionCols = new ArrayList[String]
    rangePartitionCols.add("contig_norm")
    rangePartitionCols.add("start")
    val options = new CreateTableOptions().setRangePartitionColumns(rangePartitionCols)
    // e.g. if rowsPerPartition is 10000000 then create split points at
    // (' 1', 0), (' 1', 10000000), (' 1', 20000000), ... (' 1', 240000000)
    // (' 2', 0), (' 2', 10000000), (' 2', 20000000), ... (' 2', 240000000)
    // ...
    // (' Y', 0), (' Y', 10000000), (' Y', 20000000), ... (' Y', 50000000)
    seqDict.getSequences.flatMap(seq => {
      val contig = normalizeContig(seq.getSequenceName)
      val length = seq.getSequenceLength
      Range(0, 1 + length/rowsPerPartition).toList.map(pos => (contig, pos*rowsPerPartition))
    }).map(kv => {
      val partialRow = schema.newPartialRow()
      partialRow.addString("contig_norm", kv._1)
      partialRow.addInt("start", kv._2)
      options.addSplitRow(partialRow)
    })
    options
  }

  def createTableIfNecessary(masterAddress: String, tableName: String,
                             seqDict: SAMSequenceDictionary, rowsPerPartition: Int) {
    val client = new KuduClient.KuduClientBuilder(masterAddress).build
    try {
      if (!client.tableExists(tableName)) {
        println("Table " + tableName + " does not exist")
        val schema = buildSchema()
        client.createTable(tableName, schema, createTableOptions(schema, seqDict, rowsPerPartition))
        println("Table " + tableName + " created")
      }
    } finally {
      client.close()
    }
  }

  def dropTable(masterAddress: String, tableName: String) {
    val client = new KuduClient.KuduClientBuilder(masterAddress).build
    try {
      if (client.tableExists(tableName)) {
        println("Dropping table " + tableName)
        client.deleteTable(tableName)
      }
    } finally {
      client.close()
    }
  }

}
