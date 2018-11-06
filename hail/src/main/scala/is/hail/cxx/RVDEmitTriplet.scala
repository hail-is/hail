package is.hail.cxx

import java.io.{InputStream, OutputStream}

import is.hail.HailContext
import is.hail.annotations.Region
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.types.TStruct
import is.hail.expr.types.physical.{PStruct, PType}
import is.hail.io.CodecSpec
import is.hail.nativecode.{NativeModule, NativeStatus, ObjectArray}
import is.hail.rvd.{OrderedRVDSpec, RVDPartitioner, RVDType}
import is.hail.utils._
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

object RVDEmitTriplet {
  type ProcessRowF = Variable => Code

  def empty[T: ClassTag](typ: RVDType): RVDEmitTriplet = {
    new RVDEmitTriplet(typ, RVDPartitioner.empty(typ), HailContext.get.sc.emptyRDD[Long]) {
      def partitionSetup: Code = ""

      def processRow(f: RVDEmitTriplet.ProcessRowF): Code = ""
    }
  }

  def read(
    path: String,
    t: TStruct,
    codecSpec: CodecSpec,
    partFiles: Array[String],
    requestedType: RVDType,
    partitioner: RVDPartitioner,
    tub: TranslationUnitBuilder
  ): RVDEmitTriplet = {
    val decoder = codecSpec.buildNativeDecoderClass(t.physicalType, requestedType.rowType, tub)
    tub += decoder

    tub.include("hail/Upcalls.h")
    tub.include("hail/Region.h")
    tub.include("hail/hail.h")
    tub.include("hail/Decoder.h")
    tub.include("hail/ObjectArray.h")
    tub.include("<memory>")

    val hc = HailContext.get
    val partsRDD = hc.readPartitions(path, partFiles, (_, is, m) => Iterator.single(new ObjectArray(is).get()))

    new RVDEmitTriplet(requestedType, Option(partitioner).getOrElse(RVDPartitioner.unkeyed(partsRDD.getNumPartitions)), partsRDD) {
      private[this] val up = Variable("up", "UpcallEnv")
      private[this] val dec = Variable("decoder", decoder.name, s"std::make_shared<InputStream>($up, reinterpret_cast<ObjectArray *>($iterator)->at(0))")
      def partitionSetup: Code =
        s"""
           |${ up.define }
           |${ dec.define }
         """.stripMargin

      def processRow(rowF: ProcessRowF): Code = {
        val row = Variable("row", "char *", s"$dec.decode_row($st, $region)")
          s"""
             |while($dec.decode_byte($st)) {
             |  ${ row.define }
             |  ${ rowF(row) }
             |}
           """.stripMargin
      }
    }
  }

  def write[T](t: RVDEmitTriplet, tub: TranslationUnitBuilder, path: String, stageLocally: Boolean, codecSpec: CodecSpec): Array[Long] = {
    val encClass = codecSpec.buildNativeEncoderClass(t.typ.rowType, tub)
    tub += encClass
    tub.include("hail/Encoder.h")

    val os = Variable("os", "long")
    val regionPtr = Variable("region", "long")
    val processPartition = new FunctionBuilder("process_partition", Array(t.st, regionPtr, t.iterator, os), "long")

    processPartition += t.region.defineWith(s"reinterpret_cast<Region*>($regionPtr)")
    processPartition += t.partitionSetup

    val up = Variable("up", "UpcallEnv")
    val enc = Variable("encoder", encClass.name, s"std::make_shared<OutputStream>($up, reinterpret_cast<ObjectArray *>($os)->at(0))")
    val nRows = Variable("n_rows", "long", "0")

    processPartition +=
      s"""${ up.define }
         |${ enc.define }
         |${ nRows.define }
         """.stripMargin

    processPartition += t.processRow { row: Variable =>
      s"""
         |$enc.encode_byte(${ t.st }, 1);
         |$enc.encode_row(${ t.st }, $row);
         |++$nRows;
       """.stripMargin
    }

    processPartition +=
      s"""
         |$enc.encode_byte(${ t.st }, 0);
         |$enc.flush(${ t.st });
         |return $nRows;
       """.stripMargin

    tub += processPartition.result()

    val mod = tub.result().build("-O2 -llz4")
    val modKey = mod.getKey
    val modBinary = mod.getBinary

    val writeF = { (it: Iterator[Long], os: OutputStream) =>
      val st = new NativeStatus()
      val mod = new NativeModule(modKey, modBinary)
        val f = mod.findLongFuncL3(st, "process_partition")
        assert(st.ok, st.toString())
        val osArray = new ObjectArray(os)
      Region.scoped { region =>
        val nRows = f(st, region.get(), it.next(), osArray.get())
        assert(st.ok, st.toString())
        f.close()
        osArray.close()
        mod.close()
        st.close()
        os.flush()
        os.close()
        nRows
      }
    }

    val (partFiles, partitionCounts) = t.rddBase.writePartitions(path, stageLocally, writeF)

    t.rvdSpec(codecSpec, partFiles).write(HailContext.get.sc.hadoopConfiguration, path)
    partitionCounts
  }
}

abstract class RVDEmitTriplet(val typ: RVDType, val partitioner: RVDPartitioner, val rddBase: RDD[Long]) {

  val st: Variable = Variable("st", "NativeStatus*")
  val region: Variable = Variable("region", "Region*")
  val iterator: Variable = Variable("iterator", "long")

  def partitionSetup: Code

  def processRow(f: RVDEmitTriplet.ProcessRowF): Code

  def rvdSpec(codecSpec: CodecSpec, partFiles: Array[String]) = OrderedRVDSpec(
    typ,
    codecSpec,
    partFiles,
    JSONAnnotationImpex.exportAnnotation(
      partitioner.rangeBounds.toFastSeq,
      partitioner.rangeBoundsType))

  def mapPartitions(rowType: PStruct, partSetup: Code)(mapRow: RVDEmitTriplet.ProcessRowF => RVDEmitTriplet.ProcessRowF): RVDEmitTriplet = {
    val old = this
    new RVDEmitTriplet(typ.copy(rowType = rowType), partitioner, rddBase) {

      override val st: Variable = old.st
      override val region: Variable = old.region
      override val iterator: Variable = old.iterator

      def partitionSetup: Code = s"${ old.partitionSetup }\n$partSetup"

      def processRow(f: RVDEmitTriplet.ProcessRowF): Code = old.processRow(mapRow(f))
    }
  }
}
