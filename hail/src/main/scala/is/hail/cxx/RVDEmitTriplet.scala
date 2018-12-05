package is.hail.cxx

import java.io.OutputStream

import is.hail.HailContext
import is.hail.annotations.Region
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.types.physical.{PStruct, PType}
import is.hail.expr.types.virtual.TStruct
import is.hail.io.CodecSpec
import is.hail.nativecode.{NativeModule, NativeStatus, ObjectArray}
import is.hail.rvd.{OrderedRVDSpec, RVDPartitioner, RVDType}
import is.hail.utils._
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

object RVDEmitTriplet {
  type ProcessRowF = Variable => Code

  def empty[T: ClassTag](typ: RVDType): RVDEmitTriplet = {
    new RVDEmitTriplet(
      BaseRVD(RVDPartitioner.empty(typ), HailContext.get.sc.emptyRDD[Long]), typ,
      "", Variable("??", "??"), Variable("??", "??"))
  }

  def read(
    path: String,
    t: PStruct,
    codecSpec: CodecSpec,
    partFiles: Array[String],
    requestedType: RVDType,
    partitioner: RVDPartitioner,
    tub: TranslationUnitBuilder
  ): RVDEmitTriplet = {
    val decoder = codecSpec.buildNativeDecoderClass(t, requestedType.rowType, tub)
    tub.include("hail/Decoder.h")
    tub.include("hail/ObjectArray.h")
    tub.include("<memory>")

    val hc = HailContext.get
    val partsRDD = hc.readPartitions(path, partFiles, (_, is, m) => Iterator.single(new ObjectArray(is).get()))

    val rvdBase = BaseRVD(
      Option(partitioner).getOrElse(RVDPartitioner.unkeyed(partsRDD.getNumPartitions)),
      partsRDD)
    val is = Variable.make_shared("is", "InputStream", rvdBase.up.toString, s"reinterpret_cast<ObjectArray *>(${ rvdBase.rddInput })->at(0)")
    val reader = Variable.make_shared("reader", s"Reader<${ decoder.name }>", s"${decoder.name}($is)", rvdBase.region.toString, rvdBase.st.toString)
    val iterator = Variable("it", s"Reader<${ decoder.name }> *", s"$reader.get()")
    val begin = Variable("it", s"Reader<${ decoder.name }>::Iterator", s"$iterator->begin()")
    val end = Variable("it", s"Reader<${ decoder.name }>::Iterator", s"$iterator->end()")
    val setup =
      s"""
         |${ is.define }
         |${ reader.define }
         |${ iterator.define }
         |${ begin.define }
         |${ end.define }
       """.stripMargin

    RVDEmitTriplet(rvdBase, requestedType, setup, begin, end)
  }

  def write[T](t: RVDEmitTriplet, tub: TranslationUnitBuilder, path: String, stageLocally: Boolean, codecSpec: CodecSpec): Array[Long] = {
    val encClass = codecSpec.buildNativeEncoderClass(t.typ.rowType, tub)
    tub.include("hail/Encoder.h")

    val os = Variable("os", "long")
    val enc = Variable("encoder", encClass.name, s"std::make_shared<OutputStream>(${ t.rvd.up }, reinterpret_cast<ObjectArray *>($os)->at(0))")
    val nRows = Variable("n_rows", "long", "0")
    val it = t.iterator
    val end = t.end

    tub += new Function(
      "long", "process_partition", Array(t.st, t.rvd.regionLong, t.rvd.rddInput, os),
      s"""
         |${ t.rvd.setup(tub) }
         |${ t.setup }
         |${ enc.define }
         |${ nRows.define }
         |while($it != $end) {
         |  $enc.encode_byte(${ t.st }, 1);
         |  $enc.encode_row(${ t.st }, *$it);
         |  ++$nRows;
         |  ++$it;
         |}
         |$enc.encode_byte(${ t.st }, 0);
         |$enc.flush(${ t.st });
         |return $nRows;
       """.stripMargin)

    val mod = tub.end().build("-O2 -llz4")
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

    val (partFiles, partitionCounts) = t.rvd.rddBase.writePartitions(path, stageLocally, writeF)

    t.rvdSpec(codecSpec, partFiles).write(HailContext.get.sc.hadoopConfiguration, path)
    partitionCounts
  }
}

case class BaseRVD(partitioner: RVDPartitioner, rddBase: RDD[Long]) {
  val st: Variable = Variable("st", "NativeStatus*")
  val up: Variable = Variable("up", "UpcallEnv")
  val regionLong: Variable = Variable("region", "long")
  val region: Variable = Variable("region", "ScalaRegion*")
  val rddInput: Variable = Variable("input_objects", "long")

  def setup(t: TranslationUnitBuilder): Code = {
    t.include("hail/Region.h")
    t.include("hail/NativeStatus.h")
    t.include("hail/Upcalls.h")

    s"""
       |${ up.define }
       |${ region.defineWith(s"reinterpret_cast<ScalaRegion *>($regionLong)") }
     """.stripMargin
  }
}

case class RVDEmitTriplet(rvd: BaseRVD, typ: RVDType, setup: Code, iterator: Variable, end: Variable) {

  def region: Variable = rvd.region
  def st: Variable = rvd.st
  def up: Variable = rvd.up

  def rvdSpec(codecSpec: CodecSpec, partFiles: Array[String]) = OrderedRVDSpec(
    typ,
    codecSpec,
    partFiles,
    JSONAnnotationImpex.exportAnnotation(
      rvd.partitioner.rangeBounds.toFastSeq,
      rvd.partitioner.rangeBoundsType))
}