package is.hail.backend.spark

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import is.hail.annotations._
import is.hail.cxx
import is.hail.expr.ir._
import is.hail.expr.types.physical.PTuple
import is.hail.expr.types.virtual._
import is.hail.io.CodecSpec
import is.hail.nativecode._
import is.hail.rvd.{RVDPartitioner, RVDType}
import is.hail.utils._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

object SparkPipeline {
  def local(body: IR): SparkPipeline = SparkPipeline(Map.empty[String, SparkStage], body)
}

case class SparkPipeline(stages: Map[String, SparkStage], body: IR) {
  def codecSpec: CodecSpec = CodecSpec.defaultUncompressed

  def typ: Type = body.typ

  def execute(sc: SparkContext, region: Region): (PTuple, Long) = {
    val fields: IndexedSeq[(String, Type)] =
      stages.map { case (name, stage) => (name, TArray(stage.body.typ)) }.toFastIndexedSeq

    val inputType = TStruct(fields: _*)
    val rvb = new RegionValueBuilder(region)
    rvb.start(inputType.physicalType)
    rvb.startStruct()
    stages.foreach { case (_, stage) => stage.collect(sc, rvb) }
    rvb.endStruct()

    val ref = Ref(genUID(), inputType)
    val node = Subst(body, Env[IR](fields.map { case (name, _) => name -> GetField(ref, name) }.toFastSeq : _*))
    val (typ, f) = Compile[Long, Long](ref.name, inputType.physicalType, MakeTuple(FastSeq(node)))
    (typ.asInstanceOf[PTuple], f(0)(region, rvb.end(), false))
  }
}

case class SparkShuffle(child: SparkStage)

case class SparkBinding(name: String, value: SparkPipeline)

case class SparkStage(
  globals: List[SparkBinding],
  rvdType: RVDType,
  partitioner: RVDPartitioner,
  rdds: Map[String, RDD[RegionValue]],
  contextType: Type, context: Array[Any],
  body: IR) {

  def codecSpec: CodecSpec = CodecSpec.defaultUncompressed

  def collect(sc: SparkContext, rvb: RegionValueBuilder): Unit = {
    assert(rdds.isEmpty)

    val region = rvb.region

    val gType = TStruct(globals.map { b => b.name -> b.value.typ }: _*)
    val rvb2 = new RegionValueBuilder(region)
    rvb2.start(gType.physicalType)
    rvb2.startStruct()
    globals.foreach { case SparkBinding(_, value) =>
      val (typ, off) = value.execute(sc, region)
      if (typ.isFieldDefined(region, off, 0))
        rvb2.addRegionValue(typ.types(0), region, typ.loadField(region, off, 0))
      else
        rvb2.setMissing()
    }
    rvb2.endStruct()

    val gEncoder = codecSpec.buildEncoder(gType.physicalType)
    val globsBC = RegionValue.toBytes(gEncoder, region, rvb2.end()
    val gDecoder = codecSpec.buildDecoder(gType.physicalType, gType.physicalType)

    val env = Env[IR](("context", In(1, contextType)) +: globals.map(b => b.name -> GetField(In(0, gType), b.name)): _*)
    val resultIR = MakeTuple(FastSeq(Subst(body, env)))
    val resultType = TTuple(body.typ).physicalType

    val tub = new cxx.TranslationUnitBuilder()
    tub.include("hail/hail.h")
    tub.include("hail/Utils.h")
    tub.include("hail/Region.h")

    tub.include("<cstring>")
    val ef = tub.buildFunction("process_partition", Array("Region *" -> "region", "const char *" -> "bindings", "const char *" -> "contexts"), "const char *")

    val et = cxx.Emit(ef, 1, resultIR)
    ef +=
      s"""
         |${ et.setup }
         |if (${ et.m }) {
         |  ${ ef.nativeError("return value for partition cannot be missing.") }
         |}
         |return (${ et.v });
       """.stripMargin
    ef.end()

    val wrapper = tub.buildFunction(
      "process_partition_wrapper",
      Array("NativeStatus *" -> "st", "long" -> "region", "long" -> "bindings", "long" -> "contexts"), "long")

    wrapper +=
      s"""
         |try {
         |  return (long) process_partition(((ScalaRegion *)${ wrapper.getArg(1) })->get_wrapped_region(), (char *)${ wrapper.getArg(2) }, (char *)${ wrapper.getArg(3) });
         |} catch (const FatalError& e) {
         |  NATIVE_ERROR(${ wrapper.getArg(0) }, 1005, e.what());
         |  return -1;
         |}
       """.stripMargin
    wrapper.end()
    val mod = tub.end().build("-O2")
    val st = new NativeStatus()
    mod.findOrBuild(st)
    assert(st.ok, st.toString())

    val key = mod.getKey
    val bin = mod.getBinary

    val localContextType = contextType

    val resultEncoder = codecSpec.buildEncoder(resultType)

    val values = sc.parallelize(context, context.length).mapPartitions { ctxIt =>
      val ctx = ctxIt.next()
      val st = new NativeStatus()

      val mod = new NativeModule(key, bin)
      val f = mod.findLongFuncL3(st, "process_partition_wrapper")
      assert(st.ok, st.toString())

      Region.scoped { region =>
        val rvb = new RegionValueBuilder(region)

        val globalsOff = RegionValue.fromBytes(gDecoder, region, globsBC)

        rvb.start(localContextType.physicalType)
        rvb.addAnnotation(localContextType, ctx)
        val contextOff = rvb.end()

        val result = f(st, region.get(), globalsOff, contextOff)
        if (st.fail)
          fatal(st.toString())

        val encoded = RegionValue.toBytes(resultEncoder, region, result)
        Iterator.single(encoded)
      }
    }.collect()

    rvb.startArray(values.length)
    val dec = codecSpec.buildDecoder(resultType, resultType)
    values.foreach { v =>
      val off = RegionValue.fromBytes(dec, region, v)
      if (resultType.isFieldDefined(region, off, 0))
        rvb.addRegionValue(resultType.types(0), region, resultType.loadField(region, off, 0))
      else
        rvb.setMissing()
    }
    rvb.endArray()
  }
}

object LowerTableIR {

  def lower(ir: IR): SparkPipeline = ir match {

    case TableCount(tableIR) =>
      val stage = lower(tableIR)
      val counts = Ref(genUID(), TArray(TInt32()))
      SparkPipeline(Map(counts.name -> stage.copy(body = ArrayLen(stage.body))), invoke("sum", counts))

    case TableGetGlobals(child) =>
      lower(child).globals.head.value

    case TableCollect(child) =>
      val lowered = lower(child)
      val rows = Ref(genUID(), TArray(lowered.body.typ))
      assert(lowered.body.typ.isInstanceOf[TContainer])
      val elt = genUID()
      SparkPipeline(Map(rows.name -> lowered), ArrayFlatMap(rows, elt, Ref(elt, lowered.body.typ)))

    case node if node.children.exists( _.isInstanceOf[TableIR] ) =>
      throw new cxx.CXXUnsupportedOperation("IR nodes with TableIR children must be defined explicitly")

    case _ =>
      val sparkCollects = ir.children.map { case c: IR => lower(c) }
      SparkPipeline(sparkCollects.flatMap(_.stages).toMap, ir.copy(sparkCollects.map(_.body)))
  }

  // table globals should be stored in the first element of `globals` in SparkStage;
  // globals in SparkStage should have unique identifiers.
  def lower(tir: TableIR): SparkStage = tir match {
    case TableRange(n, nPartitions) =>
      val nPartitionsAdj = math.max(math.min(n, nPartitions), 1)
      val partCounts = partition(n, nPartitionsAdj)
      val partStarts = partCounts.scanLeft(0)(_ + _)

      val rvdType = RVDType(tir.typ.rowType.physicalType, Array("idx"))

      val contextType = TStruct(
        "start" -> TInt32(),
        "end" -> TInt32())

      val i = Ref(genUID(), TInt32())

      SparkStage(
        List(SparkBinding(genUID(), SparkPipeline.local(MakeStruct(Seq())))),
        rvdType,
        new RVDPartitioner(Array("idx"), tir.typ.rowType,
          Array.tabulate(nPartitionsAdj) { i =>
            val start = partStarts(i)
            val end = partStarts(i + 1)
            Interval(Row(start), Row(end), includesStart = true, includesEnd = false)
          }),
        Map.empty,
        contextType,
        Array.tabulate(nPartitionsAdj) { i =>
          val start = partStarts(i)
          val end = partStarts(i + 1)
          Row(start, end)
        },
        ArrayMap(ArrayRange(
          GetField(Ref("context", contextType), "start"),
          GetField(Ref("context", contextType), "end"),
          I32(1)), i.name, MakeStruct(FastSeq("idx" -> i))))

    case TableMapGlobals(child, newGlobals) =>
      val loweredChild = lower(child)
      val oldGlobals = loweredChild.globals.head.value
      var loweredGlobals = lower(Let("global", oldGlobals.body, newGlobals))
      loweredGlobals = loweredGlobals.copy(stages = loweredGlobals.stages ++ oldGlobals.stages)
      loweredChild.copy(globals = SparkBinding(genUID(), loweredGlobals) +: loweredChild.globals.tail)

    case TableMapRows(child, newRow) =>
      val loweredChild = lower(child)
      val row = Ref(genUID(), child.typ.rowType)
      val global = loweredChild.globals.head
      val env: Env[IR] = Env("row" -> row, "global" -> Ref(global.name, global.value.typ))
      loweredChild.copy(body = ArrayMap(loweredChild.body, row.name, Subst(newRow, env)))

    case node =>
      throw new cxx.CXXUnsupportedOperation("undefined")
  }
}
