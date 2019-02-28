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
    val node = MakeTuple(FastSeq(Subst(body, Env[IR](fields.map { case (name, _) => name -> GetField(ref, name) }.toFastSeq : _*))))
    val f = cxx.Compile(ref.name, inputType.physicalType, node, optimize = true)
    (node.pType.asInstanceOf[PTuple], f(region.get(), rvb.end()))
  }
}

case class SparkShuffle(child: SparkStage)

case class SparkBinding(name: String, value: SparkPipeline)

case class SparkStage(
  globals: SparkBinding,
  otherBroadcastVals: List[SparkBinding],
  rvdType: RVDType,
  partitioner: RVDPartitioner,
  rdds: Map[String, RDD[RegionValue]],
  contextType: Type,
  context: Array[Any],
  body: IR) {

  val broadcastVals: List[SparkBinding] = globals +: otherBroadcastVals

  def codecSpec: CodecSpec = CodecSpec.defaultUncompressed

  def collect(sc: SparkContext, rvb: RegionValueBuilder): Unit = {
    assert(rdds.isEmpty)

    val region = rvb.region

    val gType = TStruct(broadcastVals.map { b => b.name -> b.value.typ }: _*)
    val rvb2 = new RegionValueBuilder(region)
    rvb2.start(gType.physicalType)
    rvb2.startStruct()
    broadcastVals.foreach { case SparkBinding(_, value) =>
      val (typ, off) = value.execute(sc, region)
      rvb2.addField(typ, region, off, 0)
    }
    rvb2.endStruct()

    val gEncoder = codecSpec.buildEncoder(gType.physicalType)
    val globsBC = RegionValue.toBytes(gEncoder, region, rvb2.end())
    val gDecoder = codecSpec.buildDecoder(gType.physicalType, gType.physicalType)

    val bindings = genUID()
    val env = Env[IR](broadcastVals.map(b => b.name -> GetField(Ref(bindings, gType), b.name)): _*)
    val resultIR = MakeTuple(FastSeq(Subst(body, env)))
    val resultType = TTuple(body.typ).physicalType

    val tub = new cxx.TranslationUnitBuilder()
    val f = cxx.Compile.makeNonmissingFunction(tub, resultIR,
      bindings -> gType.physicalType,
      "context" -> contextType.physicalType)

    val wrapper = tub.buildFunction(
      "process_partition_wrapper",
      Array("NativeStatus *" -> "st", "long" -> "region", "long" -> "bindings", "long" -> "contexts"), "long")

    wrapper +=
      s"""
         |try {
         |  return (long) ${ f.name }(((ScalaRegion *)${ wrapper.getArg(1) })->region_, (char *)${ wrapper.getArg(2) }, (char *)${ wrapper.getArg(3) });
         |} catch (const FatalError& e) {
         |  NATIVE_ERROR(${ wrapper.getArg(0) }, 1005, e.what());
         |  return -1;
         |}
       """.stripMargin
    wrapper.end()
    val mod = tub.end().build("-O2")
    mod.findOrBuild()

    val key = mod.getKey
    val bin = mod.getBinary

    val localContextType = contextType

    val resultEncoder = codecSpec.buildEncoder(resultType)

    val values = sc.parallelize(context, context.length).mapPartitions { ctxIt =>
      val ctx = ctxIt.next()
      assert(!ctxIt.hasNext)

      val mod = new NativeModule(key, bin)
      val f = mod.findLongFuncL3("process_partition_wrapper")

      Region.scoped { region =>
        val rvb = new RegionValueBuilder(region)

        val globalsOff = RegionValue.fromBytes(gDecoder, region, globsBC)

        rvb.start(localContextType.physicalType)
        rvb.addAnnotation(localContextType, ctx)
        val contextOff = rvb.end()

        val result = f(region.get(), globalsOff, contextOff)

        val encoded = RegionValue.toBytes(resultEncoder, region, result)
        Iterator.single(encoded)
      }
    }.collect()

    rvb.startArray(values.length)
    val dec = codecSpec.buildDecoder(resultType, resultType)
    values.foreach { v =>
      rvb.addField(resultType, region, RegionValue.fromBytes(dec, region, v), 0)
    }
    rvb.endArray()
  }
}

object LowerTableIR {

  def lower(ir: IR): SparkPipeline = ir match {

    case TableCount(tableIR) =>
      val stage = lower(tableIR)
      val counts = Ref(genUID(), TArray(TInt64()))
      SparkPipeline(
        Map(counts.name -> stage.copy(body = Cast(ArrayLen(stage.body), TInt64()))),
        invoke("sum", counts))

    case TableGetGlobals(child) =>
      lower(child).globals.value

    case TableCollect(child) =>
      val lowered = lower(child)
      val globals = lowered.globals
      val rows = Ref(genUID(), TArray(lowered.body.typ))
      assert(lowered.body.typ.isInstanceOf[TContainer])
      val elt = genUID()
      SparkPipeline(
        Map((rows.name, lowered) +: globals.value.stages.toSeq: _*),
        MakeStruct(FastIndexedSeq(
          "rows" -> ArrayFlatMap(rows, elt, Ref(elt, lowered.body.typ)),
          "global" -> lowered.globals.value.body)))

    case node if node.children.exists( _.isInstanceOf[TableIR] ) =>
      throw new cxx.CXXUnsupportedOperation(s"IR nodes with TableIR children must be defined explicitly: \n${ Pretty(node) }")

    case node if node.children.exists( _.isInstanceOf[MatrixIR] ) =>
      throw new cxx.CXXUnsupportedOperation(s"MatrixIR nodes must be lowered to TableIR nodes separately: \n${ Pretty(node) }")

    case _ =>
      val pipelines = ir.children.map { case c: IR => lower(c) }
      SparkPipeline(pipelines.flatMap(_.stages).toMap, ir.copy(pipelines.map(_.body)))
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
        SparkBinding(genUID(), SparkPipeline.local(MakeStruct(Seq()))),
        List(),
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
      val oldGlobals = loweredChild.globals.value
      var loweredGlobals = lower(Let("global", oldGlobals.body, newGlobals))
      loweredGlobals = loweredGlobals.copy(stages = loweredGlobals.stages ++ oldGlobals.stages)
      loweredChild.copy(globals = SparkBinding(genUID(), loweredGlobals))

    case TableMapRows(child, newRow) =>
      val loweredChild = lower(child)
      val row = Ref(genUID(), child.typ.rowType)
      val global = loweredChild.globals
      val env: Env[IR] = Env("row" -> row, "global" -> Ref(global.name, global.value.typ))
      loweredChild.copy(body = ArrayMap(loweredChild.body, row.name, Subst(newRow, env)))

    case node =>
      throw new cxx.CXXUnsupportedOperation(s"undefined: \n${ Pretty(node) }")
  }
}
