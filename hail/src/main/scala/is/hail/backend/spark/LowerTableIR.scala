package is.hail.backend.spark

import is.hail.annotations._
import is.hail.backend.Binding
import is.hail.cxx
import is.hail.expr.ir._
import is.hail.expr.types.virtual._
import is.hail.nativecode._
import is.hail.rvd.{RVDPartitioner, RVDType}
import is.hail.utils._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

case class SparkCollect(
  stages: Map[String, SparkStage],
  body: IR)

case class SparkShuffle(
  child: SparkStage)

case class SparkStage(
  globals: List[Binding],
  rvdType: RVDType,
  partitioner: RVDPartitioner,
  rdds: Map[String, RDD[RegionValue]],
  contextType: Type, context: Array[Any],
  body: IR) {
  def execute(sc: SparkContext): RDD[Any] = {
    assert(rdds.isEmpty)

    val gType = TStruct(globals.map(b => b.name -> b.value.typ): _*)
    val globalRow = BroadcastRow(Row(globals.map(b => Interpret[Any](b.value)): _*), gType, sc)
    val globalRowBc = globalRow.broadcast

    val env = Env[IR]("bindings" -> In(0, gType), "context" -> In(1, contextType))
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

    sc.parallelize(context, context.length).mapPartitions { ctxIt =>
      val ctx = ctxIt.next()
      val st = new NativeStatus()

      val mod = new NativeModule(key, bin)
      val f = mod.findLongFuncL3(st, "process_partition_wrapper")
      assert(st.ok, st.toString())

      Region.scoped { region =>
        val rvb = new RegionValueBuilder(region)

        rvb.start(gType.physicalType)
        rvb.addAnnotation(gType, globalRowBc.value)
        val globalsOff = rvb.end()

        rvb.start(localContextType.physicalType)
        rvb.addAnnotation(localContextType, ctx)
        val contextOff = rvb.end()

        val result = f(st, region.get(), globalsOff, contextOff)
        if (st.fail)
          fatal(st.toString())

        Iterator.single(SafeRow(resultType, region, result).get(0))
      }
    }
  }
}

object LowerTableIR {

  def lower(ir: IR): SparkCollect = ir match {

    case TableCount(tableIR) =>
      val stage = lower(tableIR)
      val counts = Ref(genUID(), TArray(TInt32()))
      SparkCollect(Map(counts.name -> stage.copy(body = ArrayLen(stage.body))), invoke("sum", counts))

    case node if node.children.exists( _.isInstanceOf[TableIR] ) =>
      throw new Exception("IR nodes with TableIR children must be defined explicitly")

    case _ =>
      val sparkCollects = ir.children.map { case c: IR => lower(c) }
      SparkCollect(sparkCollects.flatMap(_.stages).toMap, ir.copy(sparkCollects.map(_.body)))
  }

  def lower(tir: TableIR): SparkStage = tir match {
    case TableRange(n, nPartitions) =>
      val nPartitionsAdj = math.max(math.min(n, nPartitions), 1)
      val partCounts = partition(n, nPartitionsAdj)
      val partStarts = partCounts.scanLeft(0)(_ + _)

      val rvdType = RVDType(tir.typ.rowType.physicalType, Array("idx"))

      val contextType = TStruct(
        "start" -> TInt32(),
        "end" -> TInt32())

      val g = genUID()

      SparkStage(
        List(Binding(g, MakeStruct(Seq()))),
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
        ArrayRange(
          GetField(Ref("context", contextType), "start"),
          GetField(Ref("context", contextType), "end"),
          I32(1)))
  }
}
