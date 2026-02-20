package is.hail.backend

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.collection.implicits.toRichIndexedSeq
import is.hail.expr.ir.Compiled
import is.hail.expr.ir.analyses.SemanticHash
import is.hail.expr.ir.lowering.TableStageDependency
import is.hail.utils._

object BackendUtils {
  type F = AsmFunction3[Region, Array[Byte], Array[Byte], Array[Byte]]
}

class BackendUtils(mods: Array[(String, Compiled[BackendUtils.F])]) extends Logging {

  private[this] val getModule = mods.toMap

  def collectDArray(
    ctx: DriverRuntimeContext,
    modID: String,
    contexts: Array[Array[Byte]],
    globals: Array[Byte],
    stageName: String,
    tsd: Option[TableStageDependency],
  ): Array[Array[Byte]] = {
    val (failureOpt, results) = runCDA(ctx, globals, contexts, None, modID, stageName, tsd)
    failureOpt.foreach(throw _)
    Array.tabulate[Array[Byte]](results.length)(results(_)._1)
  }

  def ccCollectDArray(
    ctx: DriverRuntimeContext,
    modID: String,
    contexts: Array[Array[Byte]],
    globals: Array[Byte],
    stageName: String,
    semhash: SemanticHash.Type,
    tsd: Option[TableStageDependency],
  ): Array[Array[Byte]] = {

    val cachedResults = ctx.executionCache.lookup(semhash)
    logger.info(s"$stageName: found ${cachedResults.length} entries for $semhash.")

    val todo =
      contexts
        .indices
        .filterNot(k => cachedResults.containsOrdered[Int](k, _ < _, _._2))

    val (failureOpt, successes) =
      todo match {
        case Seq() =>
          (None, IndexedSeq.empty)

        case partitions =>
          runCDA(ctx, globals, contexts, Some(partitions), modID, stageName, tsd)
      }

    val results = merge[(Array[Byte], Int)](cachedResults, successes, _._2 < _._2)

    ctx.executionCache.put(semhash, results)
    logger.info(s"$stageName: cached ${results.length} entries for $semhash.")

    failureOpt.foreach(throw _)
    Array.tabulate[Array[Byte]](results.length)(results(_)._1)
  }

  private[this] def runCDA(
    rtx: DriverRuntimeContext,
    globals: Array[Byte],
    contexts: Array[Array[Byte]],
    partitions: Option[IndexedSeq[Int]],
    modID: String,
    stageName: String,
    tsd: Option[TableStageDependency],
  ): (Option[Throwable], IndexedSeq[(Array[Byte], Int)]) = {

    val loadFn = getModule(modID)
    val start = System.nanoTime()

    val r = rtx.mapCollectPartitions(
      globals,
      contexts,
      stageName,
      tsd,
      partitions,
    ) {
      (hcl, fs, htc, r) =>
        val fn = loadFn(hcl, fs, htc, r)
        fn(r, _, _)
    }

    val elapsed = System.nanoTime() - start
    val nTasks = partitions.map(_.length).getOrElse(contexts.length)
    logger.info(s"$stageName: executed $nTasks tasks in ${formatTime(elapsed)}")

    r
  }
}
