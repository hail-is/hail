package is.hail.expr.ir.lowering

import is.hail.annotations.{BroadcastRow, RegionValue}
import is.hail.backend.ExecuteContext
import is.hail.expr.ir._
import is.hail.expr.ir.analyses.PartitionCounts
import is.hail.rvd.{RVD, RVDPartitioner}
import is.hail.types.physical.PStruct
import is.hail.utils.{fatal, ArrayOfByteArrayInputStream}

object ExecuteRelational {
  def apply(ctx: ExecuteContext, ir: TableIR): TableExecuteIntermediate =
    execute(ctx, LoweringAnalyses(ir, ctx), ir)

  def execute(ctx: ExecuteContext, r: LoweringAnalyses, ir: TableIR): TableExecuteIntermediate = {
    def recur(ir: TableIR): TableExecuteIntermediate = execute(ctx, r, ir)

    ir match {
      case BlockMatrixToTable(child) =>
        TableValueIntermediate(child.execute(ctx).entriesTable(ctx))
      case BlockMatrixToTableApply(bm, aux, function) =>
        val b = bm.execute(ctx)
        val a = CompileAndEvaluate[Any](ctx, aux)
        TableValueIntermediate(function.execute(ctx, b, a))
      case TableAggregateByKey(child, expr) =>
        val prev = recur(child).asTableValue(ctx)
        val extracted = agg.Extract(ctx, expr, r.requirednessAnalysis).independent
        val tv = prev.aggregateByKey(extracted)
        assert(tv.typ == ir.typ, s"${tv.typ}, ${ir.typ}")
        TableValueIntermediate(tv)
      case TableDistinct(child) =>
        val prev = recur(child).asTableValue(ctx)
        TableValueIntermediate(prev.copy(rvd =
          prev.rvd.truncateKey(prev.typ.key).distinctByKey(ctx)
        ))
      case TableExplode(child, path) =>
        val prev = recur(child).asTableValue(ctx)
        TableValueIntermediate(prev.explode(path))
      case TableFilter(child, pred) =>
        val prev = recur(child).asTableValue(ctx)
        TableValueIntermediate(prev.filter(pred))
      case TableFilterIntervals(child, intervals, keep) =>
        val tv = recur(child).asTableValue(ctx)
        val partitioner =
          RVDPartitioner.union(ctx.stateManager, tv.typ.keyType, intervals, tv.typ.keyType.size - 1)
        TableValueIntermediate(
          TableValue(ctx, tv.typ, tv.globals, tv.rvd.filterIntervals(partitioner, keep))
        )
      case ir: TableGen =>
        TableStageIntermediate(LowerTableIR.applyTable(ir, DArrayLowering.All, ctx, r))
      case TableIntervalJoin(left, right, root, product) =>
        val leftTV = recur(left).asTableValue(ctx)
        val rightTV = recur(right).asTableValue(ctx)
        TableValueIntermediate(leftTV.intervalJoin(rightTV, root, product))
      case ir @ TableJoin(left, right, _, _) =>
        val leftTS = recur(left).asTableStage(ctx)
        val rightTS = recur(right).asTableStage(ctx)
        TableExecuteIntermediate(LowerTableIRHelpers.lowerTableJoin(ctx, r, ir, leftTS, rightTS))
      case ir @ TableKeyBy(child, keys, isSorted) =>
        val tv = recur(child).asTableValue(ctx)
        TableValueIntermediate(tv.copy(typ = ir.typ, rvd = tv.rvd.enforceKey(ctx, keys, isSorted)))
      case TableKeyByAndAggregate(child, expr, newKey, nPartitions, bufferSize) =>
        val prev = recur(child).asTableValue(ctx)
        val extracted = agg.Extract(ctx, expr, r.requirednessAnalysis).independent
        TableValueIntermediate(
          prev.keyByAndAggregate(ctx, newKey, extracted, nPartitions, bufferSize)
        )
      case ir @ TableLeftJoinRightDistinct(left, right, root) =>
        val leftValue = recur(left).asTableValue(ctx)
        val rightValue = recur(right).asTableValue(ctx)
        val joinKey = math.min(left.typ.key.length, right.typ.key.length)
        TableValueIntermediate(
          leftValue.copy(
            typ = ir.typ,
            rvd = leftValue.rvd
              .orderedLeftJoinDistinctAndInsert(rightValue.rvd.truncateKey(joinKey), root),
          )
        )
      case TableLiteral(typ, rvd, enc, encodedGlobals) =>
        val (globalPType: PStruct, dec) = enc.buildDecoder(ctx, typ.globalType)
        val bais = new ArrayOfByteArrayInputStream(encodedGlobals)
        val globalOffset = dec.apply(bais, ctx.theHailClassLoader).readRegionValue(ctx.r)
        val globals = BroadcastRow(ctx, RegionValue(ctx.r, globalOffset), globalPType)
        TableValueIntermediate(TableValue(ctx, typ, globals, rvd))
      case TableMapGlobals(child, newGlobals) =>
        TableValueIntermediate(recur(child).asTableValue(ctx).mapGlobals(newGlobals))
      case TableMapRows(child, newRow) =>
        val extracted = agg.Extract(ctx, newRow, r.requirednessAnalysis, isScan = true).independent
        TableValueIntermediate(recur(child).asTableValue(ctx).mapRows(extracted))
      case TableMapPartitions(child, globalName, partitionStreamName, body, _,
            allowedOverlap) =>
        TableValueIntermediate(
          recur(child).asTableValue(ctx)
            .mapPartitions(globalName, partitionStreamName, body, allowedOverlap)
        )
      case TableMultiWayZipJoin(childrenSeq, fieldName, globalName) =>
        val childValues = childrenSeq.map(recur(_).asTableValue(ctx))
        TableValueIntermediate(
          TableValue.multiWayZipJoin(childValues, fieldName, globalName)
        )
      case TableOrderBy(child, sortFields) =>
        TableValueIntermediate(recur(child).asTableValue(ctx).orderBy(sortFields))
      case TableParallelize(rowsAndGlobal, nPartitions) =>
        TableValueIntermediate(TableValue.parallelize(ctx, rowsAndGlobal, nPartitions))
      case ir: TableRange =>
        TableValueIntermediate(TableValue.range(ctx, ir.partitionCounts))
      case TableRead(typ, dropRows, tr) =>
        tr.toExecuteIntermediate(ctx, typ, dropRows)
      case TableRename(child, rowMap, globalMap) =>
        TableValueIntermediate(
          recur(child).asTableValue(ctx).rename(globalMap, rowMap)
        )
      case TableRepartition(child, n, strategy) =>
        TableValueIntermediate(recur(child).asTableValue(ctx).repartition(n, strategy))
      case TableHead(child, n) =>
        val prev = recur(child).asTableValue(ctx)
        TableValueIntermediate(prev.copy(rvd = prev.rvd.head(n, PartitionCounts(child))))
      case TableTail(child, n) =>
        val prev = recur(child).asTableValue(ctx)
        TableValueIntermediate(prev.copy(rvd = prev.rvd.tail(n, PartitionCounts(child))))
      case TableToTableApply(child, function) =>
        TableValueIntermediate(function.execute(ctx, recur(child).asTableValue(ctx)))
      case TableUnion(childrenSeq) =>
        val tvs = childrenSeq.map(recur(_).asTableValue(ctx))
        TableValueIntermediate(
          tvs(0).copy(
            rvd = RVD.union(RVD.unify(ctx, tvs.map(_.rvd)), tvs(0).typ.key.length, ctx)
          )
        )

      case _ => fatal("tried to execute unexecutable IR:\n" + Pretty(ctx, ir))
    }
  }
}
