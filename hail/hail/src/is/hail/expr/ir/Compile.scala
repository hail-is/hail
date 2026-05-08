package is.hail.expr.ir

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.{ExecuteContext, HailTaskContext}
import is.hail.expr.ir.agg.AggStateSig
import is.hail.expr.ir.defs.In
import is.hail.expr.ir.lowering.LoweringPipeline
import is.hail.expr.ir.streams.EmitStream
import is.hail.io.fs.FS
import is.hail.types.physical.{PStruct, PType}
import is.hail.types.physical.stypes.{PTypeReferenceSingleCodeType, SingleCodeType}
import is.hail.types.physical.stypes.interfaces.{NoBoxLongIterator, SStreamConcrete}
import is.hail.types.virtual.TStream

import scala.collection.mutable

import java.io.PrintWriter

import sourcecode.Enclosing

case class CompileCacheKey(
  aggSigs: IndexedSeq[AggStateSig],
  args: Seq[EmitParamType],
  body: IR,
)

private[ir] class NoBoxLongIteratorAdapter(it: NoBoxLongIterator) extends Iterator[java.lang.Long] {
  private var _stepped = false
  private var _hasNext = false
  private var _value: Long = 0L

  override def hasNext: Boolean = {
    if (!_stepped) {
      _value = it.next()
      _hasNext = !it.eos
      _stepped = true
      if (!_hasNext) it.close()
    }
    _hasNext
  }

  override def next(): java.lang.Long = {
    if (!hasNext) Iterator.empty.next(): Unit
    _stepped = false
    _value
  }
}

private[ir] trait CompileOps {

  type Compiled[A] = (HailClassLoader, FS, HailTaskContext, Region) => A
  type CompiledFunction[A] = (Option[SingleCodeType], Compiled[A])
  type CompileCache = mutable.Map[CompileCacheKey, CompiledFunction[_]]

  def Compile[F: TypeInfo](
    ctx: ExecuteContext,
    params: IndexedSeq[(Name, EmitParamType)],
    expectedCodeParamTypes: IndexedSeq[TypeInfo[_]],
    expectedCodeReturnType: TypeInfo[_],
    body: IR,
    print: Option[PrintWriter] = None,
  ): (Option[SingleCodeType], Compiled[F]) =
    Impl[F, Any](
      ctx,
      params,
      None,
      expectedCodeParamTypes,
      expectedCodeReturnType,
      body,
      print,
    )

  def CompileWithAggregators[F: TypeInfo](
    ctx: ExecuteContext,
    aggSigs: IndexedSeq[AggStateSig],
    params: IndexedSeq[(Name, EmitParamType)],
    expectedCodeParamTypes: IndexedSeq[TypeInfo[_]],
    expectedCodeReturnType: TypeInfo[_],
    body: IR,
    print: Option[PrintWriter] = None,
  ): (
    Option[SingleCodeType],
    Compiled[F with FunctionWithAggRegion],
  ) =
    Impl[F, FunctionWithAggRegion](
      ctx,
      params,
      Some(aggSigs),
      expectedCodeParamTypes,
      expectedCodeReturnType,
      body,
      print,
    )

  private[this] def Impl[F: TypeInfo, Mixin](
    ctx: ExecuteContext,
    params: IndexedSeq[(Name, EmitParamType)],
    aggSigs: Option[IndexedSeq[AggStateSig]],
    expectedCodeParamTypes: IndexedSeq[TypeInfo[_]],
    expectedCodeReturnType: TypeInfo[_],
    body: IR,
    print: Option[PrintWriter],
  )(implicit
    E: Enclosing,
    N: sourcecode.Name,
  ): (Option[SingleCodeType], Compiled[F with Mixin]) =
    ctx.time {
      val ir =
        NormalizeNames()(
          ctx,
          Subst(
            body.noSharing(ctx),
            BindingEnv(Env.fromSeq(params.zipWithIndex.map { case ((n, t), i) => n -> In(i, t) })),
          ),
        )

      val key =
        CompileCacheKey(
          aggSigs.getOrElse(IndexedSeq.empty),
          params.map(_._2),
          ir,
        )

      ctx.CompileCache.getOrElseUpdate(
        key, {
          val lowered = ForwardLets(
            ctx,
            LoweringPipeline.compileLowerer(ctx, ir)
              .asInstanceOf[IR]
              .noSharing(ctx),
          )

          val fb =
            EmitFunctionBuilder[F](
              ctx,
              N.value,
              CodeParamType(typeInfo[Region]) +: params.map(_._2),
              CodeParamType(SingleCodeType.typeInfoFromType(lowered.typ)),
              Some("Emit.scala"),
            )

          assert(
            fb.mb.parameterTypeInfo == expectedCodeParamTypes,
            s"expected $expectedCodeParamTypes, got ${fb.mb.parameterTypeInfo}",
          )
          assert(
            fb.mb.returnTypeInfo == expectedCodeReturnType,
            s"expected $expectedCodeReturnType, got ${fb.mb.returnTypeInfo}",
          )

          val emitContext = EmitContext.analyze(ctx, lowered)
          lowered.typ match {
            case _: TStream =>
              var eltPType: PType = null

              fb.emitWithBuilder[Iterator[_]] { cb =>
                val mb = fb.apply_method
                val env = EmitEnv(
                  Env.empty,
                  (0 until params.length).map(i => mb.storeEmitParamAsField(cb, i + 2)),
                )
                val (ept, iterEmitCode) = EmitStream.produceIterator(emitContext, lowered, cb, env)
                eltPType = ept
                val noBoxIter = iterEmitCode.getOrAssert(cb).asInstanceOf[SStreamConcrete].it
                cb += noBoxIter.invoke[Region, Region, Unit](
                  "init",
                  fb.partitionRegion,
                  mb.getCodeParam[Region](1),
                )
                Code.newInstance[NoBoxLongIteratorAdapter, NoBoxLongIterator](noBoxIter)
              }

              (
                Some(PTypeReferenceSingleCodeType(eltPType.asInstanceOf[PStruct])),
                fb.resultWithIndex(print).asInstanceOf[Compiled[F with Mixin]],
              )
            case _ =>
              val rt =
                Emit(emitContext, lowered, fb, expectedCodeReturnType, params.length, aggSigs)
              (rt, fb.resultWithIndex(print))
          }
        },
      ).asInstanceOf[CompiledFunction[F with Mixin]]
    }
}
