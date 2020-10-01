package is.hail.types.physical.stypes

import is.hail.asm4s.Code
import is.hail.expr.ir.Emit.E
import is.hail.expr.ir.{AggContainer, Coalesce, EmitClassBuilder, EmitCode, EmitCodeBuilder, EmitMethodBuilder, EmitStream, Env, ExecuteContext, IEmitCode, IR, If, InsertFields, Let, LoopRef, StagedRegion}
import is.hail.types.physical.mtypes.UninitializedMValue
import is.hail.types.virtual.TStruct

class FakeEmit[C](
  val ctx: ExecuteContext,
  val kb: EmitClassBuilder[C]) {
  emitSelf =>

  def emitInPlace(
    ir: IR,
    cb: EmitCodeBuilder,
    slot: UninitializedMValue,
    doIfMissing: EmitCodeBuilder => Unit, // may be called from multiple locations, should be a few instructions at most
    region: StagedRegion,
    env: E,
    container: Option[AggContainer],
    loopEnv: Option[Env[LoopRef]]
  ): Unit = {
    def emit(ir: IR, region: StagedRegion = region, env: E = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): IEmitSCode =
      this.emit(ir, cb, region, env, container, loopEnv)

    def emitInPlace(ir: IR, cb: EmitCodeBuilder = cb, slot: UninitializedMValue = slot, doIfMissing: EmitCodeBuilder => Unit = doIfMissing,
      region: StagedRegion = region, env: E = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): Unit =
      this.emitInPlace(ir, cb, slot, doIfMissing, region, env, container, loopEnv)

    ir match {

      case If(cond, cnseq, altr) =>
        emit(cond).consume(cb, doIfMissing(cb), { predicate =>
          cb.ifx(predicate.asBoolean.boolCode, {
            emitInPlace(cnseq, cb, slot, doIfMissing, region, env, container, loopEnv)
          }, {
            emitInPlace(altr, cb, slot, doIfMissing, region, env, container, loopEnv)
          })
        })

      case Coalesce(values) =>
        values.foldRight(doIfMissing) { case (value, doIfMissing) =>
          cb => emitInPlace(value, cb, slot, doIfMissing, region, env, container, loopEnv)
        }.apply(cb)

      case Let(name, value, body) =>

        /**
          * emit value into env
          * then emitInPlace body
          */
        ???
      case InsertFields(old, fields, fieldOrder) =>

        // this is a fun one

        val slotType: MConstructableStruct = slot.typ.asInstanceOf[MConstructableStruct]
        val oldUMV: UninitializedMValue = new UninitializedMValue(slot.addr,
          MSubSetStruct(slotType, old.typ.asInstanceOf[TStruct].fieldNames))

        val wasMissing = cb.newLocal[Boolean]("wasMissing")
        cb.assign(wasMissing, false)
        emitInPlace(old, slot = oldUMV, doIfMissing = (cb: EmitCodeBuilder) => {cb.assign(wasMissing, true); doIfMissing})

        cb.ifx(!wasMissing, {
          fields.foreach { case (field, value) =>
            val fieldSlot = slotType.fieldSlot(cb, field, slot)
            emitInPlace(value, fieldSlot, doIfMissing = (cb: EmitCodeBuilder) => { slotType.setFieldMissing(cb, slot, field) })
          }
        })

      /**
        * Really what I want to do here is
        *
        *
        */


      case _ =>
        this.emit(ir, cb, region, env, container, loopEnv)
          .consume[Unit](cb, doIfMissing(cb), slot.store(cb, region.code, _))
    }
  }

  def emit(
    ir: IR,
    cb: EmitCodeBuilder,
    region: StagedRegion,
    env: E,
    container: Option[AggContainer],
    loopEnv: Option[Env[LoopRef]]
  ): IEmitSCode = {

    def emit(ir: IR, region: StagedRegion = region, env: E = env, container: Option[AggContainer] = container, loopEnv: Option[Env[LoopRef]] = loopEnv): IEmitSCode =
      this.emit(ir, cb, region, env, container, loopEnv)

    ir match {

    }
  }
}
