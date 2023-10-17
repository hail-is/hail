package is.hail.expr.ir.streams

import is.hail.annotations.{Region, RegionPool}
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.ir.agg.{AggStateSig, DictState, PhysicalAggSig, StateTuple}
import is.hail.expr.ir.functions.IntervalFunctions
import is.hail.expr.ir.orderings.StructOrdering
import is.hail.linalg.LinalgCodeUtils
import is.hail.lir
import is.hail.methods.{BitPackedVector, BitPackedVectorBuilder, LocalLDPrune, LocalWhitening}
import is.hail.types.physical.stypes.concrete.{SBinaryPointer, SStackStruct, SUnreachable}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives.{SFloat64Value, SInt32Value}
import is.hail.types.physical.stypes.{EmitType, SSettable, SValue}
import is.hail.types.physical.{PCanonicalArray, PCanonicalBinary, PCanonicalStruct, PType}
import is.hail.types.virtual._
import is.hail.types.{RIterable, TypeWithRequiredness, VirtualTypeWithReq}
import is.hail.utils._
import is.hail.variant.Locus
import org.objectweb.asm.Opcodes._
import org.sparkproject.dmg.pmml.bayesian_network.ParentValue

import java.util


abstract class StreamProducer {

  // method builder where this stream is valid
  def method: EmitMethodBuilder[_]

  /**
    * Stream length, which is present if it can be computed (somewhat) cheaply without
    * consuming the stream.
    *
    * In order for `length` to be valid, the stream must have been initialized with `initialize`.
    */
  val length: Option[EmitCodeBuilder => Code[Int]]

  /**
    * Stream producer setup method. If `initialize` is called, then the `close` method
    * must be called as well to properly handle owned resources like files.
    *
    * The stream's element region must be assigned by a consumer before initialize
    * is called.
    *
    * This block cannot jump away, e.g. to `LendOfStream`.
    *
    */
  def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit

  /**
    * Stream element region, into which the `element` is emitted. The assignment, clearing,
    * and freeing of the element region is the responsibility of the stream consumer.
    */
  val elementRegion: Settable[Region]

  /**
    * This boolean parameter indicates whether the producer's elements should be allocated in
    * separate regions (by clearing when elements leave a consumer's scope). This parameter
    * propagates bottom-up from producers like [[ReadPartition]] and [[StreamRange]], but
    * it is the responsibility of consumers to implement the right memory management semantics
    * based on this flag.
    */
  val requiresMemoryManagementPerElement: Boolean

  /**
    * The `LproduceElement` label is the mechanism by which consumers drive iteration. A consumer
    * jumps to `LproduceElement` when it is ready for an element. The code block at this label,
    * defined by the producer, jumps to either `LproduceElementDone` or `LendOfStream`, both of
    * which the consumer must define.
    */
  val LproduceElement: CodeLabel

  /**
    * The `LproduceElementDone` label is jumped to by the code block at `LproduceElement` if
    * the stream has produced a valid `element`. The immediate stream consumer must define
    * this label.
    */
  final val LproduceElementDone: CodeLabel = CodeLabel()

  /**
    * The `LendOfStream` label is jumped to by the code block at `LproduceElement` if
    * the stream has no more elements to return. The immediate stream consumer must
    * define this label.
    */
  final val LendOfStream: CodeLabel = CodeLabel()


  /**
    * Stream element. This value is valid after the producer jumps to `LproduceElementDone`,
    * until a consumer jumps to `LproduceElement` again, or calls `close()`.
    */
  val element: EmitCode

  /**
    * Stream producer cleanup method. If `initialize` is called, then the `close` method
    * must be called as well to properly handle owned resources like files.
    */
  def close(cb: EmitCodeBuilder): Unit

  final def unmanagedConsume(cb: EmitCodeBuilder, outerRegion: Value[Region], setup: EmitCodeBuilder => Unit = _ => ())(perElement: EmitCodeBuilder => Unit): Unit = {

    this.initialize(cb, outerRegion)
    setup(cb)
    cb.goto(this.LproduceElement)
    cb.define(this.LproduceElementDone)
    perElement(cb)
    cb.goto(this.LproduceElement)

    cb.define(this.LendOfStream)
    this.close(cb)
  }

  // only valid if `perElement` does not retain pointers into the element region after it returns (or adds region references)
  final def memoryManagedConsume(outerRegion: Value[Region], cb: EmitCodeBuilder, setup: EmitCodeBuilder => Unit = _ => ())(perElement: EmitCodeBuilder => Unit): Unit = {
    if (requiresMemoryManagementPerElement) {
      cb.assign(elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))

      unmanagedConsume(cb, outerRegion, setup) { cb =>
        perElement(cb)
        cb += elementRegion.clearRegion()
      }
      cb += elementRegion.invalidate()
    } else {
      cb.assign(elementRegion, outerRegion)
      unmanagedConsume(cb, outerRegion, setup)(perElement)
    }
  }
}

object EmitStream {
  private[ir] def produce(
    emitter: Emit[_],
    streamIR: IR,
    cb: EmitCodeBuilder,
    mb: EmitMethodBuilder[_],
    outerRegion: Value[Region],
    env: EmitEnv,
    container: Option[AggContainer]
  ): IEmitCode = {

    def emitVoid(ir: IR, cb: EmitCodeBuilder, region: Value[Region] = outerRegion, env: EmitEnv = env, container: Option[AggContainer] = container): Unit =
      emitter.emitVoid(cb, ir, region, env, container, None)

    def emit(ir: IR, cb: EmitCodeBuilder, region: Value[Region] = outerRegion, env: EmitEnv = env, container: Option[AggContainer] = container): IEmitCode = {
      ir.typ match {
        case _: TStream => produce(ir, cb, cb.emb, region, env, container)
        case _ => emitter.emitI(ir, cb, region, env, container, None)
      }
    }

    // returns IEmitCode of SStreamConcrete
    def produceIterator(streamIR: IR, elementPType: PType, cb: EmitCodeBuilder, outerRegion: Value[Region] = outerRegion, env: EmitEnv = env): IEmitCode = {
      val ecb = cb.emb.genEmitClass[NoBoxLongIterator]("stream_to_iter")
      ecb.cb.addInterface(typeInfo[MissingnessAsMethod].iname)

      val fv = FreeVariables(streamIR, false, false).eval
      val (envParamTypes, envParams, restoreEnv) = env.asParams(fv)

      val isMissing = ecb.genFieldThisRef[Boolean]("isMissing")
      val eosField = ecb.genFieldThisRef[Boolean]("eos")
      val outerRegionField = ecb.genFieldThisRef[Region]("outer")

      ecb.makeAddPartitionRegion()

      var producer: StreamProducer = null
      var producerRequired: Boolean = false

      val next = ecb.newEmitMethod("next", FastSeq[ParamType](), LongInfo)
      val ctor = ecb.newEmitMethod("<init>", FastSeq[ParamType](typeInfo[Region], arrayInfo[Long]) ++ envParamTypes, UnitInfo)
      ctor.voidWithBuilder { cb =>
        val L = new lir.Block()
        L.append(
          lir.methodStmt(INVOKESPECIAL,
            "java/lang/Object",
            "<init>",
            "()V",
            false,
            UnitInfo,
            FastSeq(lir.load(ctor.mb._this.asInstanceOf[LocalRef[_]].l))))
        cb += new VCode(L, L, null)

        val newEnv = restoreEnv(cb, 3)
        val s = EmitStream.produce(new Emit(emitter.ctx, ecb), streamIR, cb, next, outerRegionField, newEnv, None)
        producerRequired = s.required
        s.consume(cb, {
          if (!producerRequired) cb.assign(isMissing, true)
        }, { stream =>
          if (!producerRequired) cb.assign(isMissing, false)
          producer = stream.asStream.getProducer(next)
        })

        val self = cb.memoize(Code.checkcast[FunctionWithPartitionRegion]((ctor.getCodeParam(0)(ecb.cb.ti))))

        ecb.setLiteralsArray(cb, ctor.getCodeParam[Array[Long]](2))
        val partitionRegion = cb.memoize(ctor.getCodeParam[Region](1))
        cb += self.invoke[Region, Unit]("addPartitionRegion", partitionRegion)
        cb += self.invoke[RegionPool, Unit]("setPool", partitionRegion.getPool())
      }

      next.emitWithBuilder { cb =>
        val ret = cb.newLocal[Long]("ret")
        val Lret = CodeLabel()
        cb.goto(producer.LproduceElement)
        cb.define(producer.LproduceElementDone)
        producer.element.toI(cb)
          .consume(cb, {
            cb.assign(ret, 0L)
          }, { value =>
            cb.assign(ret, elementPType.store(cb, producer.elementRegion, value, false))
          })
        cb.goto(Lret)
        cb.define(producer.LendOfStream)
        cb.assign(eosField, true)

        cb.define(Lret)
        ret
      }

      val init = ecb.newEmitMethod("init", FastSeq[ParamType](typeInfo[Region], typeInfo[Region]), UnitInfo)
      init.voidWithBuilder { cb =>
        val outerRegion = init.getCodeParam[Region](1)
        val eltRegion = init.getCodeParam[Region](2)

        cb.assign(producer.elementRegion, eltRegion)
        cb.assign(outerRegionField, outerRegion)
        producer.initialize(cb, outerRegion)
        cb.assign(eosField, false)
      }


      val isEOS = ecb.newEmitMethod("eos", FastSeq[ParamType](), BooleanInfo)
      isEOS.emitWithBuilder[Boolean](cb => eosField)

      val isMissingMethod = ecb.newEmitMethod("isMissing", FastSeq[ParamType](), BooleanInfo)
      isMissingMethod.emitWithBuilder[Boolean](cb => isMissing)


      val close = ecb.newEmitMethod("close", FastSeq[ParamType](), UnitInfo)
      close.voidWithBuilder(cb => producer.close(cb))

      val obj = cb.memoize(Code.newInstance(ecb.cb, ctor.mb,
        FastSeq(cb.emb.partitionRegion.get, cb.emb.ecb.literalsArray().get) ++ envParams.map(_.get)))

      val iter = cb.emb.genFieldThisRef[NoBoxLongIterator]("iter")
      cb.assign(iter, Code.checkcast[NoBoxLongIterator](obj))
      IEmitCode(cb,
        if (producerRequired) false else Code.checkcast[MissingnessAsMethod](obj).invoke[Boolean]("isMissing"),
        new SStreamConcrete(SStreamIteratorLong(producer.element.required, elementPType, producer.requiresMemoryManagementPerElement),
          iter))
    }

    def produce(streamIR: IR, cb: EmitCodeBuilder, mb: EmitMethodBuilder[_] = mb, region: Value[Region] = outerRegion, env: EmitEnv = env, container: Option[AggContainer] = container): IEmitCode =
      EmitStream.produce(emitter, streamIR, cb, mb, region, env, container)

    def typeWithReqx(node: IR): VirtualTypeWithReq = VirtualTypeWithReq(node.typ, emitter.ctx.req.lookup(node).asInstanceOf[TypeWithRequiredness])
    def typeWithReq: VirtualTypeWithReq = typeWithReqx(streamIR)

    streamIR match {

      case x@NA(_typ: TStream) =>
        val st = SStream(EmitType(SUnreachable.fromVirtualType(_typ.elementType), true))
        val region = mb.genFieldThisRef[Region]("na_region")
        val producer = new StreamProducer {
          override def method: EmitMethodBuilder[_] = mb
          override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {}

          override val length: Option[EmitCodeBuilder => Code[Int]] = Some(_ => Code._fatal[Int]("tried to get NA stream length"))
          override val elementRegion: Settable[Region] = region
          override val requiresMemoryManagementPerElement: Boolean = false
          override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
            cb.goto(LendOfStream)
          }
          override val element: EmitCode = EmitCode.present(mb, st.elementType.defaultValue)

          override def close(cb: EmitCodeBuilder): Unit = {}
        }
        IEmitCode.missing(cb, SStreamValue(producer))

      case Ref(name, _typ) =>
        assert(_typ.isInstanceOf[TStream])
        env.bindings.lookup(name).toI(cb)
          .map(cb) { case stream: SStreamValue =>
            val childProducer = stream.getProducer(mb)
            val producer = new StreamProducer {
              override def method: EmitMethodBuilder[_] = mb
              override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = childProducer.initialize(cb, outerRegion)

              override val length: Option[EmitCodeBuilder => Code[Int]] = childProducer.length
              override val elementRegion: Settable[Region] = childProducer.elementRegion
              override val requiresMemoryManagementPerElement: Boolean = childProducer.requiresMemoryManagementPerElement
              override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
                cb.goto(childProducer.LproduceElement)
                cb.define(childProducer.LproduceElementDone)
                cb.goto(LproduceElementDone)
              }
              override val element: EmitCode = childProducer.element

              override def close(cb: EmitCodeBuilder): Unit = childProducer.close(cb)
            }
            mb.implementLabel(childProducer.LendOfStream) { cb =>
              cb.goto(producer.LendOfStream)
            }
            SStreamValue(producer)
          }

      case Let(name, value, body) =>
        cb.withScopedMaybeStreamValue(EmitCode.fromI(cb.emb)(cb => emit(value, cb)), s"let_$name") { ev =>
          produce(body, cb, env = env.bind(name, ev))
        }

      case In(n, _) =>
        // this, Code[Region], ...
        val param = env.inputValues(n).toI(cb)
        if (!param.st.isInstanceOf[SStream])
          throw new RuntimeException(s"parameter ${ 2 + n } is not a stream! t=${ param.st } }, params=${ mb.emitParamTypes }")
        param

      case ToStream(a, _requiresMemoryManagementPerElement) =>

        emit(a, cb).map(cb) { case _ind: SIndexableValue =>
          val containerField = cb.memoizeField(_ind, "indexable").asIndexable
          val container = containerField.asInstanceOf[SIndexableValue]
          val idx = mb.genFieldThisRef[Int]("tostream_idx")
          val regionVar = mb.genFieldThisRef[Region]("tostream_region")


          SStreamValue(
            new StreamProducer {
              override def method: EmitMethodBuilder[_] = mb
              override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
                cb.assign(idx, -1)
              }

              override val length: Option[EmitCodeBuilder => Code[Int]] = Some(_ => container.loadLength())

              override val elementRegion: Settable[Region] = regionVar

              override val requiresMemoryManagementPerElement: Boolean = _requiresMemoryManagementPerElement

              override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
                cb.assign(idx, idx + 1)
                cb.ifx(idx >= container.loadLength(), cb.goto(LendOfStream))
                cb.goto(LproduceElementDone)
              }

              val element: EmitCode = EmitCode.fromI(mb) { cb =>
                container.loadElement(cb, idx) }

              def close(cb: EmitCodeBuilder): Unit = {}
            })

        }

      case x@StreamBufferedAggregate(streamChild, initAggs, newKey, seqOps, name,
        aggSignatures: IndexedSeq[PhysicalAggSig], bufferSize: Int) =>
        val region = mb.genFieldThisRef[Region]("stream_buff_agg_region")
        produce(streamChild, cb)
          .map(cb) { case childStream: SStreamValue =>
            val childProducer = childStream.getProducer(mb)
            val eltField = mb.newEmitField("stream_buff_agg_elt", childProducer.element.emitType)
            val newKeyVType = typeWithReqx(newKey)
            val kb = mb.ecb
            val nestedStates = aggSignatures.toArray.map(sig => AggStateSig.getState(sig.state, kb))
            val nested = StateTuple(nestedStates)
            val dictState = new DictState(kb, newKeyVType, nested)
            val maxSize = mb.genFieldThisRef[Int]("stream_buff_agg_max_size")
            val nodeArray = mb.genFieldThisRef[Array[Long]]("stream_buff_agg_element_array")
            val idx = mb.genFieldThisRef[Int]("stream_buff_agg_idx")
            val returnStreamType= x.typ.asInstanceOf[TStream]
            val returnElemType = returnStreamType.elementType
            val tupleFieldTypes = aggSignatures.map(_ => TBinary)
            val tupleFields = (0 to tupleFieldTypes.length).zip(tupleFieldTypes).map { case (fieldIdx, fieldType) => TupleField(fieldIdx, fieldType) }
            val serializedAggSType = SStackStruct(TTuple(tupleFields), tupleFieldTypes.map(_ => EmitType(SBinaryPointer(PCanonicalBinary()), true)).toIndexedSeq)
            val keyAndAggFields = newKeyVType.canonicalPType.asInstanceOf[PCanonicalStruct].sType.fieldEmitTypes :+ EmitType(serializedAggSType, true)
            val returnElemSType = SStackStruct(returnElemType.asInstanceOf[TBaseStruct], keyAndAggFields)
            val newStreamElem = mb.newEmitField("stream_buff_agg_new_stream_elem", EmitType(returnElemSType, true))
            val numElemInArray = mb.genFieldThisRef[Int]("stream_buff_agg_num_elem_in_size")
            val childStreamEnded = mb.genFieldThisRef[Boolean]("stream_buff_agg_child_stream_ended")
            val produceElementMode = mb.genFieldThisRef[Boolean]("stream_buff_agg_child_produce_elt_mode")
            val producer: StreamProducer = new StreamProducer {
              override def method: EmitMethodBuilder[_] = mb
              override val length: Option[EmitCodeBuilder => Code[Int]] = None

              override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
                if (childProducer.requiresMemoryManagementPerElement)
                  cb.assign(childProducer.elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
                else
                  cb.assign(childProducer.elementRegion, region)

                childProducer.initialize(cb, outerRegion)
                cb.assign(childStreamEnded, false)
                cb.assign(produceElementMode, false)
                cb.assign(idx, 0)
                cb.assign(maxSize, bufferSize)
                cb.assign(nodeArray, Code.newArray[Long](maxSize))
                cb.assign(numElemInArray, 0)
                dictState.createState(cb)
              }

              override val elementRegion: Settable[Region] = region

              override val requiresMemoryManagementPerElement: Boolean = childProducer.requiresMemoryManagementPerElement

              override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
                val elementProduceLabel = CodeLabel()
                val getElemLabel = CodeLabel()
                val startLabel = CodeLabel()

                cb.define(startLabel)
                cb.ifx(produceElementMode, {
                  cb.goto(elementProduceLabel)
                })

                // Garbage collects old aggregator state if moving onto new group
                dictState.newState(cb)
                val initContainer = AggContainer(aggSignatures.toArray.map(sig => sig.state), dictState.initContainer, cleanup = () => ())
                dictState.init(cb, { cb => emitVoid(initAggs, cb, container = Some(initContainer)) })

                cb.define(getElemLabel)
                if (childProducer.requiresMemoryManagementPerElement)
                  cb += childProducer.elementRegion.clearRegion()
                cb.goto(childProducer.LproduceElement)

                cb.define(childProducer.LproduceElementDone)
                cb.assign(eltField, childProducer.element)
                val newKeyResultCode = EmitCode.fromI(mb) { cb =>
                  emit(newKey,
                    cb = cb,
                    env = env.bind(name, eltField),
                    region = region)
                }
                val resultKeyValue = newKeyResultCode.memoize(cb, "buff_agg_stream_result_key")
                val keyedContainer = AggContainer(aggSignatures.toArray.map(sig => sig.state), dictState.keyed.container, cleanup = () => ())
                dictState.withContainer(cb, resultKeyValue, { cb =>
                  emitVoid(seqOps, cb, container = Some(keyedContainer), env = env.bind(name, eltField))
                })
                cb.ifx(dictState.size >= maxSize,{
                  cb.assign(produceElementMode, true)
                })

                cb.ifx(produceElementMode, {
                  cb.goto(elementProduceLabel)},
                  {
                  cb.goto(getElemLabel)
                  }
                )

                cb.define(childProducer.LendOfStream)
                cb.assign(childStreamEnded, true)
                cb.assign(produceElementMode, true)

                cb.define(elementProduceLabel)
                cb.ifx(numElemInArray ceq 0, {
                  dictState.tree.foreach(cb) { (cb, elementOff) =>
                    cb += nodeArray.update(numElemInArray, elementOff)
                    cb.assign(numElemInArray, numElemInArray + 1)
                  }
                })

                cb.ifx(numElemInArray <= idx, {
                  cb.assign(idx, 0)
                  cb.assign(numElemInArray, 0)
                  cb.assign(produceElementMode, false)
                  cb.ifx(childStreamEnded , {
                    cb.goto(LendOfStream)
                  }, {
                    cb.goto(startLabel)
                  })
                })
                val nodeAddress = cb.memoize(nodeArray(idx))
                cb.assign(idx, idx + 1)
                dictState.loadNode(cb, nodeAddress)

                val keyInWrongRegion = dictState.keyed.storageType.loadCheapSCode(cb, nodeAddress)
                val addrOfKeyInRightRegion = dictState.keyed.storageType.store(cb, region, keyInWrongRegion, true)
                val key = dictState.keyed.storageType.loadCheapSCode(cb, addrOfKeyInRightRegion).loadField(cb, "kt").memoize(cb, "stream_buff_agg_key_right_region")

                val serializedAggValue = keyedContainer.container.states.states.map(state => state.serializeToRegion(cb, PCanonicalBinary(), region))
                val serializedAggEmitCodes = serializedAggValue.map(aggValue => EmitCode.present(mb, aggValue))
                val serializedAggTupleSValue = SStackStruct.constructFromArgs(cb, region, serializedAggSType.virtualType, serializedAggEmitCodes: _*)
                val keyValue = key.get(cb).asInstanceOf[SBaseStructValue]
                val sStructToReturn = keyValue.insert(cb, region, returnElemType.asInstanceOf[TStruct], ("agg", EmitCode.present(mb, serializedAggTupleSValue)
                  .memoize(cb, "stream_buff_agg_return_val")))
                assert(returnElemSType.virtualType == sStructToReturn.st.virtualType)
                val casted = sStructToReturn.castTo(cb, region, returnElemSType)
                cb.assign(newStreamElem, EmitCode.present(mb, casted).toI(cb))
                cb.goto(LproduceElementDone)
              }

              override val element: EmitCode = newStreamElem.load

              override def close(cb: EmitCodeBuilder): Unit = {
                childProducer.close(cb)
                if (childProducer.requiresMemoryManagementPerElement)
                  cb += childProducer.elementRegion.invalidate()
                cb += dictState.region.invalidate()
              }
            }
            SStreamValue(producer)
          }

      case x@MakeStream(args, _, _requiresMemoryManagementPerElement) =>
        val region = mb.genFieldThisRef[Region]("makestream_region")

        // FIXME use SType.chooseCompatibleType
        val unifiedType = typeWithReq.canonicalEmitType.st.asInstanceOf[SStream].elementEmitType
        val eltField = mb.newEmitField("makestream_elt", unifiedType)

        val staticLen = args.size
        val current = mb.genFieldThisRef[Int]("makestream_current")

        IEmitCode.present(cb, SStreamValue(
          new StreamProducer {
            override def method: EmitMethodBuilder[_] = mb
            override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
              cb.assign(current, 0) // switches on 1..N
            }

            override val length: Option[EmitCodeBuilder => Code[Int]] = Some(_ => staticLen)

            override val elementRegion: Settable[Region] = region

            override val requiresMemoryManagementPerElement: Boolean = _requiresMemoryManagementPerElement

            override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
              val LendOfSwitch = CodeLabel()
              cb += Code.switch(current,
                EmitCodeBuilder.scopedVoid(mb) { cb =>
                  cb.goto(LendOfStream)
                },
                args.toFastSeq.map(a => EmitCode.fromI(mb)(cb => emit(a, cb, region))).map { elem =>
                  EmitCodeBuilder.scopedVoid(mb) { cb =>
                    cb.assign(eltField, elem.toI(cb).map(cb)(pc => pc.castTo(cb, region, unifiedType.st, false)))
                    cb.goto(LendOfSwitch)
                  }
                })
              cb.define(LendOfSwitch)
              cb.assign(current, current + 1)

              cb.goto(LproduceElementDone)
            }

            val element: EmitCode = eltField.load

            def close(cb: EmitCodeBuilder): Unit = {}
          }))

      case x@If(cond, cnsq, altr) =>
        emit(cond, cb).flatMap(cb) { cond =>
          val xCond = mb.genFieldThisRef[Boolean]("stream_if_cond")
          cb.assign(xCond, cond.asBoolean.value)

          val Lmissing = CodeLabel()
          val Lpresent = CodeLabel()

          val leftEC = EmitCode.fromI(cb.emb)(cb => produce(cnsq, cb))
          val rightEC = EmitCode.fromI(cb.emb)(cb => produce(altr, cb))

          val leftProducer = leftEC.pv.asStream.getProducer(mb)
          val rightProducer = rightEC.pv.asStream.getProducer(mb)

          val unifiedStreamSType = typeWithReq.canonicalEmitType.st.asInstanceOf[SStream]
          val unifiedElementType = unifiedStreamSType.elementEmitType

          val xElt = mb.newEmitField(unifiedElementType)

          val region = mb.genFieldThisRef[Region]("streamif_region")
          cb.ifx(xCond,
            leftEC.toI(cb).consume(cb, cb.goto(Lmissing), _ => cb.goto(Lpresent)),
            rightEC.toI(cb).consume(cb, cb.goto(Lmissing), _ => cb.goto(Lpresent)))

          val producer = new StreamProducer {
            override def method: EmitMethodBuilder[_] = mb
            override val length: Option[EmitCodeBuilder => Code[Int]] = leftProducer.length
              .liftedZip(rightProducer.length).map { case (computeL1, computeL2) =>
              cb: EmitCodeBuilder => {
                val len = cb.newLocal[Int]("if_len")
                cb.ifx(xCond, cb.assign(len, computeL1(cb)), cb.assign(len, computeL2(cb)))
                len.get
              }
            }

            override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
              cb.ifx(xCond, {
                cb.assign(leftProducer.elementRegion, region)
                leftProducer.initialize(cb, outerRegion)
              }, {
                cb.assign(rightProducer.elementRegion, region)
                rightProducer.initialize(cb, outerRegion)
              })
            }

            override val elementRegion: Settable[Region] = region
            override val requiresMemoryManagementPerElement: Boolean = leftProducer.requiresMemoryManagementPerElement || rightProducer.requiresMemoryManagementPerElement
            override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
              cb.ifx(xCond, cb.goto(leftProducer.LproduceElement), cb.goto(rightProducer.LproduceElement))

              cb.define(leftProducer.LproduceElementDone)
              cb.assign(xElt, leftProducer.element.toI(cb).map(cb)(_.castTo(cb, region, xElt.st)))
              cb.goto(LproduceElementDone)

              cb.define(rightProducer.LproduceElementDone)
              cb.assign(xElt, rightProducer.element.toI(cb).map(cb)(_.castTo(cb, region, xElt.st)))
              cb.goto(LproduceElementDone)

              cb.define(leftProducer.LendOfStream)
              cb.goto(LendOfStream)

              cb.define(rightProducer.LendOfStream)
              cb.goto(LendOfStream)
            }

            override val element: EmitCode = xElt.load

            override def close(cb: EmitCodeBuilder): Unit = {
              cb.ifx(xCond, leftProducer.close(cb), rightProducer.close(cb))
            }
          }

          IEmitCode(Lmissing, Lpresent,
            SStreamValue(producer),
            leftEC.required && rightEC.required)
        }

      case StreamIota(start, step, _requiresMemoryManagementPerElement) =>
        emit(start, cb).flatMap(cb) { startCode =>
          emit(step, cb).map(cb) { stepCode =>
            val curr = mb.genFieldThisRef[Int]("streamrange_curr")
            val stepVar = mb.genFieldThisRef[Int]("streamrange_stop")
            val regionVar = mb.genFieldThisRef[Region]("sr_region")
            val producer: StreamProducer = new StreamProducer {
              override def method: EmitMethodBuilder[_] = mb
              override val length: Option[EmitCodeBuilder => Code[Int]] = None

              override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
                val startVar = startCode.asInt.value
                cb.assign(stepVar, stepCode.asInt.value)
                cb.assign(curr, startVar - stepVar)
              }

              override val elementRegion: Settable[Region] = regionVar

              override val requiresMemoryManagementPerElement: Boolean = _requiresMemoryManagementPerElement

              override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
                cb.assign(curr, curr + stepVar)
                cb.goto(LproduceElementDone)
              }

              val element: EmitCode = EmitCode.present(mb, new SInt32Value(curr))

              def close(cb: EmitCodeBuilder): Unit = {}
            }
            SStreamValue(producer)

          }
        }
      case StreamRange(start, stop, I32(step), _requiresMemoryManagementPerElement, errorID) if (step != 0) =>
        emit(start, cb).flatMap(cb) { startCode =>
          emit(stop, cb).map(cb) { stopCode =>
            val curr = mb.genFieldThisRef[Int]("streamrange_curr")
            val startVar = mb.genFieldThisRef[Int]("range_start")
            val stopVar = mb.genFieldThisRef[Int]("streamrange_stop")
            val regionVar = mb.genFieldThisRef[Region]("sr_region")

            cb.assign(startVar, startCode.asInt.value)
            cb.assign(stopVar, stopCode.asInt.value)

            val producer: StreamProducer = new StreamProducer {
              override def method: EmitMethodBuilder[_] = mb
              override val length: Option[EmitCodeBuilder => Code[Int]] = Some({ cb =>
                val len = cb.newLocal[Int]("streamrange_len")
                if (step > 0)
                  cb.ifx(startVar >= stopVar,
                    cb.assign(len, 0),
                    cb.assign(len, ((stopVar.toL - startVar.toL - 1L) / step.toLong + 1L).toI))
                else
                  cb.ifx(startVar <= stopVar,
                    cb.assign(len, 0),
                    cb.assign(len, ((startVar.toL - stopVar.toL - 1L) / (-step.toLong) + 1L).toI))
                len
              })

              override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
                start match {
                  case I32(x) if step < 0 && ((x.toLong - Int.MinValue.toLong) / step.toLong + 1) < Int.MaxValue =>
                  case I32(x) if step > 0 && ((Int.MaxValue.toLong - x.toLong) / step.toLong + 1) < Int.MaxValue =>
                  case _ =>
                    cb.ifx((stopVar.toL - startVar.toL) / step.toLong > const(Int.MaxValue.toLong),
                      cb._fatalWithError(errorID, "Array range cannot have more than MAXINT elements."))
                }
                cb.assign(curr, startVar - step)
              }

              override val elementRegion: Settable[Region] = regionVar

              override val requiresMemoryManagementPerElement: Boolean = _requiresMemoryManagementPerElement

              override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
                cb.assign(curr, curr + step)
                if (step > 0)
                  cb.ifx(curr >= stopVar, cb.goto(LendOfStream))
                else
                  cb.ifx(curr <= stopVar, cb.goto(LendOfStream))
                cb.goto(LproduceElementDone)
              }

              val element: EmitCode = EmitCode.present(mb, new SInt32Value(curr))

              def close(cb: EmitCodeBuilder): Unit = {}
            }

            SStreamValue(producer)
          }
        }

      case StreamRange(startIR, stopIR, stepIR, _requiresMemoryManagementPerElement, errorID) =>

        emit(startIR, cb).flatMap(cb) { startc =>
          emit(stopIR, cb).flatMap(cb) { stopc =>
            emit(stepIR, cb).map(cb) { stepc =>
              val len = mb.genFieldThisRef[Int]("sr_len")
              val regionVar = mb.genFieldThisRef[Region]("sr_region")

              val start = cb.newField[Int]("sr_step")
              val stop = cb.newField[Int]("sr_stop")
              val step = cb.newField[Int]("sr_step")

              val curr = mb.genFieldThisRef[Int]("streamrange_curr")
              val idx = mb.genFieldThisRef[Int]("streamrange_idx")

              cb.assign(start, startc.asInt.value)
              cb.assign(stop, stopc.asInt.value)
              cb.assign(step, stepc.asInt.value)

              val producer: StreamProducer = new StreamProducer {
                override def method: EmitMethodBuilder[_] = mb
                override val length: Option[EmitCodeBuilder => Code[Int]] = Some(_ => len)

                override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
                  val llen = cb.newLocal[Long]("streamrange_llen")

                  cb.ifx(step ceq const(0), cb._fatalWithError(errorID, "Array range cannot have step size 0."))
                  cb.ifx(step < const(0), {
                    cb.ifx(start.toL <= stop.toL, {
                      cb.assign(llen, 0L)
                    }, {
                      cb.assign(llen, (start.toL - stop.toL - 1L) / (-step.toL) + 1L)
                    })
                  }, {
                    cb.ifx(start.toL >= stop.toL, {
                      cb.assign(llen, 0L)
                    }, {
                      cb.assign(llen, (stop.toL - start.toL - 1L) / step.toL + 1L)
                    })
                  })
                  cb.ifx(llen > const(Int.MaxValue.toLong), {
                    cb._fatalWithError(errorID, "Array range cannot have more than MAXINT elements.")
                  })
                  cb.assign(len, llen.toI)

                  cb.assign(curr, start - step)
                  cb.assign(idx, 0)
                }

                override val elementRegion: Settable[Region] = regionVar

                override val requiresMemoryManagementPerElement: Boolean = _requiresMemoryManagementPerElement

                override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
                  cb.ifx(idx >= len, cb.goto(LendOfStream))
                  cb.assign(curr, curr + step)
                  cb.assign(idx, idx + 1)
                  cb.goto(LproduceElementDone)
                }

                val element: EmitCode = EmitCode.present(mb, new SInt32Value(curr))

                def close(cb: EmitCodeBuilder): Unit = {}
              }

              SStreamValue(producer)
            }
          }
        }


      case SeqSample(totalSize, numToSample, rngState, _requiresMemoryManagementPerElement) =>
        // Implemented based on http://www.ittc.ku.edu/~jsv/Papers/Vit84.sampling.pdf Algorithm A
        emit(totalSize, cb).flatMap(cb) { case totalSizeVal: SInt32Value =>
          emit(numToSample, cb).map(cb) { case numToSampleVal: SInt32Value =>
            val len = mb.genFieldThisRef[Int]("seq_sample_len")
            val regionVar = mb.genFieldThisRef[Region]("seq_sample_region")

            val nRemaining = cb.newField[Int]("seq_sample_num_remaining", numToSampleVal.value)
            val candidate = cb.newField[Int]("seq_sample_candidate", 0)
            val elementToReturn = cb.newField[Int]("seq_sample_element_to_return", -1) // -1 should never be returned.

            val producer = new StreamProducer {
              override def method: EmitMethodBuilder[_] = mb
              override val length: Option[EmitCodeBuilder => Code[Int]] = Some(_ => len)

              override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
                cb.assign(len, numToSampleVal.asInt.value)
                cb.assign(nRemaining, numToSampleVal.value)
                cb.assign(candidate, 0)
                cb.assign(elementToReturn, -1)
              }

              override val elementRegion: Settable[Region] = regionVar

              override val requiresMemoryManagementPerElement: Boolean = _requiresMemoryManagementPerElement

              override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
                cb.ifx(nRemaining <= 0, cb.goto(LendOfStream))

                val u = cb.newLocal[Double]("seq_sample_rand_unif", Code.invokeStatic0[Math, Double]("random"))
                val fC = cb.newLocal[Double]("seq_sample_Fc", (totalSizeVal.value - candidate - nRemaining).toD / (totalSizeVal.value - candidate).toD)

                cb.whileLoop(fC > u, {
                  cb.assign(candidate, candidate + 1)
                  cb.assign(fC, fC * (const(1.0) - (nRemaining.toD / (totalSizeVal.value - candidate).toD)))
                })
                cb.assign(nRemaining, nRemaining - 1)
                cb.assign(elementToReturn, candidate)
                cb.assign(candidate, candidate + 1)
                cb.goto(LproduceElementDone)
              }

              override val element: EmitCode = EmitCode.present(mb, new SInt32Value(elementToReturn))

              override def close(cb: EmitCodeBuilder): Unit = {}
            }
            SStreamValue(producer)
          }
        }

      case StreamFilter(a, name, cond) =>
        produce(a, cb)
          .map(cb) { case childStream: SStreamValue =>
            val childProducer = childStream.getProducer(mb)

            val filterEltRegion = mb.genFieldThisRef[Region]("streamfilter_filter_region")

            val elementField = cb.emb.newEmitField("streamfilter_cond", childProducer.element.emitType)

            val producer = new StreamProducer {
              override def method: EmitMethodBuilder[_] = mb
              override val length: Option[EmitCodeBuilder => Code[Int]] = None

              override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
                if (childProducer.requiresMemoryManagementPerElement)
                  cb.assign(childProducer.elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
                else
                  cb.assign(childProducer.elementRegion, outerRegion)
                childProducer.initialize(cb, outerRegion)
              }

              override val elementRegion: Settable[Region] = filterEltRegion

              override val requiresMemoryManagementPerElement: Boolean = childProducer.requiresMemoryManagementPerElement

              override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
                val Lfiltered = CodeLabel()

                cb.goto(childProducer.LproduceElement)

                cb.define(childProducer.LproduceElementDone)
                cb.assign(elementField, childProducer.element)
                // false and NA both fail the filter
                emit(cond, cb = cb, env = env.bind(name, elementField), region = childProducer.elementRegion)
                  .consume(cb,
                    cb.goto(Lfiltered),
                    { sc =>
                      cb.ifx(!sc.asBoolean.value, cb.goto(Lfiltered))
                    })

                if (requiresMemoryManagementPerElement)
                  cb += filterEltRegion.takeOwnershipOfAndClear(childProducer.elementRegion)
                cb.goto(LproduceElementDone)

                cb.define(Lfiltered)
                if (requiresMemoryManagementPerElement)
                  cb += childProducer.elementRegion.clearRegion()
                cb.goto(childProducer.LproduceElement)
              }

              val element: EmitCode = elementField

              def close(cb: EmitCodeBuilder): Unit = {
                childProducer.close(cb)
                if (requiresMemoryManagementPerElement)
                  cb += childProducer.elementRegion.invalidate()
              }
            }
            mb.implementLabel(childProducer.LendOfStream) { cb =>
              cb.goto(producer.LendOfStream)
            }

            SStreamValue(producer)
          }

      case StreamTake(a, num) =>
        produce(a, cb)
          .flatMap(cb) { case childStream: SStreamValue =>
            emit(num, cb).map(cb) { case num: SInt32Value =>
              val childProducer = childStream.getProducer(mb)
              val n = mb.genFieldThisRef[Int]("stream_take_n")
              val idx = mb.genFieldThisRef[Int]("stream_take_idx")

              val producer = new StreamProducer {
                override def method: EmitMethodBuilder[_] = mb
                override val length: Option[EmitCodeBuilder => Code[Int]] =
                  childProducer.length.map(compLen => (cb: EmitCodeBuilder) => compLen(cb).min(n))

                override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
                  cb.assign(n, num.value)
                  cb.ifx(n < 0, cb._fatal(s"stream take: negative number of elements to take: ", n.toS))
                  cb.assign(idx, 0)
                  childProducer.initialize(cb, outerRegion)
                }

                override val elementRegion: Settable[Region] = childProducer.elementRegion
                override val requiresMemoryManagementPerElement: Boolean = childProducer.requiresMemoryManagementPerElement
                override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
                  cb.ifx(idx >= n, cb.goto(LendOfStream))
                  cb.assign(idx, idx + 1)
                  cb.goto(childProducer.LproduceElement)

                  cb.define(childProducer.LproduceElementDone)
                  cb.goto(LproduceElementDone)

                  cb.define(childProducer.LendOfStream)
                  cb.goto(LendOfStream)
                }
                override val element: EmitCode = childProducer.element

                override def close(cb: EmitCodeBuilder): Unit = {
                  childProducer.close(cb)
                }
              }

              SStreamValue(producer)
            }
          }

      case StreamDrop(a, num) =>
        produce(a, cb)
          .flatMap(cb) { case (childStream: SStreamValue) =>
            emit(num, cb).map(cb) { case num: SInt32Value =>
              val childProducer = childStream.getProducer(mb)
              val n = mb.genFieldThisRef[Int]("stream_drop_n")
              val idx = mb.genFieldThisRef[Int]("stream_drop_idx")

              val producer = new StreamProducer {
                override def method: EmitMethodBuilder[_] = mb
                override val length: Option[EmitCodeBuilder => Code[Int]] = childProducer.length.map { computeL =>
                  (cb: EmitCodeBuilder) => (computeL(cb) - n).max(0)
                }

                override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
                  cb.assign(n, num.value)
                  cb.ifx(n < 0, cb._fatal(s"stream drop: negative number of elements to drop: ", n.toS))
                  cb.assign(idx, 0)
                  childProducer.initialize(cb, outerRegion)
                }

                override val elementRegion: Settable[Region] = childProducer.elementRegion
                override val requiresMemoryManagementPerElement: Boolean = childProducer.requiresMemoryManagementPerElement
                override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
                  cb.goto(childProducer.LproduceElement)
                  cb.define(childProducer.LproduceElementDone)
                  cb.assign(idx, idx + 1)
                  cb.ifx(idx <= n, {
                    if (childProducer.requiresMemoryManagementPerElement)
                      cb += childProducer.elementRegion.clearRegion()
                    cb.goto(childProducer.LproduceElement)
                  })
                  cb.goto(LproduceElementDone)

                  cb.define(childProducer.LendOfStream)
                  cb.goto(LendOfStream)
                }
                override val element: EmitCode = childProducer.element

                override def close(cb: EmitCodeBuilder): Unit = {
                  childProducer.close(cb)
                }
              }

              SStreamValue(producer)
            }
          }

      case StreamTakeWhile(a, elt, condIR) =>
        produce(a, cb)
          .map(cb) { case childStream: SStreamValue =>
            val childProducer = childStream.getProducer(mb)

            val eltSettable = mb.newEmitField("stream_take_while_elt", childProducer.element.emitType)

            val producer = new StreamProducer {
              override def method: EmitMethodBuilder[_] = mb
              override val length: Option[EmitCodeBuilder => Code[Int]] = None

              override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
                childProducer.initialize(cb, outerRegion)
              }

              override val elementRegion: Settable[Region] = childProducer.elementRegion
              override val requiresMemoryManagementPerElement: Boolean = childProducer.requiresMemoryManagementPerElement
              override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
                cb.goto(childProducer.LproduceElement)
                cb.define(childProducer.LproduceElementDone)
                cb.assign(eltSettable, childProducer.element)

                emit(condIR, cb, region = childProducer.elementRegion, env = env.bind(elt, eltSettable))
                  .consume(cb,
                    cb.goto(LendOfStream),
                    code => cb.ifx(code.asBoolean.value,
                      cb.goto(LproduceElementDone),
                      cb.goto(LendOfStream)))

                cb.define(childProducer.LendOfStream)
                cb.goto(LendOfStream)
              }

              override val element: EmitCode = eltSettable

              override def close(cb: EmitCodeBuilder): Unit = {
                childProducer.close(cb)
              }
            }

            SStreamValue(producer)
          }

      case StreamDropWhile(a, elt, condIR) =>
        produce(a, cb)
          .map(cb) { case childStream: SStreamValue =>
            val childProducer = childStream.getProducer(mb)
            val eltSettable = mb.newEmitField("stream_drop_while_elt", childProducer.element.emitType)
            val doneComparisons = mb.genFieldThisRef[Boolean]("stream_drop_while_donecomparisons")

            val producer = new StreamProducer {
              override def method: EmitMethodBuilder[_] = mb
              override val length: Option[EmitCodeBuilder => Code[Int]] = None

              override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
                childProducer.initialize(cb, outerRegion)
                cb.assign(doneComparisons, false)
              }

              override val elementRegion: Settable[Region] = childProducer.elementRegion
              override val requiresMemoryManagementPerElement: Boolean = childProducer.requiresMemoryManagementPerElement
              override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>

                cb.goto(childProducer.LproduceElement)
                cb.define(childProducer.LproduceElementDone)
                cb.assign(eltSettable, childProducer.element)

                cb.ifx(doneComparisons, cb.goto(LproduceElementDone))

                val LdropThis = CodeLabel()
                val LdoneDropping = CodeLabel()
                emit(condIR, cb, region = childProducer.elementRegion, env = env.bind(elt, eltSettable))
                  .consume(cb,
                    cb.goto(LdoneDropping),
                    code => cb.ifx(code.asBoolean.value,
                      cb.goto(LdropThis),
                      cb.goto(LdoneDropping)))

                cb.define(LdropThis)
                if (childProducer.requiresMemoryManagementPerElement)
                  cb += childProducer.elementRegion.clearRegion()
                cb.goto(childProducer.LproduceElement)

                cb.define(LdoneDropping)
                cb.assign(doneComparisons, true)
                cb.goto(LproduceElementDone)

                cb.define(childProducer.LendOfStream)
                cb.goto(LendOfStream)
              }
              override val element: EmitCode = eltSettable

              override def close(cb: EmitCodeBuilder): Unit = {
                childProducer.close(cb)
              }
            }

            SStreamValue(producer)
          }

      case StreamMap(a, name, body) =>
        produce(a, cb)
          .map(cb) { case childStream: SStreamValue =>
            val childProducer = childStream.getProducer(mb)

            val bodyResult = EmitCode.fromI(mb) { cb =>
              cb.withScopedMaybeStreamValue(childProducer.element, "streammap_element") { childProducerElement =>
                emit(body,
                  cb = cb,
                  env = env.bind(name, childProducerElement),
                  region = childProducer.elementRegion)
              }
            }

            val producer: StreamProducer = new StreamProducer {
              override def method: EmitMethodBuilder[_] = mb
              override val length: Option[EmitCodeBuilder => Code[Int]] = childProducer.length

              override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
                childProducer.initialize(cb, outerRegion)
              }

              override val elementRegion: Settable[Region] = childProducer.elementRegion

              override val requiresMemoryManagementPerElement: Boolean = childProducer.requiresMemoryManagementPerElement

              override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
                cb.goto(childProducer.LproduceElement)
                cb.define(childProducer.LproduceElementDone)
                cb.goto(LproduceElementDone)
              }

              val element: EmitCode = bodyResult

              def close(cb: EmitCodeBuilder): Unit = childProducer.close(cb)
            }

            mb.implementLabel(childProducer.LendOfStream) { cb =>
              cb.goto(producer.LendOfStream)
            }

            SStreamValue(producer)
          }

      case x@StreamScan(childIR, zeroIR, accName, eltName, bodyIR) =>
        produce(childIR, cb).map(cb) { case childStream: SStreamValue =>
          val childProducer = childStream.getProducer(mb)

          val accEmitType = VirtualTypeWithReq(zeroIR.typ, emitter.ctx.req.lookupState(x).head.asInstanceOf[TypeWithRequiredness]).canonicalEmitType

          val accValueAccRegion = mb.newEmitField(accEmitType)
          val accValueEltRegion = mb.newEmitField(accEmitType)

          // accRegion is unused if requiresMemoryManagementPerElement is false
          val accRegion: Settable[Region] = if (childProducer.requiresMemoryManagementPerElement) mb.genFieldThisRef[Region]("streamscan_acc_region") else null
          val first = mb.genFieldThisRef[Boolean]("streamscan_first")

          val producer = new StreamProducer {
            override def method: EmitMethodBuilder[_] = mb
            override val length: Option[EmitCodeBuilder => Code[Int]] =
              childProducer.length.map(compL => (cb: EmitCodeBuilder) => compL(cb) + const(1))

            override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {

              if (childProducer.requiresMemoryManagementPerElement) {
                cb.assign(accRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
              }
              cb.assign(first, true)
              childProducer.initialize(cb, outerRegion)
            }

            override val elementRegion: Settable[Region] = childProducer.elementRegion

            override val requiresMemoryManagementPerElement: Boolean = childProducer.requiresMemoryManagementPerElement

            override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>

              val LcopyAndReturn = CodeLabel()

              cb.ifx(first, {

                cb.assign(first, false)
                cb.assign(accValueEltRegion, emit(zeroIR, cb, region = elementRegion).map(cb)(sc => sc.castTo(cb, elementRegion, accValueAccRegion.st)))

                cb.goto(LcopyAndReturn)
              })


              cb.goto(childProducer.LproduceElement)
              cb.define(childProducer.LproduceElementDone)

              if (requiresMemoryManagementPerElement) {
                // deep copy accumulator into element region, then clear accumulator region
                cb.assign(accValueEltRegion, accValueAccRegion.toI(cb).map(cb)(_.castTo(cb, childProducer.elementRegion, accEmitType.st, deepCopy = true)))
                cb += accRegion.clearRegion()
              }

              val bodyCode = cb.withScopedMaybeStreamValue(childProducer.element, "scan_child_elt") { ev =>
                emit(bodyIR, cb, env = env.bind((accName, accValueEltRegion), (eltName, ev)), region = childProducer.elementRegion)
                  .map(cb)(pc => pc.castTo(cb, childProducer.elementRegion, accEmitType.st, deepCopy = false))
              }

              cb.assign(accValueEltRegion, bodyCode)

              cb.define(LcopyAndReturn)

              if (requiresMemoryManagementPerElement) {
                cb.assign(accValueAccRegion, accValueEltRegion.toI(cb).map(cb)(pc => pc.castTo(cb, accRegion, accEmitType.st, deepCopy = true)))
              }

              cb.goto(LproduceElementDone)
            }

            val element: EmitCode = accValueEltRegion.load

            override def close(cb: EmitCodeBuilder): Unit = {
              childProducer.close(cb)
              if (requiresMemoryManagementPerElement)
                cb += accRegion.invalidate()
            }
          }

          mb.implementLabel(childProducer.LendOfStream) { cb =>
            cb.goto(producer.LendOfStream)
          }

          SStreamValue(producer)
        }

      case RunAggScan(child, name, init, seqs, result, states) =>
        val (newContainer, aggSetup, aggCleanup) = AggContainer.fromMethodBuilder(states.toArray, mb, "run_agg_scan")

        produce(child, cb).map(cb) { case childStream: SStreamValue =>
          val childProducer = childStream.getProducer(mb)

          val childEltField = mb.newEmitField("runaggscan_child_elt", childProducer.element.emitType)
          val bodyEnv = env.bind(name -> childEltField)
          val bodyResult = EmitCode.fromI(mb)(cb => emit(result, cb = cb, region = childProducer.elementRegion,
            env = bodyEnv, container = Some(newContainer)))
          val bodyResultField = mb.newEmitField("runaggscan_result_elt", bodyResult.emitType)

          val producer = new StreamProducer {
            override def method: EmitMethodBuilder[_] = mb
            override val length: Option[EmitCodeBuilder => Code[Int]] = childProducer.length

            override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
              aggSetup(cb)
              emitVoid(init, cb = cb, region = outerRegion, container = Some(newContainer))
              childProducer.initialize(cb, outerRegion)
            }

            override val elementRegion: Settable[Region] = childProducer.elementRegion
            override val requiresMemoryManagementPerElement: Boolean = childProducer.requiresMemoryManagementPerElement
            override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
              cb.goto(childProducer.LproduceElement)
              cb.define(childProducer.LproduceElementDone)
              cb.assign(childEltField, childProducer.element)
              cb.assign(bodyResultField, bodyResult.toI(cb))
              emitVoid(seqs, cb, region = elementRegion, env = bodyEnv, container = Some(newContainer))
              cb.goto(LproduceElementDone)
            }
            override val element: EmitCode = bodyResultField.load

            override def close(cb: EmitCodeBuilder): Unit = {
              childProducer.close(cb)
              aggCleanup(cb)
            }
          }

          mb.implementLabel(childProducer.LendOfStream) { cb =>
            cb.goto(producer.LendOfStream)
          }

          SStreamValue(producer)
        }

      case StreamWhiten(stream, newChunkName, prevWindowName, vecSize, windowSize, chunkSize, blockSize, normalizeAfterWhiten) =>
        produce(stream, cb).map(cb) { case blocks: SStreamValue =>
          val state = new LocalWhitening(cb, SizeValueStatic(vecSize.toLong), windowSize.toLong, chunkSize.toLong, blockSize.toLong, outerRegion, normalizeAfterWhiten)
          val eltType = blocks.st.elementType.asInstanceOf[SBaseStruct]
          var resultField: SSettable = null

          val blocksProducer = blocks.getProducer(cb.emb)
          val producer: StreamProducer = new StreamProducer {
            override def method: EmitMethodBuilder[_] = mb

            override val length: Option[EmitCodeBuilder => Code[Int]] =
              blocksProducer.length.map { l => (cb: EmitCodeBuilder) =>
                val len = cb.memoize(l(cb))
                len
              }

            override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
              state.reset(cb)
              blocksProducer.initialize(cb, outerRegion)
            }

            override val elementRegion: Settable[Region] = blocksProducer.elementRegion
            override val requiresMemoryManagementPerElement: Boolean = blocksProducer.requiresMemoryManagementPerElement

            override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
              cb.goto(blocksProducer.LproduceElement)
              cb.define(blocksProducer.LproduceElementDone)
              val row = blocksProducer.element.toI(cb).get(cb, "StreamWhiten: missing tuple").asBaseStruct
              row.loadField(cb, prevWindowName).consume(cb, {}, { prevWindow =>
                state.initializeWindow(cb, prevWindow.asNDArray)
              })
              val block = row.loadField(cb, newChunkName).get(cb, "StreamWhiten: missing chunk").asNDArray
              val whitenedBlock = LinalgCodeUtils.checkColMajorAndCopyIfNeeded(block, cb, elementRegion)
              state.whitenBlock(cb, whitenedBlock)
              // the 'newChunkName' field of 'row' is mutated in place and given
              // to the consumer
              val result = row.insert(cb, elementRegion, eltType.virtualType.asInstanceOf[TStruct], newChunkName -> EmitValue.present(whitenedBlock))
              resultField = mb.newPField("StreamWhiten_result", result.st)
              cb.assign(resultField, result)
              cb.goto(LproduceElementDone)
            }

            override val element: EmitCode = EmitCode.present(mb, resultField)

            override def close(cb: EmitCodeBuilder): Unit = {
              blocksProducer.close(cb)
            }
          }

          mb.implementLabel(blocksProducer.LendOfStream) { cb =>
            cb.goto(producer.LendOfStream)
          }

          SStreamValue(producer)
        }

      case StreamFlatMap(a, name, body) =>
        produce(a, cb).map(cb) { case outerStream: SStreamValue =>
          val outerProducer = outerStream.getProducer(mb)

          // variables used in control flow
          val first = mb.genFieldThisRef[Boolean]("flatmap_first")
          val innerUnclosed = mb.genFieldThisRef[Boolean]("flatmap_inner_unclosed")

          val innerStreamEmitCode = EmitCode.fromI(mb) { cb =>
            cb.withScopedMaybeStreamValue(outerProducer.element, "flatmap_outer_value") { outerProducerValue =>
              emit(body,
                cb = cb,
                env = env.bind(name, outerProducerValue),
                region = outerProducer.elementRegion)
            }
          }

          val resultElementRegion = mb.genFieldThisRef[Region]("flatmap_result_region")
          // grabbing emitcode.pv weird pattern but should be safe
          val innerProducer = innerStreamEmitCode.pv.asStream.getProducer(mb)

          val producer = new StreamProducer {
            override def method: EmitMethodBuilder[_] = mb
            override val length: Option[EmitCodeBuilder => Code[Int]] = None

            override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
              cb.assign(first, true)
              cb.assign(innerUnclosed, false)

              if (outerProducer.requiresMemoryManagementPerElement)
                cb.assign(outerProducer.elementRegion, Region.stagedCreate(Region.REGULAR, cb.emb.ecb.pool()))
              else
                cb.assign(outerProducer.elementRegion, outerRegion)

              outerProducer.initialize(cb, outerRegion)
            }

            override val elementRegion: Settable[Region] = resultElementRegion

            override val requiresMemoryManagementPerElement: Boolean = innerProducer.requiresMemoryManagementPerElement || outerProducer.requiresMemoryManagementPerElement

            override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
              val LnextOuter = CodeLabel()
              val LnextInner = CodeLabel()
              cb.ifx(first, {
                cb.assign(first, false)

                cb.define(LnextOuter)
                cb.define(innerProducer.LendOfStream)

                if (outerProducer.requiresMemoryManagementPerElement)
                  cb += outerProducer.elementRegion.clearRegion()


                cb.ifx(innerUnclosed, {
                  cb.assign(innerUnclosed, false)
                  innerProducer.close(cb)
                  if (innerProducer.requiresMemoryManagementPerElement) {
                    cb += innerProducer.elementRegion.invalidate()
                  }
                })

                cb.goto(outerProducer.LproduceElement)
                cb.define(outerProducer.LproduceElementDone)

                innerStreamEmitCode.toI(cb).consume(cb,
                  // missing inner streams mean we should go to the next outer element
                  cb.goto(LnextOuter),
                  {
                    _ =>
                      // the inner stream/producer is bound to a variable above
                      cb.assign(innerUnclosed, true)
                      if (innerProducer.requiresMemoryManagementPerElement)
                        cb.assign(innerProducer.elementRegion, Region.stagedCreate(Region.REGULAR, outerProducer.elementRegion.getPool()))
                      else
                        cb.assign(innerProducer.elementRegion, outerProducer.elementRegion)

                      innerProducer.initialize(cb, outerRegion)
                      cb.goto(LnextInner)
                  }
                )
              })

              cb.define(LnextInner)
              cb.goto(innerProducer.LproduceElement)
              cb.define(innerProducer.LproduceElementDone)

              if (requiresMemoryManagementPerElement) {
                cb += resultElementRegion.trackAndIncrementReferenceCountOf(innerProducer.elementRegion)

                // if outer requires memory management and inner doesn't,
                // then innerProducer.elementRegion is outerProducer.elementRegion
                // and we shouldn't clear it.
                if (innerProducer.requiresMemoryManagementPerElement) {
                  cb += resultElementRegion.trackAndIncrementReferenceCountOf(outerProducer.elementRegion)
                  cb += innerProducer.elementRegion.clearRegion()
                }
              }
              cb.goto(LproduceElementDone)
            }
            val element: EmitCode = innerProducer.element

            def close(cb: EmitCodeBuilder): Unit = {
              cb.ifx(innerUnclosed, {
                cb.assign(innerUnclosed, false)
                if (innerProducer.requiresMemoryManagementPerElement) {
                  cb += innerProducer.elementRegion.invalidate()
                }
                innerProducer.close(cb)
              })
              outerProducer.close(cb)

              if (outerProducer.requiresMemoryManagementPerElement)
                cb += outerProducer.elementRegion.invalidate()
            }
          }

          mb.implementLabel(outerProducer.LendOfStream) { cb =>
            cb.goto(producer.LendOfStream)
          }

          SStreamValue(producer)
        }

      case x@StreamLeftIntervalJoin(left, right, lKeyNames, rIntrvlName, lEltName, rEltName, body) =>

        produce(left, cb).flatMap(cb) { case lStream: SStreamValue =>
          produce(right, cb).map(cb) { case rStream: SStreamValue =>

            // map over the keyStream
            val lProd = lStream.getProducer(mb)
            val rProd = rStream.getProducer(mb)

            val rElemSTy = rProd.element.st.asInstanceOf[SBaseStruct]

            val intrvlParamTy: ParamType =
              rElemSTy.fieldTypes(rElemSTy.fieldIdx(rIntrvlName)).paramType

            val loadInterval: EmitMethodBuilder[_] =
              mb.ecb.defineEmitMethod("loadInterval", FastSeq(), intrvlParamTy) { mb =>
                mb.emitSCode { cb =>
                  mb.getEmitParam(cb, 0)
                    .get(cb)
                    .asBaseStruct
                    .loadField(cb, rIntrvlName)
                    .get(cb)
                }
              }

            val leftStructField = mb.newPField(lProd.element.st)
            val intervalResultField = mb.newPField(PCanonicalArray(rProd.element.st.storageType(), required = true).sType)
            val eltRegion = mb.genFieldThisRef[Region]("interval_join_region")

            val minHeap: StagedMinHeap =
              EmitMinHeap(mb.ecb.emodb, rElemSTy) { heapClassBuilder =>
                new EmitMinHeap.StagedComparator {

                  val parent: ThisFieldRef[_] =
                    heapClassBuilder.genFieldThisRef("parent")(mb.cb.ti)

                  val compare: EmitMethodBuilder[_] =
                    heapClassBuilder.defineEmitMethod("intervalRightEndpointCompare",
                      FastSeq(rElemSTy.paramType, rElemSTy.paramType),
                      IntInfo
                    ) { emb =>
                      emb.emitWithBuilder[Int] { cb =>
                        val l = cb.invokeSCode(loadInterval, parent, cb.emb.getEmitParam(cb, 1)).asInterval
                        val r = cb.invokeSCode(loadInterval, parent, cb.emb.getEmitParam(cb, 2)).asInterval
                        IntervalFunctions.intervalEndpointCompare(cb,
                          l.loadEnd(cb).get(cb), l.includesEnd,
                          r.loadEnd(cb).get(cb), r.includesEnd
                        )
                      }
                    }

                  override def init(cb: EmitCodeBuilder, enclosingRef: Value[AnyRef]): Unit =
                    cb.assignAny(parent, Code.checkcast(enclosingRef)(mb.cb.ti))

                  override def apply(cb: EmitCodeBuilder, a: SValue, b: SValue): Value[Int] =
                    cb.invokeCode(compare, cb._this, a, b)
                }
              }(mb.ecb)

            SStreamValue {
              new StreamProducer {
                override def method: EmitMethodBuilder[_] =
                  mb

                override val length: Option[EmitCodeBuilder => Code[Int]] =
                  lProd.length

                override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
                  lProd.initialize(cb, outerRegion)
                  rProd.initialize(cb, outerRegion)
                  minHeap.init(cb, mb.ecb.pool())
                }

                override val elementRegion: Settable[Region] =
                  eltRegion

                override val requiresMemoryManagementPerElement: Boolean =
                  lProd.requiresMemoryManagementPerElement || rProd.requiresMemoryManagementPerElement

                override val LproduceElement: CodeLabel =
                  mb.defineAndImplementLabel { cb =>
                    cb.goto(lProd.LproduceElement)
                    cb.define(lProd.LproduceElementDone)
                    val row = lProd.element.toI(cb).get(cb).asBaseStruct
                    val key = row.subset(lKeyNames: _*)

                    cb.loop { Lrecur =>
                      cb.ifx(minHeap.nonEmpty(cb),
                        cb.ifx({
                          val interval = cb.invokeSCode(loadInterval, cb._this, minHeap.peek(cb)).asInterval
                          IntervalFunctions.intervalContains(cb, interval, key).get(cb).asBoolean.value
                        }, {
                          minHeap.pop(cb)
                          cb.goto(Lrecur)
                        })
                      )
                    }

                    minHeap.realloc(cb)

                    // now pull from the interval stream and insert into minheap while
                    // the most-recent interval left endpoint is greater than the current key

                    cb.goto(rProd.LproduceElement)
                    cb.define(rProd.LproduceElementDone)

                    val rElem: SValue = rProd.element.toI(cb).get(cb)
                    val rInterval = cb.invokeSCode(loadInterval, cb._this, rElem).asInterval

                    cb.ifx(
                      IntervalFunctions.pointGTIntervalEndpoint(cb, key, rInterval.loadEnd(cb).get(cb), rInterval.includesEnd),
                      cb.goto(rProd.LproduceElement)
                    )

                    val LallIntervalsFound = CodeLabel()
                    // need lookahead
                    cb.ifx(
                      IntervalFunctions.pointLTIntervalEndpoint(cb, key, rInterval.loadStart(cb).get(cb), rInterval.includesStart),
                      cb.goto(LallIntervalsFound)
                    )

                    // we've found the first interval that contains key
                    // add interval to minheap
                    minHeap.push(cb, rElem)
                    cb.goto(rProd.LproduceElement)

                    cb.define(LallIntervalsFound)
                    cb.assign(leftStructField, row)
                    cb.assign(intervalResultField, minHeap.toArray(cb, eltRegion))
                    cb.goto(LproduceElementDone)

                    cb.define(lProd.LendOfStream)
                    cb.goto(LendOfStream)
                  }

                override val element: EmitCode =
                  EmitCode.fromI(mb) { cb =>
                    emit(body, cb, region = eltRegion, env = env.bind(
                      lEltName -> EmitValue.present(leftStructField),
                      rEltName -> EmitValue.present(intervalResultField)
                    ))
                  }

                override def close(cb: EmitCodeBuilder): Unit = {
                  minHeap.close(cb)
                  rProd.close(cb)
                  lProd.close(cb)
                }
              }
            }
          }
        }

      case x@StreamJoinRightDistinct(leftIR, rightIR, lKey, rKey, leftName, rightName, joinIR, joinType) =>
        produce(leftIR, cb).flatMap(cb) { case leftStream: SStreamValue =>
          produce(rightIR, cb).map(cb) { case rightStream: SStreamValue =>

            val leftProducer = leftStream.getProducer(mb)
            val rightProducer = rightStream.getProducer(mb)

            val lEltType = leftProducer.element.emitType
            val rEltType = rightProducer.element.emitType

            def compare(cb: EmitCodeBuilder, lelt: EmitValue, relt: EmitValue): Code[Int] = {
              assert(lelt.emitType == lEltType)
              assert(relt.emitType == rEltType)
              if (x.isIntervalJoin) {
                val rhs = relt.toI(cb).flatMap(cb)(_.asBaseStruct.loadField(cb, rKey(0)))
                val result = cb.newLocal[Int]("SJRD-interval-compare-result")
                rhs.consume(cb, {
                  cb.assign(result, -1)
                }, { case interval: SIntervalValue =>
                  val lhs = lelt.toI(cb).flatMap(cb)(_.asBaseStruct.loadField(cb, lKey(0)))
                  lhs.consume(cb, {
                    cb.assign(result, 1)
                  }, { point =>
                    val c = IntervalFunctions.pointIntervalCompare(cb, point, interval)
                    c.consume(cb, {
                      // One of the interval endpoints is missing. In this case,
                      // consider the point greater, so that the join advances
                      // past the bad interval, keeping the point.
                      cb.assign(result, 1)
                    }, { c =>
                      cb.assign(result, c.asInt.value)
                    })
                  })
                })
                result
              } else {
                val lhs = lelt.map(cb)(_.asBaseStruct.subset(lKey: _*))
                val rhs = relt.map(cb)(_.asBaseStruct.subset(rKey: _*))
                StructOrdering.make(lhs.st.asInstanceOf[SBaseStruct], rhs.st.asInstanceOf[SBaseStruct],
                  cb.emb.ecb, missingFieldsEqual = false)
                  .compare(cb, lhs, rhs, missingEqual = false)
              }
            }

            // these variables are used as inputs to the joinF
            val lx = mb.newEmitField("streamjoin_lx", lEltType) // last value received from left
            val rx = mb.newEmitField("streamjoin_rx", rEltType) // last value received from right

            val lxOut: EmitSettable = joinType match {
              case "inner" | "left" => lx
              case "outer" | "right" => mb.newEmitField("streamjoin_lxout", lx.emitType.copy(required = false))
            }
            val rxOut: EmitSettable = joinType match {
              case "inner" | "right" => rx
              case "outer" | "left" => mb.newEmitField("streamjoin_rxout", rx.emitType.copy(required = false))
            }

            val _elementRegion = mb.genFieldThisRef[Region]("join_right_distinct_element_region")
            val _requiresMemoryManagementPerElement = leftProducer.requiresMemoryManagementPerElement || rightProducer.requiresMemoryManagementPerElement

            val joinResult = EmitCode.fromI(mb)(cb => emit(joinIR, cb,
              region = _elementRegion,
              env = env.bind(leftName -> lxOut, rightName -> rxOut)))

            def sharedInit(cb: EmitCodeBuilder): Unit = {
              if (rightProducer.requiresMemoryManagementPerElement)
                cb.assign(rightProducer.elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
              else
                cb.assign(rightProducer.elementRegion, outerRegion)
              if (leftProducer.requiresMemoryManagementPerElement)
                cb.assign(leftProducer.elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
              else
                cb.assign(leftProducer.elementRegion, outerRegion)

              leftProducer.initialize(cb, outerRegion)
              rightProducer.initialize(cb, outerRegion)
            }

            def sharedClose(cb: EmitCodeBuilder): Unit = {
              leftProducer.close(cb)
              rightProducer.close(cb)
              if (leftProducer.requiresMemoryManagementPerElement)
                cb += leftProducer.elementRegion.invalidate()

              if (rightProducer.requiresMemoryManagementPerElement)
                cb += rightProducer.elementRegion.invalidate()
            }

            joinType match {
              case "left" =>
                val rightEOS = mb.genFieldThisRef[Boolean]("left_join_right_distinct_rightEOS")
                val pulledRight = mb.genFieldThisRef[Boolean]("left_join_right_distinct_pulledRight]")

                val producer = new StreamProducer {
                  override def method: EmitMethodBuilder[_] = mb
                  override val length: Option[EmitCodeBuilder => Code[Int]] = leftProducer.length

                  override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
                    cb.assign(rightEOS, false)
                    cb.assign(pulledRight, false)

                    sharedInit(cb)
                  }

                  override val elementRegion: Settable[Region] = _elementRegion
                  override val requiresMemoryManagementPerElement: Boolean = _requiresMemoryManagementPerElement
                  override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>

                    if (leftProducer.requiresMemoryManagementPerElement)
                      cb += leftProducer.elementRegion.clearRegion()
                    cb.goto(leftProducer.LproduceElement)
                    cb.define(leftProducer.LproduceElementDone)
                    cb.assign(lx, leftProducer.element)

                    // if right stream is exhausted, return immediately
                    cb.ifx(rightEOS, cb.goto(LproduceElementDone))

                    val Lpush = CodeLabel()

                    val LpullRight = CodeLabel()
                    cb.ifx(!pulledRight, {
                      cb.assign(pulledRight, true)
                      cb.goto(LpullRight)
                    })

                    val Lcompare = CodeLabel()
                    cb.define(Lcompare)
                    val c = cb.newLocal[Int]("left_join_right_distinct_c", compare(cb, lx, rx))
                    cb.ifx(c > 0, cb.goto(LpullRight))

                    cb.ifx(c < 0, {
                      cb.assign(rxOut, EmitCode.missing(mb, rxOut.st))
                    }, {
                      // c == 0
                      if (rightProducer.requiresMemoryManagementPerElement) {
                        cb += elementRegion.trackAndIncrementReferenceCountOf(rightProducer.elementRegion)
                      }
                      cb.assign(rxOut, rx)
                    })

                    cb.goto(Lpush)

                    mb.implementLabel(Lpush) { cb =>
                      if (leftProducer.requiresMemoryManagementPerElement)
                        cb += elementRegion.trackAndIncrementReferenceCountOf(leftProducer.elementRegion)

                      cb.goto(LproduceElementDone)
                    }

                    mb.implementLabel(LpullRight) { cb =>
                      if (rightProducer.requiresMemoryManagementPerElement) {
                        cb += rightProducer.elementRegion.clearRegion()
                      }
                      cb.goto(rightProducer.LproduceElement)
                      cb.define(rightProducer.LproduceElementDone)
                      cb.assign(rx, rightProducer.element)
                      cb.goto(Lcompare)
                    }

                    // if right stream ends before left
                    mb.implementLabel(rightProducer.LendOfStream) { cb =>
                      cb.assign(rxOut, EmitCode.missing(mb, rxOut.st))
                      cb.assign(rightEOS, true)
                      cb.goto(Lpush)
                    }

                    mb.implementLabel(leftProducer.LendOfStream) { cb => cb.goto(LendOfStream) }
                  }
                  override val element: EmitCode = joinResult

                  override def close(cb: EmitCodeBuilder): Unit = {
                    sharedClose(cb)
                  }
                }

                SStreamValue(producer)

              case "right" =>
                val leftEOS = mb.genFieldThisRef[Boolean]("left_join_right_distinct_leftEOS")
                val pulledRight = mb.genFieldThisRef[Boolean]("left_join_right_distinct_pulledRight]")
                val pushedRight = mb.genFieldThisRef[Boolean]("left_join_right_distinct_pulledRight]")
                val c = mb.genFieldThisRef[Int]("join_right_distinct_compResult")

                val producer = new StreamProducer {
                  override def method: EmitMethodBuilder[_] = mb
                  override val length: Option[EmitCodeBuilder => Code[Int]] = None

                  override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
                    cb.assign(leftEOS, false)
                    cb.assign(pulledRight, false)
                    cb.assign(pushedRight, false)
                    cb.assign(c, 0)
                    sharedInit(cb)
                  }

                  override val elementRegion: Settable[Region] = _elementRegion
                  override val requiresMemoryManagementPerElement: Boolean = _requiresMemoryManagementPerElement
                  override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
                    val Lpush = CodeLabel()
                    val LpullRight = CodeLabel()
                    val LpullLeft = CodeLabel()
                    val Lcompare = CodeLabel()
                    val LmaybePullRight = CodeLabel()

                    cb.ifx(leftEOS, cb.goto(Lcompare))
                    cb.ifx(c <= 0, cb.goto(LpullLeft), cb.goto(LpullRight))

                    cb.define(Lcompare)
                    cb.ifx(leftEOS, {
                      cb.ifx(pushedRight,
                        cb.goto(LpullRight),
                        cb.goto(Lpush))
                    })
                    cb.assign(c, compare(cb, lx, rx))

                    cb.ifx(c < 0, cb.goto(LpullLeft))

                    cb.ifx(c > 0, {
                      cb.ifx(pushedRight, cb.goto(LpullRight))
                      cb.assign(lxOut, EmitCode.missing(mb, lxOut.st))
                    }, {
                      // c == 0
                      if (leftProducer.requiresMemoryManagementPerElement)
                        cb += elementRegion.trackAndIncrementReferenceCountOf(leftProducer.elementRegion)
                      cb.assign(lxOut, lx)
                    })

                    cb.goto(Lpush)

                    mb.implementLabel(LmaybePullRight) { cb =>
                      cb.ifx(!pulledRight, {
                        cb.assign(pulledRight, true)
                        cb.goto(LpullRight)
                      },
                        cb.goto(Lcompare))
                    }

                    mb.implementLabel(LpullLeft) { cb =>
                      if (leftProducer.requiresMemoryManagementPerElement)
                        cb += leftProducer.elementRegion.clearRegion()
                      cb.goto(leftProducer.LproduceElement)
                      cb.define(leftProducer.LproduceElementDone)
                      cb.assign(lx, leftProducer.element)
                      cb.goto(LmaybePullRight)
                    }

                    mb.implementLabel(LpullRight) { cb =>
                      if (rightProducer.requiresMemoryManagementPerElement)
                        cb += rightProducer.elementRegion.clearRegion()
                      cb.goto(rightProducer.LproduceElement)
                      cb.define(rightProducer.LproduceElementDone)
                      cb.assign(rx, rightProducer.element)
                      cb.assign(pushedRight, false)
                      cb.goto(Lcompare)
                    }

                    mb.implementLabel(Lpush) { cb =>
                      if (rightProducer.requiresMemoryManagementPerElement)
                        cb += elementRegion.trackAndIncrementReferenceCountOf(rightProducer.elementRegion)

                      cb.assign(pushedRight, true)
                      cb.goto(LproduceElementDone)
                    }

                    // if left stream ends before right
                    mb.implementLabel(leftProducer.LendOfStream) { cb =>
                      cb.assign(lxOut, EmitCode.missing(mb, lxOut.st))
                      cb.assign(leftEOS, true)
                      cb.goto(LmaybePullRight)
                    }

                    // end if right stream ends
                    mb.implementLabel(rightProducer.LendOfStream) { cb => cb.goto(LendOfStream) }
                  }

                  override val element: EmitCode = joinResult

                  override def close(cb: EmitCodeBuilder): Unit = {
                    sharedClose(cb)
                  }
                }
                SStreamValue(producer)

              case "inner" =>
                val pulledRight = mb.genFieldThisRef[Boolean]("left_join_right_distinct_pulledRight]")

                val producer = new StreamProducer {
                  override def method: EmitMethodBuilder[_] = mb
                  override val length: Option[EmitCodeBuilder => Code[Int]] = None

                  override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
                    cb.assign(pulledRight, false)
                    sharedInit(cb)
                  }

                  override val elementRegion: Settable[Region] = _elementRegion
                  override val requiresMemoryManagementPerElement: Boolean = _requiresMemoryManagementPerElement
                  override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>

                    if (leftProducer.requiresMemoryManagementPerElement)
                      cb += leftProducer.elementRegion.clearRegion()
                    cb.goto(leftProducer.LproduceElement)
                    cb.define(leftProducer.LproduceElementDone)
                    cb.assign(lx, leftProducer.element)

                    val LpullRight = CodeLabel()
                    cb.ifx(!pulledRight, {
                      cb.assign(pulledRight, true)
                      cb.goto(LpullRight)
                    })

                    val Lcompare = CodeLabel()
                    cb.define(Lcompare)
                    val c = cb.newLocal[Int]("left_join_right_distinct_c", compare(cb, lx, rx))
                    cb.ifx(c > 0, cb.goto(LpullRight))

                    cb.ifx(c < 0, {
                      if (leftProducer.requiresMemoryManagementPerElement)
                        cb += leftProducer.elementRegion.clearRegion()
                      cb.goto(leftProducer.LproduceElement)
                    })

                    cb.goto(LproduceElementDone)

                    mb.implementLabel(LpullRight) { cb =>
                      if (rightProducer.requiresMemoryManagementPerElement)
                        cb += rightProducer.elementRegion.clearRegion()
                      cb.goto(rightProducer.LproduceElement)
                    }

                    mb.implementLabel(rightProducer.LproduceElementDone) { cb =>
                      cb.assign(rx, rightProducer.element)
                      cb.goto(Lcompare)
                    }

                    // Both producer EOS labels should jump directly to EOS
                    mb.implementLabel(rightProducer.LendOfStream) { cb => cb.goto(LendOfStream) }
                    mb.implementLabel(leftProducer.LendOfStream) { cb => cb.goto(LendOfStream) }
                  }
                  override val element: EmitCode = joinResult

                  override def close(cb: EmitCodeBuilder): Unit = {
                    sharedClose(cb)
                  }
                }

                SStreamValue(producer)

              case "outer" =>
                val pulledRight = mb.genFieldThisRef[Boolean]("join_right_distinct_pulledRight")
                val pushedRight = mb.genFieldThisRef[Boolean]("join_right_distinct_pushedRight")
                val rightEOS = mb.genFieldThisRef[Boolean]("join_right_distinct_rightEOS")
                val lOutMissing = mb.genFieldThisRef[Boolean]("join_right_distinct_leftMissing")
                val rOutMissing = mb.genFieldThisRef[Boolean]("join_right_distinct_rightMissing")
                val leftEOS = mb.genFieldThisRef[Boolean]("join_right_distinct_leftEOS")
                val c = mb.genFieldThisRef[Int]("join_right_distinct_compResult")

                val producer = new StreamProducer {
                  override def method: EmitMethodBuilder[_] = mb
                  override val length: Option[EmitCodeBuilder => Code[Int]] = None

                  override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
                    cb.assign(pulledRight, false)
                    cb.assign(leftEOS, false)
                    cb.assign(rightEOS, false)
                    cb.assign(c, 0) // lets us start stream with a pull from both

                    sharedInit(cb)
                  }

                  override val elementRegion: Settable[Region] = _elementRegion
                  override val requiresMemoryManagementPerElement: Boolean = _requiresMemoryManagementPerElement
                  override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>

                    val LpullRight = CodeLabel()
                    val LpullLeft = CodeLabel()
                    val Lpush = CodeLabel()

                    cb.ifx(leftEOS,
                      cb.goto(LpullRight),
                      cb.ifx(rightEOS,
                        cb.goto(LpullLeft),
                        cb.ifx(c <= 0,
                          cb.goto(LpullLeft),
                          cb.goto(LpullRight))))

                    cb.define(LpullRight)
                    if (rightProducer.requiresMemoryManagementPerElement)
                      cb += rightProducer.elementRegion.clearRegion()
                    cb.assign(pulledRight, true)
                    cb.goto(rightProducer.LproduceElement)

                    cb.define(LpullLeft)
                    if (leftProducer.requiresMemoryManagementPerElement)
                      cb += leftProducer.elementRegion.clearRegion()
                    cb.goto(leftProducer.LproduceElement)

                    val Lcompare = CodeLabel()
                    mb.implementLabel(Lcompare) { cb =>
                      cb.assign(c, compare(cb, lx, rx))
                      cb.assign(lOutMissing, false)
                      cb.assign(rOutMissing, false)
                      cb.ifx(c > 0,
                        {
                          cb.ifx(pulledRight && !pushedRight, {
                            cb.assign(lOutMissing, true)
                            if (rightProducer.requiresMemoryManagementPerElement) {
                              cb += elementRegion.trackAndIncrementReferenceCountOf(rightProducer.elementRegion)
                            }
                            cb.goto(Lpush)
                          },
                            cb.goto(LpullRight)
                          )
                        },
                        {
                          cb.ifx(c < 0,
                            {
                              cb.assign(rOutMissing, true)
                              if (leftProducer.requiresMemoryManagementPerElement) {
                                cb += elementRegion.trackAndIncrementReferenceCountOf(leftProducer.elementRegion)
                              }
                              cb.goto(Lpush)
                            },
                            {
                              // c == 0
                              if (leftProducer.requiresMemoryManagementPerElement) {
                                cb += elementRegion.trackAndIncrementReferenceCountOf(leftProducer.elementRegion)
                              }
                              if (rightProducer.requiresMemoryManagementPerElement) {
                                cb += elementRegion.trackAndIncrementReferenceCountOf(rightProducer.elementRegion)
                              }
                              cb.goto(Lpush)
                            })
                        })
                    }

                    mb.implementLabel(Lpush) { cb =>
                      cb.ifx(lOutMissing,
                        cb.assign(lxOut, EmitCode.missing(mb, lxOut.st)),
                        cb.assign(lxOut, lx)
                      )
                      cb.ifx(rOutMissing,
                        cb.assign(rxOut, EmitCode.missing(mb, rxOut.st)),
                        {
                          cb.assign(rxOut, rx)
                          cb.assign(pushedRight, true)
                        })
                      cb.goto(LproduceElementDone)
                    }


                    mb.implementLabel(rightProducer.LproduceElementDone) { cb =>
                      cb.assign(rx, rightProducer.element)
                      cb.assign(pushedRight, false)
                      cb.ifx(leftEOS, cb.goto(Lpush), cb.goto(Lcompare))
                    }

                    mb.implementLabel(leftProducer.LproduceElementDone) { cb =>
                      cb.assign(lx, leftProducer.element)
                      cb.ifx(pulledRight,
                        cb.ifx(rightEOS,
                          {
                            if (leftProducer.requiresMemoryManagementPerElement) {
                              cb += elementRegion.trackAndIncrementReferenceCountOf(leftProducer.elementRegion)
                            }
                            cb.goto(Lpush)
                          },
                          {
                            cb.goto(Lcompare)
                          }
                        ),
                        cb.goto(LpullRight)
                      )
                    }

                    mb.implementLabel(leftProducer.LendOfStream) { cb =>
                      cb.ifx(rightEOS,
                        cb.goto(LendOfStream),
                        {
                          cb.assign(leftEOS, true)
                          cb.assign(lOutMissing, true)
                          cb.assign(rOutMissing, false)
                          cb.ifx(pulledRight && !pushedRight,
                            {
                              if (rightProducer.requiresMemoryManagementPerElement) {
                                cb += elementRegion.trackAndIncrementReferenceCountOf(rightProducer.elementRegion)
                              }
                              cb.goto(Lpush)
                            },
                            {
                              if (rightProducer.requiresMemoryManagementPerElement) {
                                cb += rightProducer.elementRegion.clearRegion()
                              }
                              cb.goto(LpullRight)
                            })
                        })
                    }

                    mb.implementLabel(rightProducer.LendOfStream) { cb =>
                      cb.ifx(leftEOS, cb.goto(LendOfStream))
                      cb.assign(rightEOS, true)
                      cb.assign(lOutMissing, false)
                      cb.assign(rOutMissing, true)

                      if (leftProducer.requiresMemoryManagementPerElement) {
                        cb += elementRegion.trackAndIncrementReferenceCountOf(leftProducer.elementRegion)
                        cb += leftProducer.elementRegion.clearRegion()
                      }
                      cb.goto(Lpush)
                    }
                  }
                  override val element: EmitCode = joinResult

                  override def close(cb: EmitCodeBuilder): Unit = {
                    sharedClose(cb)
                  }
                }

                SStreamValue(producer)
            }
          }
        }

      case StreamGroupByKey(a, key, missingEqual) =>
        produce(a, cb).map(cb) { case childStream: SStreamValue =>

          val childProducer = childStream.getProducer(mb)

          val xCurElt = mb.newPField("st_grpby_curelt", childProducer.element.st)

          val keyRegion = mb.genFieldThisRef[Region]("st_groupby_key_region")
          def subsetCode = xCurElt.asBaseStruct.subset(key: _*)
          val curKey = mb.newPField("st_grpby_curkey", subsetCode.st)

          // This type shouldn't be a subset struct, since it is copied deeply.
          // We don't want to deep copy the parent.
          val lastKey = mb.newPField("st_grpby_lastkey", subsetCode.st.copiedType)

          val eos = mb.genFieldThisRef[Boolean]("st_grpby_eos")
          val nextGroupReady = mb.genFieldThisRef[Boolean]("streamgroupbykey_nextready")
          val inOuter = mb.genFieldThisRef[Boolean]("streamgroupbykey_inouter")
          val first = mb.genFieldThisRef[Boolean]("streamgroupbykey_first")

          // cannot reuse childProducer.elementRegion because consumers might free the region, even though
          // the outer producer needs to continue pulling. We could add more control flow that sets some
          // boolean flag when the inner stream is closed, and the outer producer reassigns a region if
          // that flag is set, but that design seems more complicated
          val innerResultRegion = mb.genFieldThisRef[Region]("streamgroupbykey_inner_result_region")

          val outerElementRegion = mb.genFieldThisRef[Region]("streamgroupbykey_outer_elt_region")

          def equiv(cb: EmitCodeBuilder, l: SBaseStructValue, r: SBaseStructValue): Value[Boolean] =
            StructOrdering.make(l.st, r.st, cb.emb.ecb, missingFieldsEqual = missingEqual).equivNonnull(cb, l, r)

          val LchildProduceDoneInner = CodeLabel()
          val LchildProduceDoneOuter = CodeLabel()
          val innerProducer = new StreamProducer {
            override def method: EmitMethodBuilder[_] = mb
            override val length: Option[EmitCodeBuilder => Code[Int]] = None

            override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {}

            override val elementRegion: Settable[Region] = innerResultRegion
            override val requiresMemoryManagementPerElement: Boolean = childProducer.requiresMemoryManagementPerElement
            override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
              val LelementReady = CodeLabel()

              // the first pull from the inner stream has the next record ready to go from the outer stream
              cb.ifx(inOuter, {
                cb.assign(inOuter, false)
                cb.goto(LelementReady)
              })

              if (childProducer.requiresMemoryManagementPerElement)
                cb += childProducer.elementRegion.clearRegion()
              cb.goto(childProducer.LproduceElement)
              // xElt and curKey are assigned before this label is jumped to
              cb.define(LchildProduceDoneInner)

              // if not equivalent, end inner stream and prepare for next outer iteration
              cb.ifx(!equiv(cb, curKey.asBaseStruct, lastKey.asBaseStruct), {
                if (requiresMemoryManagementPerElement)
                  cb += keyRegion.clearRegion()

                cb.assign(lastKey, subsetCode.castTo(cb, keyRegion, lastKey.st, deepCopy = true))
                cb.assign(nextGroupReady, true)
                cb.assign(inOuter, true)
                cb.goto(LendOfStream)
              })

              cb.define(LelementReady)

              if (requiresMemoryManagementPerElement) {
                cb += innerResultRegion.trackAndIncrementReferenceCountOf(childProducer.elementRegion)
              }

              cb.goto(LproduceElementDone)
            }
            override val element: EmitCode = EmitCode.present(mb, xCurElt)

            override def close(cb: EmitCodeBuilder): Unit = {}
          }

          val outerProducer = new StreamProducer {
            override def method: EmitMethodBuilder[_] = mb
            override val length: Option[EmitCodeBuilder => Code[Int]] = None

            override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
              cb.assign(nextGroupReady, false)
              cb.assign(eos, false)
              cb.assign(inOuter, true)
              cb.assign(first, true)

              if (childProducer.requiresMemoryManagementPerElement) {
                cb.assign(keyRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
                cb.assign(childProducer.elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
              } else {
                cb.assign(keyRegion, outerRegion)
                cb.assign(childProducer.elementRegion, outerRegion)
              }

              childProducer.initialize(cb, outerRegion)
            }

            override val elementRegion: Settable[Region] = outerElementRegion
            override val requiresMemoryManagementPerElement: Boolean = childProducer.requiresMemoryManagementPerElement
            override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
              cb.ifx(eos, {
                cb.goto(LendOfStream)
              })

              val LinnerStreamReady = CodeLabel()

              cb.ifx(nextGroupReady, cb.goto(LinnerStreamReady))

              cb.assign(inOuter, true)

              if (childProducer.requiresMemoryManagementPerElement)
                cb += childProducer.elementRegion.clearRegion()
              cb.goto(childProducer.LproduceElement)
              // xElt and curKey are assigned before this label is jumped to
              cb.define(LchildProduceDoneOuter)

              val LdifferentKey = CodeLabel()

              cb.ifx(first, {
                cb.assign(first, false)
                cb.goto(LdifferentKey)
              })

              // if equiv, go to next element. Otherwise, fall through to next group
              cb.ifx(equiv(cb, curKey.asBaseStruct, lastKey.asBaseStruct), {
                if (childProducer.requiresMemoryManagementPerElement)
                  cb += childProducer.elementRegion.clearRegion()
                cb.goto(childProducer.LproduceElement)
              })

              cb.define(LdifferentKey)
              if (requiresMemoryManagementPerElement)
                cb += keyRegion.clearRegion()

              cb.assign(lastKey, subsetCode.castTo(cb, keyRegion, lastKey.st, deepCopy = true))

              cb.define(LinnerStreamReady)
              cb.assign(nextGroupReady, false)
              cb.goto(LproduceElementDone)
            }

            override val element: EmitCode = EmitCode.present(mb, SStreamValue(innerProducer))

            override def close(cb: EmitCodeBuilder): Unit = {
              childProducer.close(cb)
              if (childProducer.requiresMemoryManagementPerElement) {
                cb += keyRegion.invalidate()
                cb += childProducer.elementRegion.invalidate()
              }
            }
          }

          mb.implementLabel(childProducer.LendOfStream) { cb =>
            cb.assign(eos, true)
            cb.ifx(inOuter,
              cb.goto(outerProducer.LendOfStream),
              cb.goto(innerProducer.LendOfStream)
            )
          }

          mb.implementLabel(childProducer.LproduceElementDone) { cb =>
            cb.assign(xCurElt, childProducer.element.toI(cb).get(cb))
            cb.assign(curKey, subsetCode)
            cb.ifx(inOuter, cb.goto(LchildProduceDoneOuter), cb.goto(LchildProduceDoneInner))
          }

          SStreamValue(outerProducer)
        }

      case StreamGrouped(a, groupSize) =>
        produce(a, cb).flatMap(cb) { case childStream: SStreamValue =>

          emit(groupSize, cb).map(cb) { case groupSize: SInt32Value =>

            val n = mb.genFieldThisRef[Int]("streamgrouped_n")

            val childProducer = childStream.getProducer(mb)

            val xCounter = mb.genFieldThisRef[Int]("streamgrouped_ctr")
            val inOuter = mb.genFieldThisRef[Boolean]("streamgrouped_io")
            val eos = mb.genFieldThisRef[Boolean]("streamgrouped_eos")

            val outerElementRegion = mb.genFieldThisRef[Region]("streamgrouped_outer_elt_region")

            // cannot reuse childProducer.elementRegion because consumers might free the region, even though
            // the outer producer needs to continue pulling. We could add more control flow that sets some
            // boolean flag when the inner stream is closed, and the outer producer reassigns a region if
            // that flag is set, but that design seems more complicated
            val innerResultRegion = mb.genFieldThisRef[Region]("streamgrouped_inner_result_region")

            val LchildProduceDoneInner = CodeLabel()
            val LchildProduceDoneOuter = CodeLabel()
            val innerProducer = new StreamProducer {
              override def method: EmitMethodBuilder[_] = mb
              override val length: Option[EmitCodeBuilder => Code[Int]] = None

              override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {}

              override val elementRegion: Settable[Region] = innerResultRegion
              override val requiresMemoryManagementPerElement: Boolean = childProducer.requiresMemoryManagementPerElement
              override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
                cb.ifx(inOuter, {
                  cb.assign(inOuter, false)
                  cb.ifx(xCounter.cne(1), cb._fatal(s"streamgrouped inner producer error, xCounter=", xCounter.toS))
                  cb.goto(LchildProduceDoneInner)
                })
                cb.ifx(xCounter >= n, {
                  cb.assign(inOuter, true)
                  cb.goto(LendOfStream)
                })

                cb.goto(childProducer.LproduceElement)
                cb.define(LchildProduceDoneInner)

                if (childProducer.requiresMemoryManagementPerElement) {
                  cb += innerResultRegion.trackAndIncrementReferenceCountOf(childProducer.elementRegion)
                  cb += childProducer.elementRegion.clearRegion()
                }

                cb.goto(LproduceElementDone)
              }
              override val element: EmitCode = childProducer.element

              override def close(cb: EmitCodeBuilder): Unit = {}
            }
            val innerStreamCode = EmitCode.present(mb, SStreamValue(innerProducer))

            val outerProducer = new StreamProducer {
              override def method: EmitMethodBuilder[_] = mb
              override val length: Option[EmitCodeBuilder => Code[Int]] =
                childProducer.length.map(compL => (cb: EmitCodeBuilder) => ((compL(cb).toL + n.toL - 1L) / n.toL).toI)

              override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
                cb.assign(n, groupSize.value)
                cb.ifx(n <= 0, cb._fatal(s"stream grouped: non-positive size: ", n.toS))
                cb.assign(eos, false)
                cb.assign(xCounter, n)

                if (childProducer.requiresMemoryManagementPerElement) {
                  cb.assign(childProducer.elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
                } else {
                  cb.assign(childProducer.elementRegion, outerRegion)
                }

                childProducer.initialize(cb, outerRegion)
              }

              override val elementRegion: Settable[Region] = outerElementRegion
              override val requiresMemoryManagementPerElement: Boolean = childProducer.requiresMemoryManagementPerElement
              override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
                cb.ifx(eos, {
                  cb.goto(LendOfStream)
                })

                cb.assign(inOuter, true)
                cb.define(LchildProduceDoneOuter)


                cb.ifx(xCounter <= n,
                  {
                    if (childProducer.requiresMemoryManagementPerElement)
                      cb += childProducer.elementRegion.clearRegion()
                    cb.goto(childProducer.LproduceElement)
                  })
                cb.assign(xCounter, 1)
                cb.goto(LproduceElementDone)
              }
              override val element: EmitCode = innerStreamCode

              override def close(cb: EmitCodeBuilder): Unit = {
                childProducer.close(cb)
                if (childProducer.requiresMemoryManagementPerElement)
                  cb += childProducer.elementRegion.invalidate()
              }
            }

            mb.implementLabel(childProducer.LendOfStream) { cb =>
              cb.assign(eos, true)
              cb.ifx(inOuter,
                cb.goto(outerProducer.LendOfStream),
                cb.goto(innerProducer.LendOfStream)
              )
            }

            mb.implementLabel(childProducer.LproduceElementDone) { cb =>
              cb.assign(xCounter, xCounter + 1)
              cb.ifx(inOuter, cb.goto(LchildProduceDoneOuter), cb.goto(LchildProduceDoneInner))
            }

            SStreamValue(outerProducer)
          }
        }

      case StreamZip(as, names, body, behavior, errorID) =>
        IEmitCode.multiMapEmitCodes(cb, as.map(a => EmitCode.fromI(cb.emb)(cb => produce(a, cb)))) { childStreams =>

          val producers = childStreams.map(_.asStream.getProducer(mb))

          assert(names.length == producers.length)

          val producer: StreamProducer = behavior match {
            case behavior@(ArrayZipBehavior.TakeMinLength | ArrayZipBehavior.AssumeSameLength) =>
              val vars = names.zip(producers).map { case (name, p) => mb.newEmitField(name, p.element.emitType) }

              val eltRegion = mb.genFieldThisRef[Region]("streamzip_eltregion")
              val bodyCode = EmitCode.fromI(mb)(cb => emit(body, cb, region = eltRegion, env = env.bind(names.zip(vars): _*)))

              new StreamProducer {
                override def method: EmitMethodBuilder[_] = mb
                override val length: Option[EmitCodeBuilder => Code[Int]] = {
                  behavior match {
                    case ArrayZipBehavior.AssumeSameLength =>
                      producers.flatMap(_.length).headOption
                    case ArrayZipBehavior.TakeMinLength =>
                      anyFailAllFail((producers, as).zipped.flatMap { (producer, child) =>
                        child match {
                          case _: StreamIota => None
                          case _ => Some(producer.length)
                        }
                      }).map { compLens =>
                        (cb: EmitCodeBuilder) => {
                          compLens.map(_.apply(cb)).reduce(_.min(_))
                        }
                      }
                  }
                }

                override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
                  producers.foreach { p =>
                    if (p.requiresMemoryManagementPerElement)
                      cb.assign(p.elementRegion, eltRegion)
                    else
                      cb.assign(p.elementRegion, outerRegion)
                    p.initialize(cb, outerRegion)
                  }
                }

                override val elementRegion: Settable[Region] = eltRegion

                override val requiresMemoryManagementPerElement: Boolean = producers.exists(_.requiresMemoryManagementPerElement)

                override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>

                  producers.zipWithIndex.foreach { case (p, i) =>
                    cb.goto(p.LproduceElement)
                    cb.define(p.LproduceElementDone)
                    cb.assign(vars(i), p.element)
                  }

                  cb.goto(LproduceElementDone)

                  // all producer EOS jumps should immediately jump to zipped EOS
                  producers.foreach { p =>
                    cb.define(p.LendOfStream)
                    cb.goto(LendOfStream)
                  }
                }

                val element: EmitCode = bodyCode

                def close(cb: EmitCodeBuilder): Unit = {
                  producers.foreach(_.close(cb))
                }
              }

            case ArrayZipBehavior.AssertSameLength =>

              val vars = names.zip(producers).map { case (name, p) => mb.newEmitField(name, p.element.emitType) }

              val eltRegion = mb.genFieldThisRef[Region]("streamzip_eltregion")
              val bodyCode = EmitCode.fromI(mb)(cb => emit(body, cb, region = eltRegion, env = env.bind(names.zip(vars): _*)))

              val anyEOS = mb.genFieldThisRef[Boolean]("zip_any_eos")
              val allEOS = mb.genFieldThisRef[Boolean]("zip_all_eos")


              new StreamProducer {
                override def method: EmitMethodBuilder[_] = mb
                override val length: Option[EmitCodeBuilder => Code[Int]] = producers.flatMap(_.length) match {
                  case Seq() => None
                  case ls =>
                    val len = mb.genFieldThisRef[Int]("zip_asl_len")
                    val lenTemp = mb.genFieldThisRef[Int]("zip_asl_len_temp")
                    Some({cb: EmitCodeBuilder =>
                      val len = cb.newLocal[Int]("zip_len", ls.head(cb))
                        ls.tail.foreach { compL =>
                          val lenTemp = cb.newLocal[Int]("lenTemp", compL(cb))
                          cb.ifx(len.cne(lenTemp), cb._fatalWithError(errorID, "zip: length mismatch: ", len.toS, ", ", lenTemp.toS))
                        }
                      len
                    })
                }

                override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
                  cb.assign(anyEOS, false)

                  producers.foreach { p =>
                    if (p.requiresMemoryManagementPerElement)
                      cb.assign(p.elementRegion, eltRegion)
                    else
                      cb.assign(p.elementRegion, outerRegion)
                    p.initialize(cb, outerRegion)
                  }
                }

                override val elementRegion: Settable[Region] = eltRegion

                override val requiresMemoryManagementPerElement: Boolean = producers.exists(_.requiresMemoryManagementPerElement)

                override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
                  cb.assign(allEOS, true)

                  producers.zipWithIndex.foreach { case (p, i) =>

                    val fallThrough = CodeLabel()

                    cb.goto(p.LproduceElement)

                    cb.define(p.LendOfStream)
                    cb.assign(anyEOS, true)
                    cb.goto(fallThrough)

                    cb.define(p.LproduceElementDone)
                    cb.assign(vars(i), p.element)
                    cb.assign(allEOS, false)

                    cb.define(fallThrough)
                  }

                  cb.ifx(anyEOS,
                    cb.ifx(allEOS,
                      cb.goto(LendOfStream),
                      cb._fatalWithError(errorID, "zip: length mismatch"))
                  )

                  cb.goto(LproduceElementDone)
                }

                val element: EmitCode = bodyCode

                def close(cb: EmitCodeBuilder): Unit = {
                  producers.foreach(_.close(cb))
                }
              }


            case ArrayZipBehavior.ExtendNA =>
              val vars = names.zip(producers).map { case (name, p) => mb.newEmitField(name, p.element.emitType.copy(required = false)) }

              val eltRegion = mb.genFieldThisRef[Region]("streamzip_eltregion")
              val bodyCode = EmitCode.fromI(mb)(cb => emit(body, cb, region = eltRegion, env = env.bind(names.zip(vars): _*)))

              val eosPerStream = producers.indices.map(i => mb.genFieldThisRef[Boolean](s"zip_eos_$i"))
              val nEOS = mb.genFieldThisRef[Int]("zip_n_eos")

              new StreamProducer {
                override def method: EmitMethodBuilder[_] = mb
                override val length: Option[EmitCodeBuilder => Code[Int]] =
                  anyFailAllFail(producers.map(_.length))
                    .map  { compLens =>
                      (cb: EmitCodeBuilder) => {
                        compLens.map(_.apply(cb)).reduce(_.max(_))
                      }
                    }

                override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
                  producers.foreach { p =>
                    if (p.requiresMemoryManagementPerElement)
                      cb.assign(p.elementRegion, eltRegion)
                    else
                      cb.assign(p.elementRegion, outerRegion)
                    p.initialize(cb, outerRegion)
                  }

                  eosPerStream.foreach { eos =>
                    cb.assign(eos, false)
                  }
                  cb.assign(nEOS, 0)
                }

                override val elementRegion: Settable[Region] = eltRegion

                override val requiresMemoryManagementPerElement: Boolean = producers.exists(_.requiresMemoryManagementPerElement)

                override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>

                  producers.zipWithIndex.foreach { case (p, i) =>

                    // label at the end of processing this element
                    val endProduce = CodeLabel()

                    cb.ifx(eosPerStream(i), cb.goto(endProduce))

                    cb.goto(p.LproduceElement)

                    // after an EOS we set the EOS boolean for that stream, and check if all streams have ended
                    cb.define(p.LendOfStream)
                    cb.assign(nEOS, nEOS + 1)

                    cb.ifx(nEOS.ceq(const(producers.length)), cb.goto(LendOfStream))

                    // this stream has ended before each other, so we set the eos flag and the element EmitSettable
                    cb.assign(eosPerStream(i), true)
                    cb.assign(vars(i), EmitCode.missing(mb, vars(i).st))

                    cb.goto(endProduce)

                    cb.define(p.LproduceElementDone)
                    cb.assign(vars(i), p.element)

                    cb.define(endProduce)
                  }

                  cb.goto(LproduceElementDone)
                }

                val element: EmitCode = bodyCode

                def close(cb: EmitCodeBuilder): Unit = {
                  producers.foreach(_.close(cb))
                }
              }

          }

          SStreamValue(producer)
        }

      case x@StreamZipJoin(as, key, keyRef, valsRef, joinIR) =>
        IEmitCode.multiMapEmitCodes(cb, as.map(a => EmitCode.fromI(cb.emb)(cb => emit(a, cb)))) { children =>
          val producers = children.map(_.asStream.getProducer(mb))

          val eltType = VirtualTypeWithReq.union(as.map(a => typeWithReqx(a))).canonicalEmitType
            .st
            .asInstanceOf[SStream]
            .elementType
            .storageType()
            .setRequired(false)
            .asInstanceOf[PCanonicalStruct]

          val keyType = eltType.selectFields(key)

          val curValsType = PCanonicalArray(eltType)

          val _elementRegion = mb.genFieldThisRef[Region]("szj_region")
          val regionArray = mb.genFieldThisRef[Array[Region]]("szj_region_array")

          val staticMemManagementArray = producers.map(_.requiresMemoryManagementPerElement).toArray
          val allMatch = staticMemManagementArray.toSet.size == 1
          val memoryManagementBooleansArray = if (allMatch) null else mb.genFieldThisRef[Array[Int]]("smm_separate_region_array")

          def initMemoryManagementPerElementArray(cb: EmitCodeBuilder): Unit = {
            if (!allMatch)
              cb.assign(memoryManagementBooleansArray, mb.getObject[Array[Int]](producers.map(_.requiresMemoryManagementPerElement.toInt).toArray))
          }

          def lookupMemoryManagementByIndex(cb: EmitCodeBuilder, idx: Code[Int]): Code[Boolean] = {
            if (allMatch)
              const(staticMemManagementArray.head)
            else
              memoryManagementBooleansArray.apply(idx).toZ
          }

          // The algorithm maintains a tournament tree of comparisons between the
          // current values of the k streams. The tournament tree is a complete
          // binary tree with k leaves. The leaves of the tree are the streams,
          // and each internal node represents the "contest" between the "winners"
          // of the two subtrees, where the winner is the stream with the smaller
          // current key. Each internal node stores the index of the stream which
          // *lost* that contest.
          // Each time we remove the overall winner, and replace that stream's
          // leaf with its next value, we only need to rerun the contests on the
          // path from that leaf to the root, comparing the new value with what
          // previously lost that contest to the previous overall winner.

          val k = producers.length
          // The leaf nodes of the tournament tree, each of which holds a pointer
          // to the current value of that stream.
          val heads = mb.genFieldThisRef[Array[Long]]("merge_heads")
          // The internal nodes of the tournament tree, laid out in breadth-first
          // order, each of which holds the index of the stream which lost that
          // contest.
          val bracket = mb.genFieldThisRef[Array[Int]]("merge_bracket")
          // When updating the tournament tree, holds the winner of the subtree
          // containing the updated leaf. Otherwise, holds the overall winner, i.e.
          // the current least element.
          val winner = mb.genFieldThisRef[Int]("merge_winner")
          val result = mb.genFieldThisRef[Array[Long]]("merge_result")
          val i = mb.genFieldThisRef[Int]("merge_i")

          val curKey = mb.newPField("st_grpby_curkey", keyType.sType)

          val xKey = mb.newEmitField("zipjoin_key", keyType.sType, required = true)
          val xElts = mb.newEmitField("zipjoin_elts", curValsType.sType, required = true)

          val joinResult: EmitCode = EmitCode.fromI(mb) { cb =>
            val newEnv = env.bind((keyRef -> xKey), (valsRef -> xElts))
            emit(joinIR, cb, env = newEnv)
          }

          val producer = new StreamProducer {
            override def method: EmitMethodBuilder[_] = mb
            override val length: Option[EmitCodeBuilder => Code[Int]] = None

            override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
              cb.assign(regionArray, Code.newArray[Region](k))
              producers.zipWithIndex.foreach { case (p, idx) =>
                if (p.requiresMemoryManagementPerElement) {
                  cb.assign(p.elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
                } else
                  cb.assign(p.elementRegion, outerRegion)
                cb += (regionArray(idx) = p.elementRegion)
                p.initialize(cb, outerRegion)
              }
              initMemoryManagementPerElementArray(cb)
              cb.assign(bracket, Code.newArray[Int](k))
              cb.assign(heads, Code.newArray[Long](k))
              cb.forLoop(cb.assign(i, 0), i < k, cb.assign(i, i + 1), {
                cb += (bracket(i) = -1)
              })
              cb.assign(result, Code._null)
              cb.assign(i, 0)
              cb.assign(winner, 0)
            }

            override val elementRegion: Settable[Region] = _elementRegion
            override val requiresMemoryManagementPerElement: Boolean = producers.exists(_.requiresMemoryManagementPerElement)
            override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
              val LrunMatch = CodeLabel()
              val LpullChild = CodeLabel()
              val LloopEnd = CodeLabel()
              val LaddToResult = CodeLabel()
              val LstartNewKey = CodeLabel()
              val Lpush = CodeLabel()

              def inSetup: Code[Boolean] = result.isNull

              cb.ifx(inSetup, {
                cb.assign(i, 0)
                cb.goto(LpullChild)
              }, {
                cb.ifx(winner.ceq(k), cb.goto(LendOfStream), cb.goto(LstartNewKey))
              })

              cb.define(Lpush)
              cb.assign(xKey, EmitCode.present(cb.emb, curKey))
              cb.assign(xElts, EmitCode.present(cb.emb, curValsType.constructFromElements(cb, elementRegion, k, false) { (cb, i) =>
                IEmitCode(cb, result(i).ceq(0L), eltType.loadCheapSCode(cb, result(i)))
              }))
              cb.goto(LproduceElementDone)

              cb.define(LstartNewKey)
              cb.forLoop(cb.assign(i, 0), i < k, cb.assign(i, i + 1), {
                cb += (result(i) = 0L)
              })
              cb.assign(curKey, eltType.loadCheapSCode(cb, heads(winner)).subset(key: _*)
                .castTo(cb, elementRegion, curKey.st, true))
              cb.goto(LaddToResult)

              cb.define(LaddToResult)
              cb += (result(winner) = heads(winner))
              cb.ifx(lookupMemoryManagementByIndex(cb, winner), {
                val r = cb.newLocal[Region]("tzj_winner_region", regionArray(winner))
                cb += elementRegion.trackAndIncrementReferenceCountOf(r)
                cb += r.clearRegion()
              })
              cb.goto(LpullChild)

              val matchIdx = mb.genFieldThisRef[Int]("merge_match_idx")
              val challenger = mb.genFieldThisRef[Int]("merge_challenger")
              // Compare 'winner' with value in 'matchIdx', loser goes in 'matchIdx',
              // winner goes on to next round. A contestant '-1' beats everything
              // (negative infinity), a contestant 'k' loses to everything
              // (positive infinity), and values in between are indices into 'heads'.

              cb.define(LrunMatch)
              cb.assign(challenger, bracket(matchIdx))
              cb.ifx(matchIdx.ceq(0) || challenger.ceq(-1), cb.goto(LloopEnd))

              val LafterChallenge = CodeLabel()

              cb.ifx(challenger.cne(k), {
                val LchallengerWins = CodeLabel()

                cb.ifx(winner.ceq(k), cb.goto(LchallengerWins))

                val left = eltType.loadCheapSCode(cb, heads(challenger)).subset(key: _*)
                val right = eltType.loadCheapSCode(cb, heads(winner)).subset(key: _*)
                val ord = StructOrdering.make(left.st, right.st, cb.emb.ecb, missingFieldsEqual = false)
                cb.ifx(ord.lteqNonnull(cb, left, right),
                  cb.goto(LchallengerWins),
                  cb.goto(LafterChallenge))

                cb.define(LchallengerWins)
                cb += (bracket(matchIdx) = winner)
                cb.assign(winner, challenger)
              })
              cb.define(LafterChallenge)
              cb.assign(matchIdx, matchIdx >>> 1)
              cb.goto(LrunMatch)

              cb.define(LloopEnd)
              cb.ifx(matchIdx.ceq(0), {
                // 'winner' is smallest of all k heads. If 'winner' = k, all heads
                // must be k, and all streams are exhausted.

                cb.ifx(inSetup, {
                  cb.ifx(winner.ceq(k),
                    cb.goto(LendOfStream),
                    {
                      cb.assign(result, Code.newArray[Long](k))
                      cb.goto(LstartNewKey)
                    })
                }, {
                  cb.ifx(!winner.cne(k), cb.goto(Lpush))
                  val left = eltType.loadCheapSCode(cb, heads(winner)).subset(key: _*)
                  val right = curKey
                  val ord = StructOrdering.make(left.st, right.st.asInstanceOf[SBaseStruct],
                    cb.emb.ecb, missingFieldsEqual = false)
                  cb.ifx(ord.equivNonnull(cb, left, right), cb.goto(LaddToResult), cb.goto(Lpush))
                })
              }, {
                // We're still in the setup phase
                cb += (bracket(matchIdx) = winner)
                cb.assign(i, i + 1)
                cb.assign(winner, i)
                cb.goto(LpullChild)
              })

              producers.zipWithIndex.foreach { case (p, idx) =>
                cb.define(p.LendOfStream)
                cb.assign(winner, k)
                cb.assign(matchIdx, (idx + k) >>> 1)
                cb.goto(LrunMatch)

                cb.define(p.LproduceElementDone)
                val storedElt = eltType.store(cb, p.elementRegion, p.element.toI(cb).get(cb), false)
                cb += (heads(idx) = storedElt)
                cb.assign(matchIdx, (idx + k) >>> 1)
                cb.goto(LrunMatch)
              }

              cb.define(LpullChild)
              cb += Code.switch(winner,
                LendOfStream.goto, // can only happen if k=0
                producers.map(_.LproduceElement.goto))
            }

            override val element: EmitCode = joinResult

            override def close(cb: EmitCodeBuilder): Unit = {
              producers.foreach { p =>
                if (p.requiresMemoryManagementPerElement)
                  cb += p.elementRegion.invalidate()
                p.close(cb)
              }
              cb.assign(bracket, Code._null)
              cb.assign(heads, Code._null)
              cb.assign(result, Code._null)
            }
          }

          SStreamValue(producer)
        }

      case x@StreamZipJoinProducers(contexts, ctxName, makeProducer, key, keyRef, valsRef, joinIR) =>
        emit(contexts, cb).map(cb) { case contextsArray: SIndexableValue =>
          val nStreams = cb.memoizeField(contextsArray.loadLength())
          val iterArray = cb.memoizeField(Code.newArray[NoBoxLongIterator](nStreams), "iterArray")
          val idx = cb.newLocal[Int]("i", 0)
          val eltType = VirtualTypeWithReq(TIterable.elementType(makeProducer.typ),
            emitter.ctx.req.lookup(makeProducer).asInstanceOf[RIterable].elementType).canonicalPType
            .asInstanceOf[PCanonicalStruct]
            .setRequired(false)
          var streamRequiresMemoryManagement = false
          cb.whileLoop(idx < nStreams, {
            val iter = produceIterator(makeProducer,
              eltType,
              cb,
              outerRegion,
              env.bind(ctxName, cb.memoize(contextsArray.loadElement(cb, idx))))
              .get(cb, "streams in zipJoinProducers cannot be missing")
              .asInstanceOf[SStreamConcrete]
            streamRequiresMemoryManagement = iter.st.requiresMemoryManagement
            cb += iterArray.update(idx, iter.it)
            cb.assign(idx, idx + 1)
          })

          val keyType = eltType.selectFields(key)

          val curValsType = PCanonicalArray(eltType)

          val _elementRegion = mb.genFieldThisRef[Region]("szj_region")

          // The algorithm maintains a tournament tree of comparisons between the
          // current values of the k streams. The tournament tree is a complete
          // binary tree with k leaves. The leaves of the tree are the streams,
          // and each internal node represents the "contest" between the "winners"
          // of the two subtrees, where the winner is the stream with the smaller
          // current key. Each internal node stores the index of the stream which
          // *lost* that contest.
          // Each time we remove the overall winner, and replace that stream's
          // leaf with its next value, we only need to rerun the contests on the
          // path from that leaf to the root, comparing the new value with what
          // previously lost that contest to the previous overall winner.

          val k = nStreams
          // The leaf nodes of the tournament tree, each of which holds a pointer
          // to the current value of that stream.
          val heads = mb.genFieldThisRef[Array[Long]]("merge_heads")
          // The internal nodes of the tournament tree, laid out in breadth-first
          // order, each of which holds the index of the stream which lost that
          // contest.
          val bracket = mb.genFieldThisRef[Array[Int]]("merge_bracket")
          // When updating the tournament tree, holds the winner of the subtree
          // containing the updated leaf. Otherwise, holds the overall winner, i.e.
          // the current least element.
          val winner = mb.genFieldThisRef[Int]("merge_winner")
          val result = mb.genFieldThisRef[Array[Long]]("merge_result")
          val i = mb.genFieldThisRef[Int]("merge_i")

          val curKey = mb.newPField("st_grpby_curkey", keyType.sType)

          val xKey = mb.newEmitField("zipjoin_key", keyType.sType, required = true)
          val xElts = mb.newEmitField("zipjoin_elts", curValsType.sType, required = true)

          val joinResult: EmitCode = EmitCode.fromI(mb) { cb =>
            val newEnv = env.bind((keyRef -> xKey), (valsRef -> xElts))
            emit(joinIR, cb, env = newEnv)
          }

          val regionArray: Settable[Array[Region]] = if (streamRequiresMemoryManagement)
            mb.genFieldThisRef[Array[Region]]("szj_region_array")
          else
            null

          val producer = new StreamProducer {
            override def method: EmitMethodBuilder[_] = mb

            override val length: Option[EmitCodeBuilder => Code[Int]] = None

            override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
              if (streamRequiresMemoryManagement)
                cb.assign(regionArray, Code.newArray[Region](nStreams))
              cb.assign(bracket, Code.newArray[Int](k))
              cb.assign(heads, Code.newArray[Long](k))
              cb.forLoop(cb.assign(i, 0), i < k, cb.assign(i, i + 1), {
                cb.updateArray(bracket, i, -1)
                val eltRegion: Value[Region] = if (streamRequiresMemoryManagement) {
                  val r = cb.memoize(Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
                  cb.updateArray(regionArray, i, r)
                  r
                } else outerRegion
                cb += iterArray(i).invoke[Region, Region, Unit]("init", outerRegion, eltRegion)
              })
              cb.assign(result, Code._null)
              cb.assign(i, 0)
              cb.assign(winner, 0)
            }

            override val elementRegion: Settable[Region] = _elementRegion
            override val requiresMemoryManagementPerElement: Boolean = streamRequiresMemoryManagement
            override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
              val LrunMatch = CodeLabel()
              val LpullChild = CodeLabel()
              val LloopEnd = CodeLabel()
              val LaddToResult = CodeLabel()
              val LstartNewKey = CodeLabel()
              val Lpush = CodeLabel()

              def inSetup: Code[Boolean] = result.isNull

              cb.ifx(inSetup, {
                cb.assign(i, 0)
                cb.goto(LpullChild)
              }, {
                cb.ifx(winner.ceq(k), cb.goto(LendOfStream), cb.goto(LstartNewKey))
              })

              cb.define(Lpush)
              cb.assign(xKey, EmitCode.present(cb.emb, curKey))
              cb.assign(xElts, EmitCode.present(cb.emb, curValsType.constructFromElements(cb, elementRegion, k, false) { (cb, i) =>
                IEmitCode(cb, result(i).ceq(0L), eltType.loadCheapSCode(cb, result(i)))
              }))
              cb.goto(LproduceElementDone)

              cb.define(LstartNewKey)
              cb.forLoop(cb.assign(i, 0), i < k, cb.assign(i, i + 1), {
                cb.updateArray(result, i, 0L)
              })
              cb.assign(curKey, eltType.loadCheapSCode(cb, heads(winner)).subset(key: _*)
                .castTo(cb, elementRegion, curKey.st, true))
              cb.goto(LaddToResult)

              cb.define(LaddToResult)
              cb += (result(winner) = heads(winner))
              if (streamRequiresMemoryManagement) {
                val r = cb.newLocal[Region]("tzj_winner_region", regionArray(winner))
                cb += elementRegion.trackAndIncrementReferenceCountOf(r)
                cb += r.clearRegion()
              }
              cb.goto(LpullChild)

              val matchIdx = mb.genFieldThisRef[Int]("merge_match_idx")
              val challenger = mb.genFieldThisRef[Int]("merge_challenger")
              // Compare 'winner' with value in 'matchIdx', loser goes in 'matchIdx',
              // winner goes on to next round. A contestant '-1' beats everything
              // (negative infinity), a contestant 'k' loses to everything
              // (positive infinity), and values in between are indices into 'heads'.

              cb.define(LrunMatch)
              cb.assign(challenger, bracket(matchIdx))
              cb.ifx(matchIdx.ceq(0) || challenger.ceq(-1), cb.goto(LloopEnd))

              val LafterChallenge = CodeLabel()

              cb.ifx(challenger.cne(k), {
                val LchallengerWins = CodeLabel()

                cb.ifx(winner.ceq(k), cb.goto(LchallengerWins))

                val left = eltType.loadCheapSCode(cb, heads(challenger)).subset(key: _*)
                val right = eltType.loadCheapSCode(cb, heads(winner)).subset(key: _*)
                val ord = StructOrdering.make(left.st, right.st, cb.emb.ecb, missingFieldsEqual = false)
                cb.ifx(ord.lteqNonnull(cb, left, right),
                  cb.goto(LchallengerWins),
                  cb.goto(LafterChallenge))

                cb.define(LchallengerWins)
                cb.updateArray(bracket, matchIdx, winner)
                cb.assign(winner, challenger)
              })
              cb.define(LafterChallenge)
              cb.assign(matchIdx, matchIdx >>> 1)
              cb.goto(LrunMatch)

              cb.define(LloopEnd)
              cb.ifx(matchIdx.ceq(0), {
                // 'winner' is smallest of all k heads. If 'winner' = k, all heads
                // must be k, and all streams are exhausted.

                cb.ifx(inSetup, {
                  cb.ifx(winner.ceq(k),
                    cb.goto(LendOfStream),
                    {
                      cb.assign(result, Code.newArray[Long](k))
                      cb.goto(LstartNewKey)
                    })
                }, {
                  cb.ifx(!winner.cne(k), cb.goto(Lpush))
                  val left = eltType.loadCheapSCode(cb, heads(winner)).subset(key: _*)
                  val right = curKey
                  val ord = StructOrdering.make(left.st, right.st.asInstanceOf[SBaseStruct],
                    cb.emb.ecb, missingFieldsEqual = false)
                  cb.ifx(ord.equivNonnull(cb, left, right), cb.goto(LaddToResult), cb.goto(Lpush))
                })
              }, {
                // We're still in the setup phase
                cb.updateArray(bracket, matchIdx, winner)
                cb.assign(i, i + 1)
                cb.assign(winner, i)
                cb.goto(LpullChild)
              })

              cb.define(LpullChild)
              cb.ifx(winner >= nStreams, LendOfStream.goto) // can only happen if k=0
              val winnerIter = cb.memoize(iterArray(winner))
              val winnerNextElt = cb.memoize(winnerIter.invoke[Long]("next"))
              cb.ifx(winnerIter.invoke[Boolean]("eos"), {
                cb.assign(matchIdx, (winner + k) >>> 1)
                cb.assign(winner, k)
              }, {
                cb.assign(matchIdx, (winner + k) >>> 1)
                cb.updateArray(heads, winner, winnerNextElt)
              })
              cb.goto(LrunMatch)
            }

            override val element: EmitCode = joinResult

            override def close(cb: EmitCodeBuilder): Unit = {
              cb.assign(i, 0)
              cb.whileLoop(i < nStreams, {
                cb += iterArray(i).invoke[Unit]("close")
                if (requiresMemoryManagementPerElement)
                  cb += regionArray(i).invoke[Unit]("invalidate")
                cb.assign(i, i + 1)
              })
              if (requiresMemoryManagementPerElement)
                cb.assign(regionArray, Code._null)
              cb.assign(bracket, Code._null)
              cb.assign(heads, Code._null)
              cb.assign(result, Code._null)
            }
          }

          SStreamValue(producer)
        }


      case x@StreamMultiMerge(as, key) =>
        IEmitCode.multiMapEmitCodes(cb, as.map(a => EmitCode.fromI(mb)(cb => emit(a, cb)))) { children =>
          val producers = children.map(_.asStream.getProducer(mb))

          val unifiedType = VirtualTypeWithReq.union(as.map(a => typeWithReqx(a))).canonicalEmitType
            .st
            .asInstanceOf[SStream]
            .elementEmitType
            .storageType
            .asInstanceOf[PCanonicalStruct]

          val region = mb.genFieldThisRef[Region]("smm_region")
          val regionArray = mb.genFieldThisRef[Array[Region]]("smm_region_array")

          val staticMemManagementArray = producers.map(_.requiresMemoryManagementPerElement).toArray
          val allMatch = staticMemManagementArray.toSet.size == 1
          val memoryManagementBooleansArray = if (allMatch) null else mb.genFieldThisRef[Array[Int]]("smm_separate_region_array")

          def initMemoryManagementPerElementArray(cb: EmitCodeBuilder): Unit = {
            if (!allMatch)
              cb.assign(memoryManagementBooleansArray, mb.getObject[Array[Int]](producers.map(_.requiresMemoryManagementPerElement.toInt).toArray))
          }

          def lookupMemoryManagementByIndex(cb: EmitCodeBuilder, idx: Code[Int]): Code[Boolean] = {
            if (allMatch)
              const(staticMemManagementArray.head)
            else
              memoryManagementBooleansArray.apply(idx).toZ
          }

          val producer = new StreamUtils.StreamMultiMergeBase(key, unifiedType, const(producers.length), mb) {
            override def method: EmitMethodBuilder[_] = mb

            override val length: Option[EmitCodeBuilder => Code[Int]] =
              anyFailAllFail(producers.map(_.length))
                .map { compLens =>
                  (cb: EmitCodeBuilder) => {
                    compLens.map(_.apply(cb)).reduce(_ + _)
                  }
                }

            override def implInit(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
              cb.assign(regionArray, Code.newArray[Region](k))
              producers.zipWithIndex.foreach { case (p, i) =>
                if (p.requiresMemoryManagementPerElement) {
                  cb.assign(p.elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
                } else
                  cb.assign(p.elementRegion, outerRegion)
                cb += (regionArray(i) = p.elementRegion)
                p.initialize(cb, outerRegion)
              }
              initMemoryManagementPerElementArray(cb)
            }

            override val requiresMemoryManagementPerElement: Boolean = producers.exists(_.requiresMemoryManagementPerElement)
            override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
              val LrunMatch = CodeLabel()
              val LpullChild = CodeLabel()
              val LloopEnd = CodeLabel()

              cb.define(LpullChild)
              // FIXME codebuilderify switch
              cb += Code.switch(winner,
                LendOfStream.goto, // can only happen if k=0
                producers.map(p => p.LproduceElement.goto))


              cb.define(LrunMatch)
              cb.assign(challenger, bracket(matchIdx))
              cb.ifx(matchIdx.ceq(0) || challenger.ceq(-1), cb.goto(LloopEnd))

              val LafterChallenge = CodeLabel()
              cb.ifx(challenger.cne(k), {
                val Lwon = CodeLabel()
                cb.ifx(winner.ceq(k), cb.goto(Lwon))
                cb.ifx(comp(cb, challenger, heads(challenger), winner, heads(winner)), cb.goto(Lwon), cb.goto(LafterChallenge))

                cb.define(Lwon)
                cb += (bracket(matchIdx) = winner)
                cb.assign(winner, challenger)
              })
              cb.define(LafterChallenge)

              cb.assign(matchIdx, matchIdx >>> 1)
              cb.goto(LrunMatch)

              cb.define(LloopEnd)

              cb.ifx(matchIdx.ceq(0), {
                // 'winner' is smallest of all k heads. If 'winner' = k, all heads
                // must be k, and all streams are exhausted.
                cb.ifx(winner.ceq(k),
                  cb.goto(LendOfStream),
                  {
                    // we have a winner
                    cb.ifx(lookupMemoryManagementByIndex(cb, winner), {
                      val winnerRegion = cb.newLocal[Region]("smm_winner_region", regionArray(winner))
                      cb += elementRegion.trackAndIncrementReferenceCountOf(winnerRegion)
                      cb += winnerRegion.clearRegion()
                    })
                    cb.goto(LproduceElementDone)
                  })
              }, {
                cb += (bracket(matchIdx) = winner)
                cb.assign(i, i + 1)
                cb.assign(winner, i)
                cb.goto(LpullChild)
              })

              // define producer labels
              producers.zipWithIndex.foreach { case (p, idx) =>
                cb.define(p.LendOfStream)
                cb.assign(winner, k)
                cb.assign(matchIdx, (const(idx) + k) >>> 1)
                cb.goto(LrunMatch)

                cb.define(p.LproduceElementDone)
                cb += (heads(idx) = unifiedType.store(cb, p.elementRegion, p.element.toI(cb).get(cb), false))
                cb.assign(matchIdx, (const(idx) + k) >>> 1)
                cb.goto(LrunMatch)
              }
            }

            override def implClose(cb: EmitCodeBuilder): Unit = {
              producers.foreach { p =>
                if (p.requiresMemoryManagementPerElement)
                  cb += p.elementRegion.invalidate()
                p.close(cb)
              }
            }
          }
          SStreamValue(producer)
        }

      case StreamLocalLDPrune(a, r2Threshold, winSize, maxQueueSize, nSamples) =>
        produce(a, cb)
          .map(cb) { case childStream: SStreamValue =>
            val childProducer = childStream.getProducer(mb)
            val queueSize = cb.newField[Int]("max_queue_size")
            val queue = cb.newField[util.ArrayDeque[BitPackedVector]]("queue")
            val threshold = cb.newField[Double]("r2_threshold")
            val windowSize = cb.newField[Int]("window_size")
            val builder = cb.newField[BitPackedVectorBuilder]("vector_builder")
            val elementType = typeWithReq.canonicalEmitType.st.asInstanceOf[SStream].elementEmitType
            val elementField = mb.newEmitField(elementType)

            val producer = new StreamProducer {
              def method: EmitMethodBuilder[_] = mb

              val length: Option[EmitCodeBuilder => Code[Int]] = None

              val elementRegion: Settable[Region] = childProducer.elementRegion

              val element: EmitCode = elementField

              val requiresMemoryManagementPerElement: Boolean = childProducer.requiresMemoryManagementPerElement

              def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
                cb.assign(queueSize, emit(maxQueueSize, cb).get(cb).asInt32.value)
                cb.assign(queue, Code.newInstance[util.ArrayDeque[BitPackedVector], Int](queueSize))
                cb.assign(threshold, emit(r2Threshold, cb).get(cb).asFloat64.value)
                cb.assign(windowSize, emit(winSize, cb).get(cb).asInt32.value)
                cb.assign(builder, Code.newInstance[BitPackedVectorBuilder, Int](emit(nSamples, cb).get(cb).asInt32.value))
                childProducer.initialize(cb, outerRegion)
              }

              val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
                val Lpruned = CodeLabel()

                cb.goto(childProducer.LproduceElement)
                cb.define(childProducer.LproduceElementDone)

                childProducer.element.toI(cb).consume(cb,
                  cb.goto(Lpruned),
                  { case sc: SBaseStructValue =>
                    val locus = sc.loadField(cb, "locus").get(cb).asLocus
                    val locusObj = locus.getLocusObj(cb)
                    val genotypes = sc.loadField(cb, "genotypes").get(cb).asIndexable
                    cb += builder.invoke[Unit]("reset")
                    genotypes.forEachDefinedOrMissing(cb)({ (cb, _) =>
                      cb += builder.invoke[Unit]("addMissing")
                    }, { (cb, _, gt) =>
                      cb += builder.invoke[Int, Unit]("addGT", gt.asCall.canonicalCall(cb))
                    })
                    val bpv = cb.memoize(builder.invoke[Locus, Array[String], BitPackedVector]("finish", locusObj, Code._null[Array[String]]))
                    cb.ifx(bpv.isNull, cb.goto(Lpruned))
                    val keepVariant = Code.invokeScalaObject5[util.ArrayDeque[BitPackedVector], BitPackedVector, Double, Int, Int, Boolean](LocalLDPrune.getClass, "pruneLocal",
                      queue, bpv, threshold, windowSize, queueSize)
                    cb.ifx(!keepVariant, cb.goto(Lpruned))

                    val mean = SFloat64Value(cb.memoize(bpv.invoke[Double]("mean")))
                    val centeredLengthRec = SFloat64Value(cb.memoize(bpv.invoke[Double]("centeredLengthRec")))
                    val elt = SStackStruct.constructFromArgs(cb, elementRegion, elementType.virtualType.asInstanceOf[TBaseStruct],
                      EmitCode.present(mb, locus), EmitCode.fromI(mb)(cb => sc.loadField(cb, "alleles")), EmitCode.present(mb, mean), EmitCode.present(mb, centeredLengthRec))
                    cb.assign(elementField, EmitCode.present(mb, elt.castTo(cb, elementRegion, elementField.emitType.st)))
                  })

                cb.goto(LproduceElementDone)

                cb.define(Lpruned)
                if (requiresMemoryManagementPerElement)
                  cb += childProducer.elementRegion.clearRegion()
                cb.goto(childProducer.LproduceElement)
             }

              def close(cb: EmitCodeBuilder): Unit = childProducer.close(cb)
            }

            mb.implementLabel(childProducer.LendOfStream) { cb =>
              cb.goto(producer.LendOfStream)
            }
            SStreamValue(producer)
          }

      case ReadPartition(context, rowType, reader) =>
        val ctxCode = EmitCode.fromI(cb.emb)(cb => emit(context, cb))
        reader.emitStream(emitter.ctx.executeContext, cb, mb, ctxCode, rowType)
    }
  }
}
