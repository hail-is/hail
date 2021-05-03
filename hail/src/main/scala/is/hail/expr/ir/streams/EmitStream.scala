package is.hail.expr.ir.streams

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.ir.orderings.StructOrdering
import is.hail.services.shuffler.{CodeShuffleClient, ShuffleClient, ValueShuffleClient}
import is.hail.types.physical.stypes.concrete.{SBinaryPointer, SBinaryPointerSettable, SCanonicalShufflePointerCode, SCanonicalShufflePointerSettable}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives.{SInt32, SInt32Code}
import is.hail.types.physical.{PCanonicalStream, PCode, PInterval, PStruct, PType}
import is.hail.types.virtual.{TInterval, TShuffle, TStream}
import is.hail.utils._


object StreamProducer {
  def defineUnusedLabels(producer: StreamProducer, mb: EmitMethodBuilder[_]): Unit = {
    (producer.LendOfStream.isImplemented, producer.LproduceElementDone.isImplemented) match {
      case (true, true) =>
      case (false, false) =>

        EmitCodeBuilder.scopedVoid(mb) { cb =>
          cb.define(producer.LendOfStream)
          cb.define(producer.LproduceElementDone)
          cb._fatal("unreachable")
        }

      case (eos, ped) => throw new RuntimeException(s"unrealizable value unused asymmetrically: eos=$eos, ped=$ped")
    }
    producer.element.pv match {
      case SStreamCode(_, nested) => defineUnusedLabels(nested, mb)
      case _ =>
    }
  }
}

abstract class StreamProducer {
  /**
    * Stream length, which is present if it can be computed (somewhat) cheaply without
    * consuming the stream.
    *
    * In order for `length` to be valid, the stream must have been initialized with `initialize`.
    */
  val length: Option[Code[Int]]

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
  def initialize(cb: EmitCodeBuilder): Unit

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

  final def unmanagedConsume(cb: EmitCodeBuilder, setup: EmitCodeBuilder => Unit = _ => ())(perElement: EmitCodeBuilder => Unit): Unit = {

    this.initialize(cb)
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

      unmanagedConsume(cb, setup) { cb =>
        perElement(cb)
        cb += elementRegion.clearRegion()
      }
      cb += elementRegion.invalidate()
    } else {
      cb.assign(elementRegion, outerRegion)
      unmanagedConsume(cb, setup)(perElement)
    }
  }
}

object EmitStream {
  private[ir] def produce(
    emitter: Emit[_],
    streamIR: IR,
    cb: EmitCodeBuilder,
    outerRegion: Value[Region],
    env: Emit.E,
    container: Option[AggContainer]
  ): IEmitCode = {

    val mb = cb.emb


    def emitVoid(ir: IR, cb: EmitCodeBuilder, region: Value[Region] = outerRegion, env: Emit.E = env, container: Option[AggContainer] = container): Unit =
      emitter.emitVoid(cb, ir, region, env, container, None)

    def emit(ir: IR, cb: EmitCodeBuilder, region: Value[Region] = outerRegion, env: Emit.E = env, container: Option[AggContainer] = container): IEmitCode = {
      ir.typ match {
        case _: TStream => produce(ir, cb, region, env, container)
        case _ => emitter.emitI(ir, cb, region, env, container, None)
      }
    }

    def produce(streamIR: IR, cb: EmitCodeBuilder, region: Value[Region] = outerRegion, env: Emit.E = env, container: Option[AggContainer] = container): IEmitCode =
      EmitStream.produce(emitter, streamIR, cb, region, env, container)

    streamIR match {

      case NA(_typ) =>
        val eltType = streamIR.pType.asInstanceOf[PCanonicalStream].elementType
        val st = SStream(eltType.sType, false)
        val region = mb.genFieldThisRef[Region]("na_region")
        val producer = new StreamProducer {
          override def initialize(cb: EmitCodeBuilder): Unit = {}

          override val length: Option[Code[Int]] = Some(Code._fatal[Int]("tried to get NA stream length"))
          override val elementRegion: Settable[Region] = region
          override val requiresMemoryManagementPerElement: Boolean = false
          override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
            cb.goto(LendOfStream)
          }
          override val element: EmitCode = EmitCode.missing(mb, eltType)

          override def close(cb: EmitCodeBuilder): Unit = {}
        }
        IEmitCode.missing(cb, SStreamCode(st, producer))

      case Ref(name, _typ) =>
        assert(_typ.isInstanceOf[TStream])
        env.lookup(name).toI(cb)
          .map(cb) { case (stream: SStreamCode) =>
            val childProducer = stream.producer
            val producer = new StreamProducer {
              override def initialize(cb: EmitCodeBuilder): Unit = childProducer.initialize(cb)

              override val length: Option[Code[Int]] = childProducer.length
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
            stream.copy(producer = producer)
          }

      case Let(name, value, body) =>
        cb.withScopedMaybeStreamValue(EmitCode.fromI(mb)(cb => emit(value, cb)), s"let_$name") { ev =>
          produce(body, cb, env = env.bind(name, ev))
        }

      case In(n, _) =>
        // this, Code[Region], ...
        val param = mb.getEmitParam(2 + n, outerRegion)
        if (!param.st.isInstanceOf[SStream])
          throw new RuntimeException(s"parameter ${ 2 + n } is not a stream! t=${ param.st } }, params=${ mb.emitParamTypes }")
        param.load.toI(cb)

      case ToStream(a, _requiresMemoryManagementPerElement) =>

        emit(a, cb).map(cb) { case ind: SIndexableCode =>
          val containerField = mb.newPField("tostream_arr", ind.pt)
          val container = containerField.asInstanceOf[SIndexableValue]
          val idx = mb.genFieldThisRef[Int]("tostream_idx")
          val regionVar = mb.genFieldThisRef[Region]("tostream_region")

          SStreamCode(
            SStream(ind.st.elementType, ind.pt.required),
            new StreamProducer {
              override def initialize(cb: EmitCodeBuilder): Unit = {
                cb.assign(containerField, ind)
                cb.assign(idx, -1)
              }

              override val length: Option[Code[Int]] = Some(container.loadLength())

              override val elementRegion: Settable[Region] = regionVar

              override val requiresMemoryManagementPerElement: Boolean = _requiresMemoryManagementPerElement

              override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
                cb.assign(idx, idx + 1)
                cb.ifx(idx >= container.loadLength(), cb.goto(LendOfStream))
                cb.goto(LproduceElementDone)
              }

              val element: EmitCode = EmitCode.fromI(mb) { cb =>
                container.loadElement(cb, idx).typecast[PCode] }

              def close(cb: EmitCodeBuilder): Unit = {}
            })

        }

      case x@MakeStream(args, _, _requiresMemoryManagementPerElement) =>
        val region = mb.genFieldThisRef[Region]("makestream_region")
        val emittedArgs = args.map(a => EmitCode.fromI(mb)(cb => emit(a, cb, region))).toFastIndexedSeq

        val unifiedType = x.pType.asInstanceOf[PCanonicalStream].elementType.sType // FIXME
        val eltField = mb.newEmitField("makestream_elt", unifiedType.pType)

        val staticLen = args.size
        val current = mb.genFieldThisRef[Int]("makestream_current")

        IEmitCode.present(cb, SStreamCode(
          SStream(unifiedType, required = true),
          new StreamProducer {
            override def initialize(cb: EmitCodeBuilder): Unit = {
              cb.assign(current, 0) // switches on 1..N
            }

            override val length: Option[Code[Int]] = Some(staticLen)

            override val elementRegion: Settable[Region] = region

            override val requiresMemoryManagementPerElement: Boolean = _requiresMemoryManagementPerElement

            override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
              val LendOfSwitch = CodeLabel()
              cb += Code.switch(current,
                EmitCodeBuilder.scopedVoid(mb) { cb =>
                  cb.goto(LendOfStream)
                },
                emittedArgs.map { elem =>
                  EmitCodeBuilder.scopedVoid(mb) { cb =>
                    cb.assign(eltField, elem.toI(cb).map(cb)(pc => pc.castTo(cb, region, unifiedType.pType, false)))
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
          cb.assign(xCond, cond.asBoolean.boolCode(cb))

          val Lmissing = CodeLabel()
          val Lpresent = CodeLabel()

          val leftEC = EmitCode.fromI(mb)(cb => produce(cnsq, cb))
          val rightEC = EmitCode.fromI(mb)(cb => produce(altr, cb))

          val leftProducer = leftEC.pv.asStream.producer
          val rightProducer = rightEC.pv.asStream.producer

          val xElt = mb.newEmitField(x.pType.asInstanceOf[PCanonicalStream].elementType) // FIXME unify here

          val region = mb.genFieldThisRef[Region]("streamif_region")
          cb.ifx(xCond,
            leftEC.toI(cb).consume(cb, cb.goto(Lmissing), _ => cb.goto(Lpresent)),
            rightEC.toI(cb).consume(cb, cb.goto(Lmissing), _ => cb.goto(Lpresent)))

          val producer = new StreamProducer {
            override val length: Option[Code[Int]] = leftProducer.length
              .liftedZip(rightProducer.length).map { case (l1, l2) =>
              xCond.mux(l1, l2)
            }

            override def initialize(cb: EmitCodeBuilder): Unit = {
              cb.ifx(xCond, {
                cb.assign(leftProducer.elementRegion, region)
                leftProducer.initialize(cb)
              }, {
                cb.assign(rightProducer.elementRegion, region)
                rightProducer.initialize(cb)
              })
            }

            override val elementRegion: Settable[Region] = region
            override val requiresMemoryManagementPerElement: Boolean = leftProducer.requiresMemoryManagementPerElement || rightProducer.requiresMemoryManagementPerElement
            override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
              cb.ifx(xCond, cb.goto(leftProducer.LproduceElement), cb.goto(rightProducer.LproduceElement))

              cb.define(leftProducer.LproduceElementDone)
              cb.assign(xElt, leftProducer.element.toI(cb).map(cb)(_.castTo(cb, region, xElt.pt)))
              cb.goto(LproduceElementDone)

              cb.define(rightProducer.LproduceElementDone)
              cb.assign(xElt, rightProducer.element.toI(cb).map(cb)(_.castTo(cb, region, xElt.pt)))
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
            SStreamCode(SStream(xElt.st, required = leftEC.pt.required && rightEC.pt.required), producer),
            leftEC.required && rightEC.required)
        }

      case StreamRange(startIR, stopIR, stepIR, _requiresMemoryManagementPerElement) =>

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


              val producer: StreamProducer = new StreamProducer {
                override val length: Option[Code[Int]] = Some(len)

                override def initialize(cb: EmitCodeBuilder): Unit = {
                  val llen = cb.newLocal[Long]("streamrange_llen")

                  cb.assign(start, startc.asInt.intCode(cb))
                  cb.assign(stop, stopc.asInt.intCode(cb))
                  cb.assign(step, stepc.asInt.intCode(cb))

                  cb.ifx(step ceq const(0), cb._fatal("Array range cannot have step size 0."))
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
                    cb._fatal("Array range cannot have more than MAXINT elements.")
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

                val element: EmitCode = EmitCode.present(mb, new SInt32Code(true, curr))

                def close(cb: EmitCodeBuilder): Unit = {}
              }
              SStreamCode(
                SStream(SInt32(true), required = true),
                producer
              )
            }
          }
        }

      case StreamFilter(a, name, cond) =>
        produce(a, cb)
          .map(cb) { case (childStream: SStreamCode) =>
            val childProducer = childStream.producer

            val filterEltRegion = mb.genFieldThisRef[Region]("streamfilter_filter_region")

            val elementField = cb.emb.newEmitField("streamfilter_cond", childStream.st.elementType.pType)

            val producer = new StreamProducer {
              override val length: Option[Code[Int]] = None

              override def initialize(cb: EmitCodeBuilder): Unit = {
                if (childProducer.requiresMemoryManagementPerElement)
                  cb.assign(childProducer.elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
                else
                  cb.assign(childProducer.elementRegion, outerRegion)
                childProducer.initialize(cb)
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
                      cb.ifx(!sc.asBoolean.boolCode(cb), cb.goto(Lfiltered))
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

            SStreamCode(
              childStream.st,
              producer)
          }

      case StreamTake(a, num) =>
        produce(a, cb)
          .flatMap(cb) { case (childStream: SStreamCode) =>
            emit(num, cb).map(cb) { case (num: SInt32Code) =>
              val childProducer = childStream.producer
              val n = mb.genFieldThisRef[Int]("stream_take_n")
              val idx = mb.genFieldThisRef[Int]("stream_take_idx")

              val producer = new StreamProducer {
                override val length: Option[Code[Int]] = childProducer.length.map(_.min(n))

                override def initialize(cb: EmitCodeBuilder): Unit = {
                  cb.assign(n, num.intCode(cb))
                  cb.ifx(n < 0, cb._fatal(s"stream take: negative number of elements to take: ", n.toS))
                  cb.assign(idx, 0)
                  childProducer.initialize(cb)
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

              SStreamCode(childStream.st, producer)
            }
          }

      case StreamDrop(a, num) =>
        produce(a, cb)
          .flatMap(cb) { case (childStream: SStreamCode) =>
            emit(num, cb).map(cb) { case (num: SInt32Code) =>
              val childProducer = childStream.producer
              val n = mb.genFieldThisRef[Int]("stream_drop_n")
              val idx = mb.genFieldThisRef[Int]("stream_drop_idx")

              val producer = new StreamProducer {
                override val length: Option[Code[Int]] = childProducer.length.map(l => (l - n).max(0))

                override def initialize(cb: EmitCodeBuilder): Unit = {
                  cb.assign(n, num.intCode(cb))
                  cb.ifx(n < 0, cb._fatal(s"stream drop: negative number of elements to drop: ", n.toS))
                  cb.assign(idx, 0)
                  childProducer.initialize(cb)
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

              SStreamCode(childStream.st, producer)
            }
          }

      case StreamMap(a, name, body) =>
        produce(a, cb)
          .map(cb) { case (childStream: SStreamCode) =>
            val childProducer = childStream.producer

            val bodyResult = EmitCode.fromI(mb) { cb =>
              cb.withScopedMaybeStreamValue(childProducer.element, "streammap_element") { childProducerElement =>
                emit(body,
                  cb = cb,
                  env = env.bind(name, childProducerElement),
                  region = childProducer.elementRegion)
              }
            }

            val producer: StreamProducer = new StreamProducer {
              override val length: Option[Code[Int]] = childProducer.length

              override def initialize(cb: EmitCodeBuilder): Unit = {
                childProducer.initialize(cb)
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

            SStreamCode(
              SStream(bodyResult.st, required = childStream.st.required),
              producer
            )
          }

      case x@StreamScan(childIR, zeroIR, accName, eltName, bodyIR) =>
        produce(childIR, cb).map(cb) { case (childStream: SStreamCode) =>
          val childProducer = childStream.producer

          val accValueAccRegion = mb.newEmitField(x.accPType)
          val accValueEltRegion = mb.newEmitField(x.accPType)

          // accRegion is unused if requiresMemoryManagementPerElement is false
          val accRegion: Settable[Region] = if (childProducer.requiresMemoryManagementPerElement) mb.genFieldThisRef[Region]("streamscan_acc_region") else null
          val first = mb.genFieldThisRef[Boolean]("streamscan_first")

          val producer = new StreamProducer {
            override val length: Option[Code[Int]] = childProducer.length.map(_ + const(1))

            override def initialize(cb: EmitCodeBuilder): Unit = {

              if (childProducer.requiresMemoryManagementPerElement) {
                cb.assign(accRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
              }
              cb.assign(first, true)
              childProducer.initialize(cb)
            }

            override val elementRegion: Settable[Region] = childProducer.elementRegion

            override val requiresMemoryManagementPerElement: Boolean = childProducer.requiresMemoryManagementPerElement

            override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>

              val LcopyAndReturn = CodeLabel()

              cb.ifx(first, {

                cb.assign(first, false)
                cb.assign(accValueEltRegion, emit(zeroIR, cb, region = elementRegion))

                cb.goto(LcopyAndReturn)
              })


              cb.goto(childProducer.LproduceElement)
              cb.define(childProducer.LproduceElementDone)

              if (requiresMemoryManagementPerElement) {
                // deep copy accumulator into element region, then clear accumulator region
                cb.assign(accValueEltRegion, accValueAccRegion.toI(cb).map(cb)(_.castTo(cb, childProducer.elementRegion, x.accPType, deepCopy = true)))
                cb += accRegion.clearRegion()
              }

              val bodyCode = cb.withScopedMaybeStreamValue(childProducer.element, "scan_child_elt") { ev =>
                emit(bodyIR, cb, env = env.bind((accName, accValueEltRegion), (eltName, ev)), region = childProducer.elementRegion)
                  .map(cb)(pc => pc.castTo(cb, childProducer.elementRegion, x.accPType, deepCopy = false))
              }

              cb.assign(accValueEltRegion, bodyCode)

              cb.define(LcopyAndReturn)

              if (requiresMemoryManagementPerElement) {
                cb.assign(accValueAccRegion, accValueEltRegion.toI(cb).map(cb)(pc => pc.castTo(cb, accRegion, x.accPType, deepCopy = true)))
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

          SStreamCode(SStream(accValueEltRegion.st, childStream.st.required), producer)
        }

      case RunAggScan(child, name, init, seqs, result, states) =>
        val (newContainer, aggSetup, aggCleanup) = AggContainer.fromMethodBuilder(states.toArray, mb, "run_agg_scan")

        produce(child, cb).map(cb) { case (childStream: SStreamCode) =>
          val childProducer = childStream.producer

          val childEltField = mb.newEmitField("runaggscan_child_elt", childProducer.element.pt)
          val bodyEnv = env.bind(name -> childEltField)
          val bodyResult = EmitCode.fromI(mb)(cb => emit(result, cb = cb, region = childProducer.elementRegion,
            env = bodyEnv, container = Some(newContainer)))
          val bodyResultField = mb.newEmitField("runaggscan_result_elt", bodyResult.pt)

          val producer = new StreamProducer {
            override val length: Option[Code[Int]] = childProducer.length

            override def initialize(cb: EmitCodeBuilder): Unit = {
              aggSetup(cb)
              emitVoid(init, cb = cb, region = outerRegion, container = Some(newContainer))
              childProducer.initialize(cb)
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

          SStreamCode(SStream(producer.element.st, childStream.st.required), producer)
        }

      case StreamFlatMap(a, name, body) =>
        produce(a, cb).map(cb) { case (outerStream: SStreamCode) =>
          val outerProducer = outerStream.producer

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
          val SStreamCode(_, innerProducer) = innerStreamEmitCode.pv

          val producer = new StreamProducer {
            override val length: Option[Code[Int]] = None

            override def initialize(cb: EmitCodeBuilder): Unit = {
              cb.assign(first, true)
              cb.assign(innerUnclosed, false)

              if (outerProducer.requiresMemoryManagementPerElement)
                cb.assign(outerProducer.elementRegion, Region.stagedCreate(Region.REGULAR, cb.emb.ecb.pool()))
              else
                cb.assign(outerProducer.elementRegion, outerRegion)

              outerProducer.initialize(cb)
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

                      innerProducer.initialize(cb)
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

          SStreamCode(
            SStream(innerProducer.element.st, required = outerStream.st.required),
            producer
          )
        }

      case x@StreamJoinRightDistinct(leftIR, rightIR, lKey, rKey, leftName, rightName, joinIR, joinType) =>
        produce(leftIR, cb).flatMap(cb) { case (leftStream: SStreamCode) =>
          produce(rightIR, cb).map(cb) { case (rightStream: SStreamCode) =>

            val leftProducer = leftStream.producer
            val rightProducer = rightStream.producer

            val lEltType = leftProducer.element.pt
            val rEltType = rightProducer.element.pt

            // these variables are used as inputs to the joinF

            def compare(cb: EmitCodeBuilder, lelt: EmitValue, relt: EmitValue): Code[Int] = {
              assert(lelt.pt == lEltType)
              assert(relt.pt == rEltType)

              val lhs = EmitCode.fromI(mb)(cb => lelt.toI(cb).map(cb)(_.asBaseStruct.subset(lKey: _*).asPCode))
              val rhs = EmitCode.fromI(mb)(cb => relt.toI(cb).map(cb)(_.asBaseStruct.subset(rKey: _*).asPCode))
              StructOrdering.make(lhs.st.asInstanceOf[SBaseStruct], rhs.st.asInstanceOf[SBaseStruct],
                cb.emb.ecb, missingFieldsEqual = false)
                .compare(cb, lhs, rhs, missingEqual = false)
            }

            joinType match {
              case "left" =>
                val lx = mb.newEmitField(lEltType) // last value received from left
                val rx = mb.newEmitField(rEltType) // last value received from right
                val rxOut = mb.newEmitField(rEltType.setRequired(false)) // right value in joinF (may be missing while rx is not)

                val joinResult = EmitCode.fromI(mb)(cb => emit(joinIR, cb,
                  region = leftProducer.elementRegion,
                  env = env.bind(leftName -> lx, rightName -> rxOut)))

                val rightEOS = mb.genFieldThisRef[Boolean]("left_join_right_distinct_rightEOS")
                val pulledRight = mb.genFieldThisRef[Boolean]("left_join_right_distinct_pulledRight]")
                val _elementRegion = mb.genFieldThisRef[Region]("join_right_distinct_element_region")

                val producer = new StreamProducer {
                  override val length: Option[Code[Int]] = leftProducer.length

                  override def initialize(cb: EmitCodeBuilder): Unit = {
                    cb.assign(rightEOS, false)
                    cb.assign(pulledRight, false)

                    if (rightProducer.requiresMemoryManagementPerElement)
                      cb.assign(rightProducer.elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
                    else
                      cb.assign(rightProducer.elementRegion, outerRegion)
                    if (leftProducer.requiresMemoryManagementPerElement)
                      cb.assign(leftProducer.elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
                    else
                      cb.assign(leftProducer.elementRegion, outerRegion)

                    leftProducer.initialize(cb)
                    rightProducer.initialize(cb)
                  }

                  override val elementRegion: Settable[Region] = _elementRegion
                  override val requiresMemoryManagementPerElement: Boolean = leftProducer.requiresMemoryManagementPerElement || rightProducer.requiresMemoryManagementPerElement
                  override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>

                    if (leftProducer.requiresMemoryManagementPerElement)
                      cb += leftProducer.elementRegion.clearRegion()
                    cb.goto(leftProducer.LproduceElement)
                    cb.define(leftProducer.LproduceElementDone)
                    cb.assign(lx, leftProducer.element)

                    // if right stream is exhausted, return immediately
                    cb.ifx(rightEOS, cb.goto(LproduceElementDone))

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
                      cb.assign(rxOut, EmitCode.missing(mb, rxOut.pt))
                    }, {
                      // c == 0
                      if (rightProducer.requiresMemoryManagementPerElement) {
                        cb += elementRegion.trackAndIncrementReferenceCountOf(rightProducer.elementRegion)
                      }
                      cb.assign(rxOut, rx)
                    })

                    cb.goto(LproduceElementDone)

                    cb.define(LpullRight)
                    if (rightProducer.requiresMemoryManagementPerElement) {
                      cb += rightProducer.elementRegion.clearRegion()
                    }

                    cb.goto(rightProducer.LproduceElement)
                    cb.define(rightProducer.LproduceElementDone)
                    cb.assign(rx, rightProducer.element)
                    cb.goto(Lcompare)

                    // if right stream ends before left
                    cb.define(rightProducer.LendOfStream)
                    cb.assign(rxOut, EmitCode.missing(mb, rxOut.pt))
                    cb.assign(rightEOS, true)

                    if (leftProducer.requiresMemoryManagementPerElement)
                      cb += elementRegion.trackAndIncrementReferenceCountOf(leftProducer.elementRegion)
                    cb.goto(LproduceElementDone)
                  }
                  override val element: EmitCode = joinResult

                  override def close(cb: EmitCodeBuilder): Unit = {
                    leftProducer.close(cb)
                    rightProducer.close(cb)
                    if (rightProducer.requiresMemoryManagementPerElement)
                      cb += rightProducer.elementRegion.invalidate()
                  }
                }

                mb.implementLabel(leftProducer.LendOfStream) { cb =>
                  cb.goto(producer.LendOfStream)
                }


                SStreamCode(SStream(producer.element.st, leftStream.st.required && rightStream.st.required), producer)

              case "outer" =>

                val lx = mb.newEmitField(lEltType) // last value received from left
                val rx = mb.newEmitField(rEltType) // last value received from right
                val lxOut = mb.newEmitField(lEltType.setRequired(false)) // left value in joinF (may be missing while lx is not)
                val rxOut = mb.newEmitField(rEltType.setRequired(false)) // right value in joinF (may be missing while rx is not)

                val pulledRight = mb.genFieldThisRef[Boolean]("join_right_distinct_pulledRight")
                val rightEOS = mb.genFieldThisRef[Boolean]("join_right_distinct_rightEOS")
                val lOutMissing = mb.genFieldThisRef[Boolean]("join_right_distinct_leftMissing")
                val rOutMissing = mb.genFieldThisRef[Boolean]("join_right_distinct_rightMissing")
                val leftEOS = mb.genFieldThisRef[Boolean]("join_right_distinct_leftEOS")
                val c = mb.genFieldThisRef[Int]("join_right_distinct_compResult")
                val _elementRegion = mb.genFieldThisRef[Region]("join_right_distinct_element_region")

                val joinResult = EmitCode.fromI(mb)(cb => emit(joinIR, cb,
                  region = _elementRegion,
                  env = env.bind(leftName -> lxOut, rightName -> rxOut)))

                val producer = new StreamProducer {
                  override val length: Option[Code[Int]] = None

                  override def initialize(cb: EmitCodeBuilder): Unit = {
                    cb.assign(pulledRight, false)
                    cb.assign(leftEOS, false)
                    cb.assign(rightEOS, false)
                    cb.assign(c, 0) // lets us start stream with a pull from both

                    if (rightProducer.requiresMemoryManagementPerElement)
                      cb.assign(rightProducer.elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
                    else
                      cb.assign(rightProducer.elementRegion, outerRegion)
                    if (leftProducer.requiresMemoryManagementPerElement)
                      cb.assign(leftProducer.elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
                    else
                      cb.assign(leftProducer.elementRegion, outerRegion)

                    leftProducer.initialize(cb)
                    rightProducer.initialize(cb)
                  }

                  override val elementRegion: Settable[Region] = _elementRegion
                  override val requiresMemoryManagementPerElement: Boolean = leftProducer.requiresMemoryManagementPerElement || rightProducer.requiresMemoryManagementPerElement
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
                    cb.goto(rightProducer.LproduceElement)

                    cb.define(LpullLeft)
                    cb.goto(leftProducer.LproduceElement)

                    val Lcompare = CodeLabel()
                    mb.implementLabel(Lcompare) { cb =>
                      cb.assign(c, compare(cb, lx, rx))
                      cb.assign(lOutMissing, false)
                      cb.assign(rOutMissing, false)
                      cb.ifx(c > 0,
                        {
                          cb.ifx(pulledRight, {
                            cb.assign(lOutMissing, true)
                            if (rightProducer.requiresMemoryManagementPerElement) {
                              cb += elementRegion.trackAndIncrementReferenceCountOf(rightProducer.elementRegion)
                              cb += rightProducer.elementRegion.clearRegion()
                            }
                            cb.goto(Lpush)
                          },
                            {
                              cb.assign(pulledRight, true)
                              cb.goto(LpullRight)
                            })
                        },
                        {
                          cb.ifx(c < 0,
                            {
                              cb.assign(rOutMissing, true)
                              if (leftProducer.requiresMemoryManagementPerElement) {
                                cb += elementRegion.trackAndIncrementReferenceCountOf(leftProducer.elementRegion)
                                cb += leftProducer.elementRegion.clearRegion()
                              }
                              cb.goto(Lpush)
                            },
                            {
                              // c == 0
                              if (leftProducer.requiresMemoryManagementPerElement) {
                                cb += elementRegion.trackAndIncrementReferenceCountOf(leftProducer.elementRegion)
                                cb += leftProducer.elementRegion.clearRegion()
                              }
                              if (rightProducer.requiresMemoryManagementPerElement) {
                                cb += elementRegion.trackAndIncrementReferenceCountOf(rightProducer.elementRegion)
                                cb += rightProducer.elementRegion.clearRegion()
                              }
                              cb.goto(Lpush)
                            })
                        })
                    }

                    mb.implementLabel(Lpush) { cb =>
                      cb.ifx(lOutMissing,
                        cb.assign(lxOut, EmitCode.missing(mb, lxOut.pt)),
                        cb.assign(lxOut, lx)
                      )
                      cb.ifx(rOutMissing,
                        cb.assign(rxOut, EmitCode.missing(mb, rxOut.pt)),
                        cb.assign(rxOut, rx))
                      cb.goto(LproduceElementDone)
                    }


                    mb.implementLabel(rightProducer.LproduceElementDone) { cb =>
                      cb.assign(rx, rightProducer.element)
                      cb.ifx(leftEOS, cb.goto(Lpush), cb.goto(Lcompare))
                    }

                    mb.implementLabel(leftProducer.LproduceElementDone) { cb =>
                      cb.assign(lx, leftProducer.element)
                      cb.ifx(pulledRight,
                        cb.ifx(rightEOS,
                          {
                            if (leftProducer.requiresMemoryManagementPerElement) {
                              cb += elementRegion.trackAndIncrementReferenceCountOf(leftProducer.elementRegion)
                              cb += leftProducer.elementRegion.clearRegion()
                            }
                            cb.goto(Lpush)
                          },
                          {
                            cb.ifx(c.ceq(0),
                              cb.assign(pulledRight, false))
                            cb.goto(Lcompare)
                          }
                        ),
                        {
                          cb.assign(pulledRight, true)
                          cb.goto(LpullRight)
                        })
                    }

                    mb.implementLabel(leftProducer.LendOfStream) { cb =>
                      cb.ifx(rightEOS,
                        cb.goto(LendOfStream),
                        {
                          cb.assign(leftEOS, true)
                          cb.assign(lOutMissing, true)
                          cb.assign(rOutMissing, false)
                          cb.ifx(pulledRight && c.cne(0),
                            {
                              if (rightProducer.requiresMemoryManagementPerElement) {
                                cb += elementRegion.trackAndIncrementReferenceCountOf(rightProducer.elementRegion)
                                cb += rightProducer.elementRegion.clearRegion()
                              }
                              cb.goto(Lpush)
                            },
                            {
                              cb.assign(pulledRight, true)

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
                    leftProducer.close(cb)
                    rightProducer.close(cb)
                    if (rightProducer.requiresMemoryManagementPerElement)
                      cb += rightProducer.elementRegion.invalidate()
                  }
                }

                SStreamCode(SStream(producer.element.st, leftStream.st.required && rightStream.st.required), producer)
            }
          }
        }

      case StreamGroupByKey(a, key) =>
        produce(a, cb).map(cb) { case (childStream: SStreamCode) =>

          val childProducer = childStream.producer

          val xCurElt = mb.newPField("st_grpby_curelt", childProducer.element.pt)

          val keyRegion = mb.genFieldThisRef[Region]("st_groupby_key_region")
          def subsetCode = xCurElt.asBaseStruct.subset(key: _*)
          val curKey = mb.newPField("st_grpby_curkey", subsetCode.st.pType)
          // FIXME: PType.canonical is the wrong infrastructure here. This should be some
          // notion of "cheap stype with a copy". We don't want to use a subset struct,
          // since we don't want to deep copy the parent.
          val lastKey = mb.newPField("st_grpby_lastkey", PType.canonical(subsetCode.st.pType))

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

          def equiv(cb: EmitCodeBuilder, l: SBaseStructCode, r: SBaseStructCode): Code[Boolean] =
            StructOrdering.make(l.st, r.st, cb.emb.ecb, missingFieldsEqual = false).equivNonnull(cb, l.asPCode, r.asPCode)

          val LchildProduceDoneInner = CodeLabel()
          val LchildProduceDoneOuter = CodeLabel()
          val innerProducer = new StreamProducer {
            override val length: Option[Code[Int]] = None

            override def initialize(cb: EmitCodeBuilder): Unit = {}

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

                cb.assign(lastKey, subsetCode.castTo(cb, keyRegion, lastKey.pt, deepCopy = true))
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
            override val length: Option[Code[Int]] = None

            override def initialize(cb: EmitCodeBuilder): Unit = {
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

              childProducer.initialize(cb)
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

              cb.assign(lastKey, subsetCode.castTo(cb, keyRegion, lastKey.pt, deepCopy = true))

              cb.define(LinnerStreamReady)
              cb.assign(nextGroupReady, false)
              cb.goto(LproduceElementDone)
            }

            override val element: EmitCode = EmitCode.present(mb, SStreamCode(SStream(innerProducer.element.st, true), innerProducer))

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
            cb.assign(xCurElt, childProducer.element.get())
            cb.assign(curKey, subsetCode)
            cb.ifx(inOuter, cb.goto(LchildProduceDoneOuter), cb.goto(LchildProduceDoneInner))
          }

          SStreamCode(SStream(outerProducer.element.st, required = childStream.st.required), outerProducer)
        }

      case StreamGrouped(a, groupSize) =>
        produce(a, cb).flatMap(cb) { case (childStream: SStreamCode) =>

          emit(groupSize, cb).map(cb) { case (groupSize: SInt32Code) =>

            val n = mb.genFieldThisRef[Int]("streamgrouped_n")

            val childProducer = childStream.producer

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
              override val length: Option[Code[Int]] = None

              override def initialize(cb: EmitCodeBuilder): Unit = {}

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
            val innerStreamCode = EmitCode.present(mb, SStreamCode(SStream(innerProducer.element.st, true), innerProducer))

            val outerProducer = new StreamProducer {
              override val length: Option[Code[Int]] = childProducer.length.map(l => ((l.toL + n.toL - 1L) / n.toL).toI)

              override def initialize(cb: EmitCodeBuilder): Unit = {
                cb.assign(n, groupSize.intCode(cb))
                cb.ifx(n <= 0, cb._fatal(s"stream grouped: non-positive size: ", n.toS))
                cb.assign(eos, false)
                cb.assign(xCounter, n)

                if (childProducer.requiresMemoryManagementPerElement) {
                  cb.assign(childProducer.elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
                } else {
                  cb.assign(childProducer.elementRegion, outerRegion)
                }

                childProducer.initialize(cb)
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

            SStreamCode(SStream(outerProducer.element.st, required = childStream.st.required), outerProducer)
          }
        }

      case StreamZip(as, names, body, behavior) =>
        IEmitCode.multiMapEmitCodes(cb, as.map(a => EmitCode.fromI(mb)(cb => produce(a, cb)))) { childStreams =>

          val producers = childStreams.map(_.asInstanceOf[SStreamCode].producer)

          assert(names.length == producers.length)

          val producer: StreamProducer = behavior match {
            case behavior@(ArrayZipBehavior.TakeMinLength | ArrayZipBehavior.AssumeSameLength) =>
              val vars = names.zip(producers).map { case (name, p) => mb.newEmitField(name, p.element.pt) }

              val eltRegion = mb.genFieldThisRef[Region]("streamzip_eltregion")
              val bodyCode = EmitCode.fromI(mb)(cb => emit(body, cb, region = eltRegion, env = env.bind(names.zip(vars): _*)))

              new StreamProducer {
                override val length: Option[Code[Int]] = {
                  behavior match {
                    case ArrayZipBehavior.AssumeSameLength =>
                      producers.flatMap(_.length).headOption
                    case ArrayZipBehavior.TakeMinLength =>
                      producers.map(_.length).reduceLeft(_.liftedZip(_).map {
                        case (l1, l2) => l1.min(l2)
                      })
                  }
                }

                override def initialize(cb: EmitCodeBuilder): Unit = {
                  producers.foreach { p =>
                    if (p.requiresMemoryManagementPerElement)
                      cb.assign(p.elementRegion, eltRegion)
                    else
                      cb.assign(p.elementRegion, outerRegion)
                    p.initialize(cb)
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

              val vars = names.zip(producers).map { case (name, p) => mb.newEmitField(name, p.element.pt.setRequired(false)) }

              val eltRegion = mb.genFieldThisRef[Region]("streamzip_eltregion")
              val bodyCode = EmitCode.fromI(mb)(cb => emit(body, cb, region = eltRegion, env = env.bind(names.zip(vars): _*)))

              val anyEOS = mb.genFieldThisRef[Boolean]("zip_any_eos")
              val allEOS = mb.genFieldThisRef[Boolean]("zip_all_eos")


              new StreamProducer {
                override val length: Option[Code[Int]] = producers.flatMap(_.length) match {
                  case Seq() => None
                  case ls =>
                    val len = mb.genFieldThisRef[Int]("zip_asl_len")
                    val lenTemp = mb.genFieldThisRef[Int]("zip_asl_len_temp")
                    Some(Code(
                      len := ls.head,
                      ls.tail.foldLeft(Code._empty) { (acc, l) =>
                        Code(acc,
                          lenTemp := l,
                          len.cne(lenTemp).orEmpty(Code._fatal[Unit](
                            const("zip: length mismatch: ").concat(len.toS).concat(", ").concat(lenTemp.toS))))
                      },
                      len))
                }

                override def initialize(cb: EmitCodeBuilder): Unit = {
                  cb.assign(anyEOS, false)

                  producers.foreach { p =>
                    if (p.requiresMemoryManagementPerElement)
                      cb.assign(p.elementRegion, eltRegion)
                    else
                      cb.assign(p.elementRegion, outerRegion)
                    p.initialize(cb)
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
                      cb._fatal("zip: length mismatch"))
                  )

                  cb.goto(LproduceElementDone)
                }

                val element: EmitCode = bodyCode

                def close(cb: EmitCodeBuilder): Unit = {
                  producers.foreach(_.close(cb))
                }
              }


            case ArrayZipBehavior.ExtendNA =>
              val vars = names.zip(producers).map { case (name, p) => mb.newEmitField(name, p.element.pt.setRequired(false)) }

              val eltRegion = mb.genFieldThisRef[Region]("streamzip_eltregion")
              val bodyCode = EmitCode.fromI(mb)(cb => emit(body, cb, region = eltRegion, env = env.bind(names.zip(vars): _*)))

              val eosPerStream = producers.indices.map(i => mb.genFieldThisRef[Boolean](s"zip_eos_$i"))
              val nEOS = mb.genFieldThisRef[Int]("zip_n_eos")

              new StreamProducer {
                override val length: Option[Code[Int]] = producers.map(_.length).reduceLeft(_.liftedZip(_).map {
                  case (l1, l2) => l1.max(l2)
                })

                override def initialize(cb: EmitCodeBuilder): Unit = {
                  producers.foreach { p =>
                    if (p.requiresMemoryManagementPerElement)
                      cb.assign(p.elementRegion, eltRegion)
                    else
                      cb.assign(p.elementRegion, outerRegion)
                    p.initialize(cb)
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
                    cb.assign(vars(i), EmitCode.missing(mb, vars(i).pt))

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

          SStreamCode(SStream(producer.element.st, childStreams.forall(_.pt.required)), producer)
        }

      case x@StreamZipJoin(as, key, keyRef, valsRef, joinIR) =>
        IEmitCode.multiMapEmitCodes(cb, as.map(a => EmitCode.fromI(mb)(cb => emit(a, cb)))) { children =>
          val producers = children.map(_.asStream.producer)

          // FIXME: unify
          val curValsType = x.curValsType
          val eltType = curValsType.elementType.setRequired(true).asInstanceOf[PStruct]

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

          val keyType = eltType.selectFields(key)
          val curKey = mb.newPField("st_grpby_curkey", keyType)

          val xKey = mb.newPresentEmitField("zipjoin_key", keyType)
          val xElts = mb.newPresentEmitField("zipjoin_elts", curValsType)

          val joinResult: EmitCode = EmitCode.fromI(mb) { cb =>
            val newEnv = env.bind((keyRef -> xKey), (valsRef -> xElts))
            emit(joinIR, cb, env = newEnv)
          }

          val producer = new StreamProducer {
            override val length: Option[Code[Int]] = None

            override def initialize(cb: EmitCodeBuilder): Unit = {
              cb.assign(regionArray, Code.newArray[Region](k))
              producers.zipWithIndex.foreach { case (p, idx) =>
                if (p.requiresMemoryManagementPerElement) {
                  cb.assign(p.elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
                } else
                  cb.assign(p.elementRegion, outerRegion)
                cb += (regionArray(idx) = p.elementRegion)
                p.initialize(cb)
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
              cb.assign(xKey, curKey)
              cb.assign(xElts, curValsType.constructFromElements(cb, elementRegion, k, false) { (cb, i) =>
                IEmitCode(cb, result(i).ceq(0L), eltType.loadCheapPCode(cb, result(i)))
              })
              cb.goto(LproduceElementDone)

              cb.define(LstartNewKey)
              cb.forLoop(cb.assign(i, 0), i < k, cb.assign(i, i + 1), {
                cb += (result(i) = 0L)
              })
              cb.assign(curKey, eltType.loadCheapPCode(cb, heads(winner)).subset(key: _*)
                .castTo(cb, elementRegion, curKey.pt, true))
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

                val left = eltType.loadCheapPCode(cb, heads(challenger)).subset(key: _*)
                val right = eltType.loadCheapPCode(cb, heads(winner)).subset(key: _*)
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
                  val left = eltType.loadCheapPCode(cb, heads(winner)).subset(key: _*)
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

          SStreamCode(SStream(producer.element.st, children.forall(_.pt.required)), producer)
        }

      case x@StreamMultiMerge(as, key) =>
        IEmitCode.multiMapEmitCodes(cb, as.map(a => EmitCode.fromI(mb)(cb => emit(a, cb)))) { children =>
          val producers = children.map(_.asStream.producer)

          val unifiedType = x.pType.elementType.asInstanceOf[PStruct] // FIXME unify

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
          val i = mb.genFieldThisRef[Int]("merge_i")
          val challenger = mb.genFieldThisRef[Int]("merge_challenger")

          val matchIdx = mb.genFieldThisRef[Int]("merge_match_idx")
          // Compare 'winner' with value in 'matchIdx', loser goes in 'matchIdx',
          // winner goes on to next round. A contestant '-1' beats everything
          // (negative infinity), a contestant 'k' loses to everything
          // (positive infinity), and values in between are indices into 'heads'.

          /**
            * The ordering function in StreamMultiMerge should use missingFieldsEqual=false to be consistent
            * with other nodes that deal with struct keys. When keys compare equal, the earlier index (in
            * the list of stream children) should win. These semantics extend to missing key fields, which
            * requires us to compile two orderings (l/r and r/l) to maintain the abilty to take from the
            * left when key fields are missing.
            */
          def comp(cb: EmitCodeBuilder, li: Code[Int], lv: Code[Long], ri: Code[Int], rv: Code[Long]): Code[Boolean] = {
            val l = unifiedType.loadCheapPCode(cb, lv).asBaseStruct.subset(key: _*).memoize(cb, "stream_merge_l")
            val r = unifiedType.loadCheapPCode(cb, rv).asBaseStruct.subset(key: _*).memoize(cb, "stream_merge_r")
            val ord1 = StructOrdering.make(l.asBaseStruct.st, r.asBaseStruct.st, cb.emb.ecb, missingFieldsEqual = false)
            val ord2 = StructOrdering.make(r.asBaseStruct.st, l.asBaseStruct.st, cb.emb.ecb, missingFieldsEqual = false)
            val b = cb.newLocal[Boolean]("stream_merge_comp_result")
            cb.ifx(li < ri,
              cb.assign(b, ord1.compareNonnull(cb, l, r) <= 0),
              cb.assign(b, ord2.compareNonnull(cb, r, l) > 0))
            b
          }

          val producer = new StreamProducer {
            override val length: Option[Code[Int]] = producers.map(_.length).reduce(_.liftedZip(_).map { case (l, r) => l + r })

            override def initialize(cb: EmitCodeBuilder): Unit = {
              cb.assign(regionArray, Code.newArray[Region](k))
              producers.zipWithIndex.foreach { case (p, i) =>
                if (p.requiresMemoryManagementPerElement) {
                  cb.assign(p.elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
                } else
                  cb.assign(p.elementRegion, outerRegion)
                cb += (regionArray(i) = p.elementRegion)
                p.initialize(cb)
              }
              initMemoryManagementPerElementArray(cb)
              cb.assign(bracket, Code.newArray[Int](k))
              cb.assign(heads, Code.newArray[Long](k))
              cb.forLoop(cb.assign(i, 0), i < k, cb.assign(i, i + 1), {
                cb += (bracket(i) = -1)
              })
              cb.assign(i, 0)
              cb.assign(winner, 0)
            }

            override val elementRegion: Settable[Region] = region
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
                cb.assign(matchIdx, (idx + k) >>> 1)
                cb.goto(LrunMatch)

                cb.define(p.LproduceElementDone)
                cb += (heads(idx) = unifiedType.store(cb, p.elementRegion, p.element.toI(cb).get(cb), false))
                cb.assign(matchIdx, (idx + k) >>> 1)
                cb.goto(LrunMatch)
              }
            }

            override val element: EmitCode = EmitCode.fromI(mb)(cb => IEmitCode.present(cb, unifiedType.loadCheapPCode(cb, heads(winner))))

            override def close(cb: EmitCodeBuilder): Unit = {
              producers.foreach { p =>
                if (p.requiresMemoryManagementPerElement)
                  cb += p.elementRegion.invalidate()
                p.close(cb)
              }
              cb.assign(bracket, Code._null)
              cb.assign(heads, Code._null)
            }
          }
          SStreamCode(SStream(producer.element.st, children.forall(_.pt.required)), producer)
        }

      case ReadPartition(context, rowType, reader) =>
        val ctxCode = EmitCode.fromI(mb)(cb => emit(context, cb))
        reader.emitStream(emitter.ctx.executeContext, cb, ctxCode, outerRegion, rowType)

      case ShuffleRead(idIR, keyRangeIR) =>
        val shuffleType = idIR.typ.asInstanceOf[TShuffle]
        val keyType = keyRangeIR.typ.asInstanceOf[TInterval].pointType
        val keyPType = keyRangeIR.pType.asInstanceOf[PInterval].pointType
        assert(keyType == shuffleType.keyType)
        assert(keyPType == shuffleType.keyDecodedPType)


        val region = mb.genFieldThisRef[Region]("shuffleread_region")
        val clientVar = mb.genFieldThisRef[ShuffleClient]("shuffleClient")
        val shuffle = new ValueShuffleClient(clientVar)

        val producer = new StreamProducer {
          override val length: Option[Code[Int]] = None

          override def initialize(cb: EmitCodeBuilder): Unit = {
            val idt = emit(idIR, cb).get(cb, "ShuffleRead cannot have null ID").asShuffle
            val keyRangeCode =
              emit(keyRangeIR, cb).get(cb, "ShuffleRead cannot have null key range").asInterval

            val uuid = idt.memoize(cb, "shuffleUUID")
            val keyRange = keyRangeCode.memoizeField(cb, "shuffleClientKeyRange")

            cb.ifx(!keyRange.startDefined(cb) || !keyRange.endDefined(cb), {
              Code._fatal[Unit]("ShuffleRead cannot have null start or end points of key range")
            })

            val keyPType = keyRange.st.pointType.canonicalPType()
            cb.assign(clientVar, CodeShuffleClient.create(
              mb.ecb.getType(shuffleType),
              uuid.loadBytes(),
              Code._null,
              mb.ecb.getPType(keyPType)))

            val startt = keyPType.store(cb, outerRegion, keyRange.loadStart(cb)
              .get(cb, "shuffle expects defined endpoints")
              .asPCode, false)
            val endt = keyPType.store(cb, outerRegion, keyRange.loadEnd(cb)
              .get(cb, "shuffle expects defined endpoints")
              .asPCode, false)

            cb.append(shuffle.startGet(startt, keyRange.includesStart(), endt, keyRange.includesEnd()))
          }

          override val elementRegion: Settable[Region] = region
          override val requiresMemoryManagementPerElement: Boolean = true
          override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
            cb.ifx(shuffle.getValueFinished(), cb.goto(LendOfStream))
            cb.goto(LproduceElementDone)
          }
          override val element: EmitCode = EmitCode.present(mb, PCode(shuffleType.rowDecodedPType, shuffle.getValue(region)))

          override def close(cb: EmitCodeBuilder): Unit = {
            cb += shuffle.getDone()
            cb += shuffle.close()
          }
        }

        IEmitCode.present(cb, SStreamCode(SStream(producer.element.st, true), producer))

      case ShufflePartitionBounds(idIR, nPartitionsIR) =>

        val region = mb.genFieldThisRef[Region]("shuffle_partition_bounds_region")
        val shuffleLocal = mb.genFieldThisRef[ShuffleClient]("shuffle_partition_bounds_client")
        val shuffle = new ValueShuffleClient(shuffleLocal)
        val currentAddr = mb.genFieldThisRef[Long]("shuffle_partition_bounds_addr")

        val shuffleType = idIR.typ.asInstanceOf[TShuffle]

        val producer = new StreamProducer {
          override val length: Option[Code[Int]] = None

          override def initialize(cb: EmitCodeBuilder): Unit = {

            val idt = emit(idIR, cb).get(cb, "ShufflePartitionBounds cannot have null ID").asInstanceOf[SCanonicalShufflePointerCode]
            val nPartitionst = emit(nPartitionsIR, cb).get(cb, "ShufflePartitionBounds cannot have null number of partitions").asInt

            val uuidLocal = mb.newLocal[Long]("shuffleUUID")
            val uuid = new SCanonicalShufflePointerSettable(idt.st, new SBinaryPointerSettable(SBinaryPointer(idt.st.pType.representation), uuidLocal))
            uuid.store(cb, idt)
            cb.assign(shuffleLocal, CodeShuffleClient.create(mb.ecb.getType(shuffleType), uuid.loadBytes()))
            cb += shuffle.startPartitionBounds(nPartitionst.intCode(cb))
          }

          override val elementRegion: Settable[Region] = region
          override val requiresMemoryManagementPerElement: Boolean = false
          override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
            cb.ifx(shuffle.partitionBoundsValueFinished(), cb.goto(LendOfStream))
            cb.assign(currentAddr, shuffle.partitionBoundsValue(region))
            cb.goto(LproduceElementDone)
          }
          override val element: EmitCode = EmitCode.fromI(mb)(cb =>
            IEmitCode.present(cb, shuffleType.keyDecodedPType.loadCheapPCode(cb, currentAddr)))

          override def close(cb: EmitCodeBuilder): Unit = {
            cb += shuffle.endPartitionBounds()
            cb += shuffle.close()
          }
        }
        IEmitCode.present(cb, SStreamCode(SStream(producer.element.st, true), producer))
    }
  }
}

