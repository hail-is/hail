package is.hail.expr.ir.streams

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitStream.emit
import is.hail.expr.ir.Stream.unfold
import is.hail.expr.ir._
import is.hail.expr.ir.orderings.StructOrdering
import is.hail.services.shuffler.{CodeShuffleClient, ShuffleClient, ValueShuffleClient}
import is.hail.types.physical.stypes.concrete.{SBinaryPointer, SBinaryPointerSettable, SCanonicalShufflePointerCode, SCanonicalShufflePointerSettable}
import is.hail.types.physical.stypes.{SType, interfaces}
import is.hail.types.physical.stypes.interfaces.{SBaseStruct, SIndexableCode, SIndexableValue, SStream}
import is.hail.types.physical.stypes.primitives.{SInt32, SInt32Code}
import is.hail.types.physical.{PCanonicalStream, PCode, PInterval, PStream, PStreamCode, PValue}
import is.hail.types.virtual.{TInterval, TShuffle, TStream}
import is.hail.utils._


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
    * This boolean parameter indicates whether the producer's elements should be emitted into
    * separate regions (by clearing when elements leave a consumer's scope). This parameter
    * propagates bottom-up from producers like [[ReadPartition]] and [[StreamRange]], but
    * it is the responsibility of consumers to implement the right memory management semantics
    * based on this flag.
    */
  val separateRegions: Boolean

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

  final def consume2(cb: EmitCodeBuilder, setup: EmitCodeBuilder => Unit = _ => ())(perElement: EmitCodeBuilder => Unit): Unit = {

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

    if (separateRegions)
      cb.assign(elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
    else
      cb.assign(elementRegion, outerRegion)

    this.initialize(cb)
    setup(cb)

    cb.goto(this.LproduceElement)
    cb.define(this.LproduceElementDone)
    perElement(cb)

    if (separateRegions)
      cb += elementRegion.clearRegion()

    cb.goto(this.LproduceElement)

    cb.define(this.LendOfStream)
    this.close(cb)

    if (separateRegions)
      cb += elementRegion.freeRegion()
  }
}

final case class SStreamCode2(st: SStream, producer: StreamProducer) extends PStreamCode {
  self =>
  override def pt: PStream = st.pType

  def memoize(cb: EmitCodeBuilder, name: String): PValue = new PValue {
    def pt: PStream = PCanonicalStream(st.pType)

    override def st: SType = self.st

    var used: Boolean = false

    def get: PCode = {
      assert(!used)
      used = true
      self
    }
  }
}

object EmitStream2 {
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
      emitter.emitVoid(cb, ir, StagedRegion(region), env, container, None)

    def emit(ir: IR, cb: EmitCodeBuilder, region: Value[Region] = outerRegion, env: Emit.E = env, container: Option[AggContainer] = container): IEmitCode = {
      ir.typ match {
        case _: TStream => produce(ir, cb, outerRegion, env, container)
        case _ => emitter.emitI(ir, cb, StagedRegion(region), env, container, None)
      }
    }

    def produce(streamIR: IR, cb: EmitCodeBuilder, region: Value[Region] = outerRegion, env: Emit.E = env, container: Option[AggContainer] = container): IEmitCode =
      EmitStream2.produce(emitter, streamIR, cb, region, env, container)

    streamIR match {

      case NA(_typ) =>
        val eltType = streamIR.pType.asInstanceOf[PCanonicalStream].elementType
        val st = SStream(eltType.sType, false, false)
        val region = mb.genFieldThisRef[Region]("na_region")
        val producer = new StreamProducer {
          override def initialize(cb: EmitCodeBuilder): Unit = {}

          override val length: Option[Code[Int]] = None
          override val elementRegion: Settable[Region] = region
          override val separateRegions: Boolean = false
          override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>
            cb.goto(LendOfStream)
          }
          override val element: EmitCode = EmitCode.missing(mb, eltType)

          override def close(cb: EmitCodeBuilder): Unit = {}
        }
        IEmitCode.missing(cb, SStreamCode2(st, producer))

      case Ref(name, _typ) =>
        assert(_typ.isInstanceOf[TStream])
        env.lookup(name).toI(cb)
          .map(cb) { case (stream: SStreamCode2) =>
            val childProducer = stream.producer
            val producer = new StreamProducer {
              override def initialize(cb: EmitCodeBuilder): Unit = childProducer.initialize(cb)

              override val length: Option[Code[Int]] = childProducer.length
              override val elementRegion: Settable[Region] = childProducer.elementRegion
              override val separateRegions: Boolean = childProducer.separateRegions
              override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>
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
        val (fixupUnusedStreamLabels, xValue) = cb.memoizeMaybeUnrealizableField(
          EmitCode.fromI(mb)(cb => emit(value, cb)), s"let_$name")
        val bodyStream = produce(body, cb, env = env.bind(name, xValue))
        fixupUnusedStreamLabels()
        bodyStream

      case In(n, _) =>
        // this, Code[Region], ...
        val param = mb.getEmitParam(2 + n, outerRegion)
        param.st match {
          case _: SStream =>
          case t => throw new RuntimeException(s"parameter ${ 2 + n } is not a stream! t=$t, params=${ mb.emitParamTypes }")
        }
        param.load.toI(cb)


      case ToStream(a, _separateRegions) =>

        emit(a, cb).map(cb) { case ind: SIndexableCode =>
          val containerField = mb.newPField(ind.pt)
          val container = containerField.asInstanceOf[SIndexableValue]
          val idx = mb.genFieldThisRef[Int]("tostream_idx")
          val regionVar = mb.genFieldThisRef[Region]("tostream_region")

          SStreamCode2(
            SStream(ind.st.elementType, ind.pt.required),
            new StreamProducer {
              override def initialize(cb: EmitCodeBuilder): Unit = {
                cb.assign(containerField, ind)
                cb.assign(idx, -1)
              }

              override val length: Option[Code[Int]] = Some(container.loadLength())

              override val elementRegion: Settable[Region] = regionVar

              override val separateRegions: Boolean = _separateRegions

              override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>
                cb.assign(idx, idx + 1)
                cb.ifx(idx >= container.loadLength(), cb.goto(LendOfStream))
                cb.goto(LproduceElementDone)
              }

              val element: EmitCode = EmitCode.fromI(mb)(cb => container.loadElement(cb, idx).typecast[PCode])

              def close(cb: EmitCodeBuilder): Unit = {}
            })

        }

      case x@MakeStream(args, _, _separateRegions) =>
        val region = mb.genFieldThisRef[Region]("makestream_region")
        val emittedArgs = args.map(a => EmitCode.fromI(mb)(cb => emit(a, cb, region))).toFastIndexedSeq

        val unifiedType = x.pType.asInstanceOf[PCanonicalStream].elementType.sType // FIXME
        val eltField = mb.newEmitField("makestream_elt", unifiedType.pType)

        val staticLen = args.size
        val current = mb.genFieldThisRef[Int]("makestream_current")

        IEmitCode.present(cb, SStreamCode2(
          SStream(unifiedType, required = true, separateRegions = _separateRegions),
          new StreamProducer {
            override def initialize(cb: EmitCodeBuilder): Unit = {
              cb.assign(current, 0) // switches on 1..N
            }

            override val length: Option[Code[Int]] = Some(staticLen)

            override val elementRegion: Settable[Region] = region

            override val separateRegions: Boolean = _separateRegions

            override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>

              val LendOfSwitch = CodeLabel()
              cb += Code.switch(current, LendOfStream.goto,
                emittedArgs.map(elem => EmitCodeBuilder.scopedVoid(mb) { cb =>
                  cb.assign(eltField, elem.map(pc => pc.castTo(cb, region, unifiedType.pType, false)))
                  cb.goto(LendOfSwitch)
                }))
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

          val leftProducer = leftEC.pv.asStream2.producer
          val rightProducer = rightEC.pv.asStream2.producer

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
            override val separateRegions: Boolean = leftProducer.separateRegions || rightProducer.separateRegions
            override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>
              cb.ifx(xCond, cb.goto(leftProducer.LproduceElement), cb.goto(rightProducer.LproduceElement))

              cb.define(leftProducer.LproduceElementDone)
              cb.assign(xElt, leftProducer.element.map(_.castTo(cb, region, xElt.pt)))
              cb.goto(LproduceElementDone)

              cb.define(rightProducer.LproduceElementDone)
              cb.assign(xElt, rightProducer.element.map(_.castTo(cb, region, xElt.pt)))
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
            SStreamCode2(SStream(xElt.st, required = leftEC.pt.required && rightEC.pt.required, producer.separateRegions), producer))
        }

      case StreamRange(startIR, stopIR, stepIR, _separateRegions) =>

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

                override val separateRegions: Boolean = _separateRegions

                override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>
                  cb.ifx(idx >= len, cb.goto(LendOfStream))
                  cb.assign(curr, curr + step)
                  cb.assign(idx, idx + 1)
                  cb.goto(LproduceElementDone)
                }

                val element: EmitCode = EmitCode.present(mb, new SInt32Code(true, curr))

                def close(cb: EmitCodeBuilder): Unit = {}
              }
              SStreamCode2(
                SStream(SInt32(true), required = true),
                producer
              )
            }
          }
        }

      case StreamFilter(a, name, cond) =>
        produce(a, cb)
          .map(cb) { case (childStream: SStreamCode2) =>
            val childProducer = childStream.producer

            val filterEltRegion = mb.genFieldThisRef[Region]("streamfilter_filter_region")

            val elementField = cb.emb.newEmitField("streamfilter_cond", childStream.st.elementType.pType)

            val producer = new StreamProducer {
              override val length: Option[Code[Int]] = None

              override def initialize(cb: EmitCodeBuilder): Unit = {
                if (childProducer.separateRegions)
                  cb.assign(childProducer.elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
                else
                  cb.assign(childProducer.elementRegion, outerRegion)
                childProducer.initialize(cb)
              }

              override val elementRegion: Settable[Region] = filterEltRegion

              override val separateRegions: Boolean = childProducer.separateRegions

              override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>
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

                if (separateRegions)
                  cb += childProducer.elementRegion.addReferenceTo(filterEltRegion)

                cb.goto(LproduceElementDone)

                cb.define(Lfiltered)
                if (separateRegions)
                  cb += childProducer.elementRegion.clearRegion()
                cb.goto(childProducer.LproduceElement)
              }

              val element: EmitCode = elementField

              def close(cb: EmitCodeBuilder): Unit = {
                childProducer.close(cb)
                if (separateRegions)
                  cb += childProducer.elementRegion.freeRegion()
              }
            }
            mb.implementLabel(childProducer.LendOfStream) { cb =>
              cb.goto(producer.LendOfStream)
            }

            SStreamCode2(
              childStream.st,
              producer)
          }

      case StreamTake(a, num) =>
        produce(a, cb)
          .flatMap(cb) { case (childStream: SStreamCode2) =>
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
                override val separateRegions: Boolean = childProducer.separateRegions
                override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>
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

              SStreamCode2(childStream.st, producer)
            }
          }

      case StreamDrop(a, num) =>
        produce(a, cb)
          .flatMap(cb) { case (childStream: SStreamCode2) =>
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
                override val separateRegions: Boolean = childProducer.separateRegions
                override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>
                  cb.goto(childProducer.LproduceElement)
                  cb.define(childProducer.LproduceElementDone)
                  cb.assign(idx, idx + 1)
                  cb.ifx(idx <= n, {
                    if (childProducer.separateRegions)
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

              SStreamCode2(childStream.st, producer)
            }
          }

      case StreamMap(a, name, body) =>
        produce(a, cb)
          .map(cb) { case (childStream: SStreamCode2) =>
            val childProducer = childStream.producer

            val bodyResult = EmitCode.fromI(mb) { cb =>
              val (fixUnusedStreamLabels, childProducerElement) = cb.memoizeMaybeUnrealizableField(childProducer.element, "streammap_element")

              val emitted = emit(body,
                cb = cb,
                env = env.bind(name, childProducerElement),
                region = childProducer.elementRegion)

              fixUnusedStreamLabels()
              emitted
            }

            val producer: StreamProducer = new StreamProducer {
              override val length: Option[Code[Int]] = childProducer.length

              override def initialize(cb: EmitCodeBuilder): Unit = {
                childProducer.initialize(cb)
              }

              override val elementRegion: Settable[Region] = childProducer.elementRegion

              override val separateRegions: Boolean = childProducer.separateRegions

              override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>
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

            SStreamCode2(
              SStream(bodyResult.st, required = childStream.st.required),
              producer
            )
          }

      case x@StreamScan(childIR, zeroIR, accName, eltName, bodyIR) =>
        produce(childIR, cb).map(cb) { case (childStream: SStreamCode2) =>
          val childProducer = childStream.producer

          val accValueAccRegion = mb.newEmitField(x.accPType)
          val accValueEltRegion = mb.newEmitField(x.accPType)

          // accRegion is unused if separateRegions is false
          var accRegion: Settable[Region] = null
          val first = mb.genFieldThisRef[Boolean]("streamscan_first")

          val producer = new StreamProducer {
            override val length: Option[Code[Int]] = childProducer.length.map(_ + const(1))

            override def initialize(cb: EmitCodeBuilder): Unit = {

              if (childProducer.separateRegions) {
                accRegion = mb.genFieldThisRef[Region]("streamscan_acc_region")
                cb.assign(accRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
              }
              cb.assign(first, true)
              childProducer.initialize(cb)
            }

            override val elementRegion: Settable[Region] = childProducer.elementRegion

            override val separateRegions: Boolean = childProducer.separateRegions

            override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>

              val LcopyAndReturn = CodeLabel()

              cb.ifx(first, {

                cb.assign(first, false)
                cb.assign(accValueEltRegion, emit(zeroIR, cb, region = elementRegion))

                cb.goto(LcopyAndReturn)
              })


              cb.goto(childProducer.LproduceElement)
              cb.define(childProducer.LproduceElementDone)

              if (separateRegions) {
                // deep copy accumulator into element region, then clear accumulator region
                cb.assign(accValueEltRegion, accValueAccRegion.map(_.castTo(cb, childProducer.elementRegion, x.accPType, deepCopy = true)))
                cb += accRegion.clearRegion()
              }

              val bodyCode = {
                val (fixUnusedStreamLabels, childEltValue) = cb.memoizeMaybeUnrealizableField(childProducer.element, "scan_child_elt")
                val bodyCode = emit(bodyIR, cb, env = env.bind((accName, accValueEltRegion), (eltName, childEltValue)), region = childProducer.elementRegion)
                  .map(cb)(pc => pc.castTo(cb, childProducer.elementRegion, x.accPType, deepCopy = false))

                fixUnusedStreamLabels()
                bodyCode
              }

              cb.assign(accValueEltRegion, bodyCode)

              cb.define(LcopyAndReturn)

              if (separateRegions) {
                cb.assign(accValueAccRegion, accValueEltRegion.map(pc => pc.castTo(cb, accRegion, x.accPType, deepCopy = true)))
              }

              cb.goto(LproduceElementDone)
            }

            val element: EmitCode = accValueEltRegion.load

            override def close(cb: EmitCodeBuilder): Unit = {
              childProducer.close(cb)
              if (separateRegions)
                cb += accRegion.freeRegion()
            }
          }

          mb.implementLabel(childProducer.LendOfStream) { cb =>
            cb.goto(producer.LendOfStream)
          }

          SStreamCode2(SStream(accValueEltRegion.st, childStream.st.required, producer.separateRegions), producer)
        }

      case RunAggScan(child, name, init, seqs, result, states) =>
        val (newContainer, aggSetup, aggCleanup) = AggContainer.fromMethodBuilder(states.toArray, mb, "run_agg_scan")

        produce(child, cb).map(cb) { case (childStream: SStreamCode2) =>
          val childProducer = childStream.producer

          val childEltField = mb.newEmitField("runaggscan_child_elt", childProducer.element.pt)
          val bodyEnv = env.bind(name -> childEltField)
          val bodyResult = EmitCode.fromI(mb)(cb => emit(result, cb = cb, region = childProducer.elementRegion,
            env = bodyEnv, container = Some(newContainer)))
          val bodyResultField = mb.newEmitField("runaggscan_result_elt", bodyResult.pt)

          val producer = new StreamProducer {
            override val length: Option[Code[Int]] = childProducer.length

            override def initialize(cb: EmitCodeBuilder): Unit = {
              cb += aggSetup
              emitVoid(init, cb = cb, region = outerRegion, container = Some(newContainer))
              childProducer.initialize(cb)
            }

            override val elementRegion: Settable[Region] = childProducer.elementRegion
            override val separateRegions: Boolean = childProducer.separateRegions
            override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>
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
              cb += aggCleanup
            }
          }

          mb.implementLabel(childProducer.LendOfStream) { cb =>
            cb.goto(producer.LendOfStream)
          }

          SStreamCode2(SStream(producer.element.st, childStream.st.required, producer.separateRegions), producer)
        }

      case StreamFlatMap(a, name, body) =>
        produce(a, cb).map(cb) { case (outerStream: SStreamCode2) =>
          val outerProducer = outerStream.producer

          // variables used in control flow
          val first = mb.genFieldThisRef[Boolean]("flatmap_first")
          val innerUnclosed = mb.genFieldThisRef[Boolean]("flatmap_inner_unclosed")

          val innerStreamEmitCode = EmitCode.fromI(mb) { cb =>
            val (fixUnusedStreamLabels, outerProducerValue) = cb.memoizeMaybeUnrealizableField(outerProducer.element, "flatmap_outer_value")
            val emitted = emit(body,
              cb = cb,
              env = env.bind(name, outerProducerValue),
              region = outerProducer.elementRegion)

            fixUnusedStreamLabels()
            emitted
          }

          // grabbing emitcode.pv weird pattern but should be safe
          val SStreamCode2(_, innerProducer) = innerStreamEmitCode.pv

          val producer = new StreamProducer {
            override val length: Option[Code[Int]] = None

            override def initialize(cb: EmitCodeBuilder): Unit = {
              cb.assign(first, true)

              if (outerProducer.separateRegions)
                cb.assign(outerProducer.elementRegion, Region.stagedCreate(Region.REGULAR, cb.emb.ecb.pool()))
              else
                cb.assign(outerProducer.elementRegion, outerRegion)

              outerProducer.initialize(cb)
            }

            override val elementRegion: Settable[Region] = innerProducer.elementRegion

            override val separateRegions: Boolean = innerProducer.separateRegions || outerProducer.separateRegions

            override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>
              val LnextOuter = CodeLabel()
              val LnextInner = CodeLabel()
              cb.ifx(first, {
                cb.assign(first, false)

                cb.define(LnextOuter)
                cb.define(innerProducer.LendOfStream)

                if (outerProducer.separateRegions)
                  cb += outerProducer.elementRegion.clearRegion()


                cb.ifx(innerUnclosed, {
                  cb.assign(innerUnclosed, false)
                  innerProducer.close(cb)
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
                      cb.goto(LnextInner)
                  }
                )
              }, {

                cb.define(LnextInner)
                cb.goto(innerProducer.LproduceElement)
                cb.define(innerProducer.LproduceElementDone)
                cb.goto(LproduceElementDone)
              })
            }

            val element: EmitCode = innerProducer.element

            def close(cb: EmitCodeBuilder): Unit = {
              cb.ifx(innerUnclosed, {
                cb.assign(innerUnclosed, false)
                innerProducer.close(cb)
              })
              outerProducer.close(cb)

              if (outerProducer.separateRegions)
                cb += outerProducer.elementRegion.freeRegion()
            }
          }

          mb.implementLabel(outerProducer.LendOfStream) { cb =>
            cb.goto(producer.LendOfStream)
          }

          SStreamCode2(
            SStream(innerProducer.element.st, required = outerStream.st.required),
            producer
          )
        }

      case x@StreamJoinRightDistinct(leftIR, rightIR, lKey, rKey, leftName, rightName, joinIR, joinType) =>
        produce(leftIR, cb).flatMap(cb) { case (leftStream: SStreamCode2) =>
          produce(rightIR, cb).map(cb) { case (rightStream: SStreamCode2) =>

            val leftProducer = leftStream.producer
            val rightProducer = rightStream.producer

            val lEltType = leftProducer.element.pt
            val rEltType = rightProducer.element.pt

            // these variables are used as inputs to the joinF

            def compare(cb: EmitCodeBuilder, lelt: EmitValue, relt: EmitValue): Code[Int] = {
              assert(lelt.pt == lEltType)
              assert(relt.pt == rEltType)

              val lhs = lelt.map(_.asBaseStruct.subset(lKey: _*).asPCode)
              val rhs = relt.map(_.asBaseStruct.subset(rKey: _*).asPCode)
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

                val producer = new StreamProducer {
                  override val length: Option[Code[Int]] = leftProducer.length

                  override def initialize(cb: EmitCodeBuilder): Unit = {
                    cb.assign(rightEOS, false)
                    cb.assign(pulledRight, false)

                    if (rightProducer.separateRegions)
                      cb.assign(rightProducer.elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))

                    leftProducer.initialize(cb)
                    rightProducer.initialize(cb)
                  }

                  override val elementRegion: Settable[Region] = leftProducer.elementRegion
                  override val separateRegions: Boolean = leftProducer.separateRegions
                  override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>

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
                      if (rightProducer.separateRegions)
                        cb += rightProducer.elementRegion.addReferenceTo(elementRegion)
                      cb.assign(rxOut, rx)
                    })

                    cb.goto(LproduceElementDone)

                    cb.define(LpullRight)
                    if (rightProducer.separateRegions) {
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
                    cb.goto(LproduceElementDone)
                  }
                  override val element: EmitCode = joinResult

                  override def close(cb: EmitCodeBuilder): Unit = {
                    leftProducer.close(cb)
                    rightProducer.close(cb)
                    if (rightProducer.separateRegions)
                      cb += rightProducer.elementRegion.freeRegion()
                  }
                }

                mb.implementLabel(leftProducer.LendOfStream) { cb =>
                  cb.goto(producer.LendOfStream)
                }


                SStreamCode2(SStream(producer.element.st, leftStream.st.required && rightStream.st.required, producer.separateRegions), producer)

              case "outer" =>

                val lx = mb.newEmitField(lEltType) // last value received from left
                val rx = mb.newEmitField(rEltType) // last value received from right
                val lxOut = mb.newEmitField(lEltType.setRequired(false)) // left value in joinF (may be missing while lx is not)
                val rxOut = mb.newEmitField(rEltType.setRequired(false)) // right value in joinF (may be missing while rx is not)

                val pulledRight = mb.genFieldThisRef[Boolean]("join_right_distinct_pulledRight")
                val rightEOS = mb.genFieldThisRef[Boolean]("join_right_distinct_rightEOS")
                val leftMissing = mb.genFieldThisRef[Boolean]("join_right_distinct_leftMissing")
                val rightMissing = mb.genFieldThisRef[Boolean]("join_right_distinct_rightMissing")
                val leftEOS = mb.genFieldThisRef[Boolean]("join_right_distinct_leftEOS")
                val compResult = mb.genFieldThisRef[Int]("join_right_distinct_compResult")
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
                    cb.assign(compResult, 0) // lets us start stream with a pull from both

                    if (rightProducer.separateRegions) {
                      cb.assign(leftProducer.elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
                      cb.assign(rightProducer.elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
                    } else {
                      cb.assign(leftProducer.elementRegion, outerRegion)
                      cb.assign(rightProducer.elementRegion, outerRegion)
                    }

                    leftProducer.initialize(cb)
                    rightProducer.initialize(cb)
                  }

                  override val elementRegion: Settable[Region] = _elementRegion
                  override val separateRegions: Boolean = leftProducer.separateRegions || rightProducer.separateRegions
                  override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>

                    val Lcompare = CodeLabel()
                    val LclearingLeftProduceElement = CodeLabel()
                    val LclearingRightProduceElement = CodeLabel()

                    val Lstart = CodeLabel()
                    val Ldone = CodeLabel()
                    cb.define(Lstart)

                    cb.ifx(rightEOS, cb.goto(LclearingLeftProduceElement))
                    cb.ifx(leftEOS, cb.goto(LclearingRightProduceElement))
                    cb.ifx(compResult <= 0,
                      cb.goto(LclearingLeftProduceElement),
                      cb.goto(LclearingRightProduceElement))

                    cb.define(rightProducer.LproduceElementDone)
                    cb.assign(rx, rightProducer.element)
                    cb.assign(pulledRight, true)
                    cb.goto(Lcompare)

                    cb.define(leftProducer.LproduceElementDone)
                    cb.assign(lx, leftProducer.element)
                    cb.ifx((compResult <= 0) && !pulledRight,
                      cb.goto(LclearingRightProduceElement))
                    cb.goto(Lcompare)

                    cb.define(Lcompare)
                    cb.ifx(rightEOS || leftEOS, cb.goto(Ldone))
                    cb.assign(compResult, compare(cb, lx, rx))
                    cb.ifx(compResult < 0, {
                      cb.assign(leftMissing, false)
                      cb.assign(rightMissing, true)
                    }, {
                      cb.ifx(compResult > 0, {
                        cb.assign(leftMissing, true)
                        cb.assign(rightMissing, false)
                      }, {
                        // compResult == 0
                        cb.assign(leftMissing, false)
                        cb.assign(rightMissing, false)
                      })
                    })

                    cb.define(Ldone)
                    cb.ifx(leftMissing,
                      cb.assign(lxOut, EmitCode.missing(mb, lxOut.pt)),
                      cb.assign(lxOut, lx)
                    )
                    cb.ifx(rightMissing,
                      cb.assign(lxOut, EmitCode.missing(mb, lxOut.pt)),
                      cb.assign(lxOut, lx)
                    )

                    cb.goto(LproduceElementDone)


                    // EOS labels
                    cb.define(rightProducer.LendOfStream)
                    cb.ifx(leftEOS, cb.goto(LendOfStream))
                    cb.assign(rightEOS, true)
                    cb.assign(rightMissing, true)
                    cb.goto(Lstart)

                    cb.define(leftProducer.LendOfStream)
                    cb.ifx(rightEOS, cb.goto(LendOfStream))
                    cb.assign(leftEOS, true)
                    cb.assign(leftMissing, true)
                    cb.goto(Lstart)

                    // clear regions before producing next elements
                    cb.define(LclearingLeftProduceElement)
                    if (leftProducer.separateRegions)
                      cb += leftProducer.elementRegion.clearRegion()
                    cb.goto(leftProducer.LproduceElement)

                    cb.define(LclearingRightProduceElement)
                    if (rightProducer.separateRegions)
                      cb += rightProducer.elementRegion.clearRegion()
                    cb.goto(rightProducer.LproduceElement)
                  }
                  override val element: EmitCode = joinResult

                  override def close(cb: EmitCodeBuilder): Unit = {
                    leftProducer.close(cb)
                    rightProducer.close(cb)
                    if (rightProducer.separateRegions)
                      cb += rightProducer.elementRegion.freeRegion()
                  }
                }

                SStreamCode2(SStream(producer.element.st, leftStream.st.required && rightStream.st.required, producer.separateRegions), producer)
            }
          }
        }

      case StreamGroupByKey(a, key) =>
        produce(a, cb).map(cb) { case (childStream: SStreamCode2) =>

          val childProducer = childStream.producer

          val xCurElt = mb.newPField("st_grpby_curelt", childProducer.element.pt)

          val keyRegion = mb.genFieldThisRef[Region]("st_groupby_key_region")
          val subsetCode = xCurElt.asBaseStruct.subset(key: _*)
          val curKey = mb.newPField("st_grpby_curkey", subsetCode.st.pType)
          val lastKey = mb.newPField("st_grpby_lastkey", subsetCode.st.pType)

          val eos = mb.genFieldThisRef[Boolean]("st_grpby_eos")
          val nextReady = mb.genFieldThisRef[Boolean]("streamgrouped_nextready")
          val inOuter = mb.genFieldThisRef[Boolean]("streamgrouped_inouter")
          val first = mb.genFieldThisRef[Boolean]("streamgrouped_first")

          // cannot reuse childProducer.elementRegion because consumers might free the region, even though
          // the outer producer needs to continue pulling. We could add more control flow that sets some
          // boolean flag when the inner stream is closed, and the outer producer reassigns a region if
          // that flag is set, but that design seems more complicated
          val innerResultRegion = mb.genFieldThisRef[Region]("streamgrouped_inner_result_region")

          val outerElementRegion = mb.genFieldThisRef[Region]("streamgrouped_outer_elt_region")

          val LchildProduceDoneInner = CodeLabel()
          val LchildProduceDoneOuter = CodeLabel()
          val innerProducer = new StreamProducer {
            override val length: Option[Code[Int]] = None

            override def initialize(cb: EmitCodeBuilder): Unit = {}

            override val elementRegion: Settable[Region] = innerResultRegion
            override val separateRegions: Boolean = childProducer.separateRegions
            override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>
              val LelementReady = CodeLabel()

              cb.ifx(nextReady, {
                cb.assign(nextReady, false)
                cb.goto(LelementReady)
              })

              cb.goto(childProducer.LproduceElement)
              // xElt and curKey are assigned before this label is jumped to
              cb.define(LchildProduceDoneInner)

              // the first element of the outer stream has not yet initialized lastKey
              cb.ifx(first, {
                cb.assign(first, false)
                cb.assign(lastKey, subsetCode.castTo(cb, keyRegion, lastKey.pt, deepCopy = true))
                cb.goto(LelementReady)
              })

              val equiv = cb.emb.ecb.getOrdering(curKey.st, lastKey.st).equivNonnull(cb, curKey, lastKey)

              // if not equivalent, end inner stream and prepare for next outer iteration
              cb.ifx(!equiv, {
                if (separateRegions)
                  cb += keyRegion.clearRegion()

                cb.assign(lastKey, subsetCode.castTo(cb, keyRegion, lastKey.pt, deepCopy = true))
                cb.assign(nextReady, true)
                cb.assign(inOuter, true)
                cb.goto(LendOfStream)
              })

              cb.define(LelementReady)

              if (separateRegions) {
                cb += childProducer.elementRegion.addReferenceTo(innerResultRegion)
                cb += childProducer.elementRegion.clearRegion()
              }

              cb.goto(LproduceElementDone)
            }
            override val element: EmitCode = EmitCode.present(mb, xCurElt)

            override def close(cb: EmitCodeBuilder): Unit = {}
          }

          val innerStreamCode = EmitCode.fromI(mb) { cb =>
            if (childProducer.separateRegions)
              cb.assign(innerProducer.elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
            else
              cb.assign(innerProducer.elementRegion, outerRegion)

            IEmitCode.present(cb, SStreamCode2(SStream(innerProducer.element.st, true, childProducer.separateRegions), innerProducer))
          }

          val outerProducer = new StreamProducer {
            override val length: Option[Code[Int]] = None

            override def initialize(cb: EmitCodeBuilder): Unit = {
              cb.assign(nextReady, false)
              cb.assign(eos, false)
              cb.assign(inOuter, true)
              cb.assign(first, true)

              if (childProducer.separateRegions) {
                cb.assign(keyRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
                cb.assign(childProducer.elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
              } else {
                cb.assign(keyRegion, outerRegion)
                cb.assign(childProducer.elementRegion, outerRegion)
              }

              childProducer.initialize(cb)
            }

            override val elementRegion: Settable[Region] = outerElementRegion
            override val separateRegions: Boolean = childProducer.separateRegions
            override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>
              cb.ifx(eos, {
                cb.goto(LendOfStream)
              })

              val LinnerStreamReady = CodeLabel()

              // if !nextReady, there is no key available, and it must be assigned
              cb.ifx(inOuter, cb.goto(LinnerStreamReady))

              // inOuter == false, so inner stream didn't reach completion and find the next key

              cb.goto(childProducer.LproduceElement)
              // xElt and curKey are assigned before this label is jumped to
              cb.define(LchildProduceDoneOuter)

              // it's possible to end up in the outer stream with first=true if the inner stream never iterates
              // in this case, the first element (and its equal keys) should be skipped, not returned
              cb.ifx(first, {
                cb.assign(first, false)
                cb.assign(lastKey, subsetCode.castTo(cb, keyRegion, lastKey.pt, deepCopy = true))
                cb.goto(childProducer.LproduceElement)
              })

              val equiv = cb.emb.ecb.getOrdering(curKey.st, lastKey.st).equivNonnull(cb, curKey, lastKey)

              // if not equivalent, end inner stream and prepare for next outer iteration
              cb.ifx(!equiv, {
                if (separateRegions)
                  cb += keyRegion.clearRegion()

                cb.assign(lastKey, subsetCode.castTo(cb, keyRegion, lastKey.pt, deepCopy = true))
                cb.assign(nextReady, true)
                cb.goto(LinnerStreamReady)
              }, {
                cb.goto(childProducer.LproduceElement)
              })

              cb.define(LinnerStreamReady)
              cb.assign(inOuter, false)
              cb.goto(LproduceElementDone)
            }

            override val element: EmitCode = innerStreamCode

            override def close(cb: EmitCodeBuilder): Unit = {
              childProducer.close(cb)
              if (childProducer.separateRegions) {
                cb += keyRegion.freeRegion()
                cb += childProducer.elementRegion.freeRegion()
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

          SStreamCode2(SStream(outerProducer.element.st, required = childStream.st.required, childProducer.separateRegions), outerProducer)
        }

      case StreamGrouped(a, groupSize) =>
        produce(a, cb).flatMap(cb) { case (childStream: SStreamCode2) =>

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
              override val separateRegions: Boolean = childProducer.separateRegions
              override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>
                cb.ifx(inOuter, {
                  cb.assign(inOuter, false)
                  cb.ifx(xCounter.cne(0), cb._fatal(s"streamgrouped inner producer error, xCounter=", xCounter.toS))
                })
                cb.ifx(xCounter >= n, {
                  cb.assign(xCounter, 0)
                  cb.assign(inOuter, true)
                  cb.goto(LendOfStream)
                })

                cb.goto(childProducer.LproduceElement)
                cb.define(LchildProduceDoneInner)

                cb += childProducer.elementRegion.addReferenceTo(innerResultRegion)

                cb.goto(LproduceElementDone)
              }
              override val element: EmitCode = childProducer.element

              override def close(cb: EmitCodeBuilder): Unit = {}
            }
            val innerStreamCode = EmitCode.present(mb, SStreamCode2(SStream(innerProducer.element.st, true, childProducer.separateRegions), innerProducer))

            val outerProducer = new StreamProducer {
              override val length: Option[Code[Int]] = childProducer.length.map(l => ((l.toL + n.toL - 1L) / n.toL).toI)

              override def initialize(cb: EmitCodeBuilder): Unit = {
                cb.assign(n, groupSize.intCode(cb))
                cb.ifx(n < 0, cb._fatal(s"stream grouped: negative size: ", n.toS))
                cb.assign(inOuter, true)
                cb.assign(eos, false)
                cb.assign(xCounter, 0)
              }

              override val elementRegion: Settable[Region] = outerElementRegion
              override val separateRegions: Boolean = childProducer.separateRegions
              override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>
                cb.ifx(eos, {
                  cb.goto(LendOfStream)
                })

                // if we didn't hit EOS from the inner stream and return to outer, iterate through remaining elements
                cb.ifx(!inOuter, {
                  cb.assign(inOuter, true)
                  cb.define(LchildProduceDoneOuter)

                  if (childProducer.separateRegions)
                    cb += childProducer.elementRegion.clearRegion()

                  cb.ifx(xCounter < n, cb.goto(childProducer.LproduceElement))
                })

                cb.goto(LproduceElementDone)
              }
              override val element: EmitCode = innerStreamCode

              override def close(cb: EmitCodeBuilder): Unit = {
                childProducer.close(cb)
                if (childProducer.separateRegions)
                  cb += childProducer.elementRegion.freeRegion()
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

            SStreamCode2(SStream(outerProducer.element.st, required = childStream.st.required, childProducer.separateRegions), outerProducer)
          }
        }

      case StreamZip(as, names, body, behavior) =>
        IEmitCode.multiMapEmitCodes(cb, as.map(a => EmitCode.fromI(mb)(cb => produce(a, cb)))) { childStreams =>

          val producers = childStreams.map(_.asInstanceOf[SStreamCode2].producer)

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
                    if (p.separateRegions)
                      cb.assign(p.elementRegion, eltRegion)
                    else
                      cb.assign(p.elementRegion, outerRegion)
                    p.initialize(cb)
                  }
                }

                override val elementRegion: Settable[Region] = eltRegion

                override val separateRegions: Boolean = producers.exists(_.separateRegions)

                override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>

                  producers.foreach { p =>
                    cb.goto(p.LproduceElement)
                    cb.define(p.LproduceElementDone)
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
                  cb.assign(allEOS, true)

                  producers.foreach { p =>
                    if (p.separateRegions)
                      cb.assign(p.elementRegion, eltRegion)
                    else {
                      cb.assign(p.elementRegion, outerRegion)
                      p.initialize(cb)
                    }
                  }
                }

                override val elementRegion: Settable[Region] = eltRegion

                override val separateRegions: Boolean = producers.exists(_.separateRegions)

                override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>

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
                    if (p.separateRegions)
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

                override val separateRegions: Boolean = producers.exists(_.separateRegions)

                override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>

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

          SStreamCode2(SStream(producer.element.st, childStreams.forall(_.pt.required)), producer)
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
          override val separateRegions: Boolean = true
          override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>
            cb.ifx(shuffle.getValueFinished(), cb.goto(LendOfStream))
            cb.goto(LproduceElementDone)
          }
          override val element: EmitCode = EmitCode.present(mb, PCode(shuffleType.rowDecodedPType, shuffle.getValue(region)))

          override def close(cb: EmitCodeBuilder): Unit = {
            cb += shuffle.getDone()
            cb += shuffle.close()
          }
        }

        IEmitCode.present(cb, SStreamCode2(SStream(producer.element.st, true, producer.separateRegions), producer))

      case ShufflePartitionBounds(idIR, nPartitionsIR) =>

        val region = mb.genFieldThisRef[Region]("shuffle_partition_bounds_region")
        val shuffleLocal = mb.genFieldThisRef[ShuffleClient]("shuffle_partition_bounds_client")
        val shuffle = new ValueShuffleClient(shuffleLocal)

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
          override val separateRegions: Boolean = false
          override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>
            cb.ifx(shuffle.partitionBoundsValueFinished(), cb.goto(LendOfStream))
            cb.goto(LproduceElementDone)
          }
          override val element: EmitCode = EmitCode.present(mb, PCode(shuffleType.keyDecodedPType, shuffle.partitionBoundsValue(region)))

          override def close(cb: EmitCodeBuilder): Unit = {
            cb += shuffle.endPartitionBounds()
            cb += shuffle.close()
          }
        }
        IEmitCode.present(cb, SStreamCode2(SStream(producer.element.st, true, producer.separateRegions), producer))
    }
  }
}

