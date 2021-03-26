package is.hail.expr.ir.streams

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.types.physical.stypes.SType
import is.hail.types.physical.stypes.interfaces.{SIndexableCode, SStream}
import is.hail.types.physical.stypes.primitives.{SInt32, SInt32Code}
import is.hail.types.physical.{PCanonicalStream, PCode, PStream, PStreamCode, PValue}
import is.hail.types.virtual.TStream
import is.hail.utils._

abstract class StreamProducer {
  val length: Option[Code[Int]]

  // the stream's element region, and the consumer should manage memory appropriately according to `separateRegions`
  val elementRegion: Settable[Region]
  val separateRegions: Boolean

  // defined by consumer
  val LproduceElementDone: CodeLabel

  // defined by consumer
  val LendOfStream: CodeLabel

  // defined by producer. Jumps to either LproduceElementDone or LendOfStream
  val LproduceElement: CodeLabel

  val element: EmitCode

  def close(cb: EmitCodeBuilder): Unit
}

abstract class StreamConsumer {
  def init(cb: EmitCodeBuilder, eltType: SType, length: Option[Code[Int]], eltRegion: Settable[Region], separateRegions: Boolean): Unit

  def consumeElement(cb: EmitCodeBuilder, elt: EmitCode): Unit

  def done: CodeLabel

  def finish(cb: EmitCodeBuilder): IEmitCode
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

  private[ir] def feed(emitter: Emit[_],
    cb: EmitCodeBuilder,
    consumer: StreamConsumer,
    streamIR: IR,
    outerRegion: Value[Region],
    env0: Emit.E,
    container: Option[AggContainer]): IEmitCode = {

    val mb = cb.emb

    def emitIR(cb: EmitCodeBuilder, ir: IR, env: Emit.E = env0, region: Value[Region] = outerRegion, container: Option[AggContainer] = container): IEmitCode =
      emitter.emitI(ir, cb, StagedRegion(region), env, container, None)

    def produce(cb: EmitCodeBuilder, streamIR: IR, outerRegion: Value[Region] = outerRegion, env: Emit.E = env0, container: Option[AggContainer] = container): IEmitCode = {
      EmitStream2.produce(emitter, cb, streamIR, outerRegion, env, container)
    }

    def feed(cb: EmitCodeBuilder,
      consumer: StreamConsumer,
      streamIR: IR,
      outerRegion: Value[Region] = outerRegion,
      env0: Emit.E = env0,
      container: Option[AggContainer] = container): IEmitCode = EmitStream2.feed(emitter, cb, consumer, streamIR, outerRegion, env0, container)


    streamIR match {
      case _ =>
        produce(cb, streamIR).flatMap(cb) { case (sc: SStreamCode2) =>

          val producer = sc.producer

          consumer.init(cb, producer.element.st, producer.length, producer.elementRegion, producer.separateRegions)

          cb.goto(producer.LproduceElement)
          cb.define(producer.LproduceElementDone)
          consumer.consumeElement(cb, producer.element)
          cb.goto(producer.LproduceElement)

          cb.define(consumer.done)
          cb.define(producer.LendOfStream)
          producer.close(cb)
          consumer.finish(cb)
        }
    }
  }


  private[ir] def produce(
    emitter: Emit[_],
    cb: EmitCodeBuilder,
    streamIR: IR,
    outerRegion: Value[Region],
    env: Emit.E,
    container: Option[AggContainer]
  ): IEmitCode = {

    val mb = cb.emb


    def emitVoid(ir: IR, cb: EmitCodeBuilder, env: Emit.E = env, region: Value[Region] = outerRegion, container: Option[AggContainer] = container): Unit =
      emitter.emitVoid(cb, ir, StagedRegion(region), env, container, None)

    def emitIR(ir: IR, cb: EmitCodeBuilder, env: Emit.E = env, region: Value[Region] = outerRegion, container: Option[AggContainer] = container): IEmitCode =
      emitter.emitI(ir, cb, StagedRegion(region), env, container, None)

    def produce(cb: EmitCodeBuilder, streamIR: IR, outerRegion: Value[Region] = outerRegion,
      env: Emit.E = env, container: Option[AggContainer] = container): IEmitCode =
      EmitStream2.produce(emitter, cb, streamIR, outerRegion, env, container)


    streamIR match {

      case Ref(name, _typ) =>
        assert(_typ.isInstanceOf[TStream])
        env.lookup(name).toI(cb)

      case In(n, _) =>
        // this, Code[Region], ...
        val param = mb.getEmitParam(2 + n, outerRegion)
        param.st match {
          case _: SStream =>
          case t => throw new RuntimeException(s"parameter ${ 2 + n } is not a stream! t=$t, params=${ mb.emitParamTypes }")
        }
        param.load.toI(cb)


      case ToStream(a, _separateRegions) =>
        val idx = mb.genFieldThisRef[Int]("tostream_idx")
        val regionVar = mb.genFieldThisRef[Region]("tostream_region")

        emitIR(a, cb).map(cb) { case ind: SIndexableCode =>
          val container = ind.memoizeField(cb, "tostream_a")
          cb.assign(idx, -1)

          SStreamCode2(
            SStream(ind.st.elementType, ind.pt.required),
            new StreamProducer {
              override val length: Option[Code[Int]] = Some(container.loadLength())

              override val elementRegion: Settable[Region] = regionVar

              override val separateRegions: Boolean = _separateRegions

              override val LendOfStream: CodeLabel = CodeLabel()
              override val LproduceElementDone: CodeLabel = CodeLabel()
              override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>
                cb.assign(idx, idx + 1)
                cb.ifx(idx >= container.loadLength(), cb.goto(LendOfStream))
                cb.goto(LproduceElementDone)
              }

              val element: EmitCode = EmitCode.fromI(mb)(cb => container.loadElement(cb, idx).typecast[PCode])

              def close(cb: EmitCodeBuilder): Unit = {}
            })

        }

      case StreamRange(startIR, stopIR, stepIR, _separateRegions) =>
        val llen = mb.genFieldThisRef[Long]("sr_llen")
        val len = mb.genFieldThisRef[Int]("sr_len")

        val regionVar = mb.genFieldThisRef[Region]("sr_region")

        emitIR(startIR, cb).flatMap(cb) { startc =>
          emitIR(stopIR, cb).flatMap(cb) { stopc =>
            emitIR(stepIR, cb).map(cb) { stepc =>
              val start = cb.memoizeField(startc, "sr_step").asInt.intCode(cb)
              val stop = cb.memoizeField(stopc, "sr_stop").asInt.intCode(cb)
              val step = cb.memoizeField(stepc, "sr_step").asInt.intCode(cb)
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

              val curr = mb.genFieldThisRef[Int]("streamrange_curr")
              val idx = mb.genFieldThisRef[Int]("streamrange_idx")

              cb.assign(curr, start - step)
              cb.assign(idx, 0)

              val producer: StreamProducer = new StreamProducer {
                override val length: Option[Code[Int]] = Some(len)

                override val elementRegion: Settable[Region] = regionVar

                override val separateRegions: Boolean = _separateRegions

                override val LendOfStream: CodeLabel = CodeLabel()
                override val LproduceElementDone: CodeLabel = CodeLabel()
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
        produce(cb, a)
          .map(cb) { case (childStream: SStreamCode2) =>
            val childProducer = childStream.producer

            val filterEltRegion = mb.genFieldThisRef[Region]("streamfilter_filter_region")

            if (childProducer.separateRegions)
              cb.assign(childProducer.elementRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
            else
              cb.assign(childProducer.elementRegion, outerRegion)

            SStreamCode2(
              childStream.st,
              new StreamProducer {
                override val length: Option[Code[Int]] = None

                override val elementRegion: Settable[Region] = filterEltRegion

                override val separateRegions: Boolean = childProducer.separateRegions

                override val LendOfStream: CodeLabel = childProducer.LendOfStream
                override val LproduceElementDone: CodeLabel = CodeLabel()
                override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>
                  val Lfiltered = CodeLabel()
                  cb.goto(childProducer.LproduceElement)
                  cb.define(childProducer.LproduceElementDone)

                  // false and NA both fail the filter
                  val childProducerElement = cb.memoize(childProducer.element, "streamfilter_cond")
                  emitIR(cond, cb = cb, env = env.bind(name, childProducerElement), region = childProducer.elementRegion)
                    .consume(cb,
                      cb.goto(Lfiltered),
                      { sc =>
                        cb.ifx(sc.asBoolean.boolCode(cb), cb.goto(Lfiltered))
                      })

                  if (separateRegions)
                    cb += childProducer.elementRegion.addReferenceTo(filterEltRegion)

                  cb.goto(LproduceElementDone)

                  cb.define(Lfiltered)
                  if (separateRegions)
                    cb += childProducer.elementRegion.clearRegion()
                  cb.goto(childProducer.LproduceElement)

                }

                val element: EmitCode = childProducer.element

                def close(cb: EmitCodeBuilder): Unit = {
                  childProducer.close(cb)
                  if (separateRegions)
                    cb += childProducer.elementRegion.freeRegion()
                }

              })
          }

      case StreamTake(a, num) =>
        produce(cb, a)
          .flatMap(cb) { case (childStream: SStreamCode2) =>
            emitIR(num, cb).map(cb) { case (num: SInt32Code) =>
              val childProducer = childStream.producer
              val n = mb.genFieldThisRef[Int]("stream_take_n")
              cb.assign(n, num.intCode(cb))
              cb.ifx(n < 0, cb._fatal(s"stream take: negative number of elements to take: ", n.toS))

              val idx = mb.genFieldThisRef[Int]("stream_take_idx")

              val producer = new StreamProducer {
                override val length: Option[Code[Int]] = childProducer.length.map(_.min(n))
                override val elementRegion: Settable[Region] = childProducer.elementRegion
                override val separateRegions: Boolean = childProducer.separateRegions
                override val LproduceElementDone: CodeLabel = CodeLabel()
                override val LendOfStream: CodeLabel = CodeLabel()
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
        produce(cb, a)
          .flatMap(cb) { case (childStream: SStreamCode2) =>
            emitIR(num, cb).map(cb) { case (num: SInt32Code) =>
              val childProducer = childStream.producer
              val n = mb.genFieldThisRef[Int]("stream_drop_n")
              cb.assign(n, num.intCode(cb))
              cb.ifx(n < 0, cb._fatal(s"stream drop: negative number of elements to drop: ", n.toS))

              val idx = mb.genFieldThisRef[Int]("stream_drop_idx")

              val producer = new StreamProducer {
                override val length: Option[Code[Int]] = childProducer.length.map(l => (l - n).max(0))
                override val elementRegion: Settable[Region] = childProducer.elementRegion
                override val separateRegions: Boolean = childProducer.separateRegions
                override val LproduceElementDone: CodeLabel = CodeLabel()
                override val LendOfStream: CodeLabel = CodeLabel()
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
        produce(cb, a)
          .map(cb) { case (childStream: SStreamCode2) =>
            val childProducer = childStream.producer

            val bodyResult = EmitCode.fromI(mb) { cb =>
              val childProducerElement = cb.memoize(childProducer.element, "streammap_element")
              emitIR(body,
                cb = cb,
                env = env.bind(name, childProducerElement),
                region = childProducer.elementRegion)
            }

            val producer: StreamProducer = new StreamProducer {
              override val length: Option[Code[Int]] = childProducer.length

              override val elementRegion: Settable[Region] = childProducer.elementRegion

              override val separateRegions: Boolean = childProducer.separateRegions

              override val LendOfStream: CodeLabel = childProducer.LendOfStream
              override val LproduceElementDone: CodeLabel = CodeLabel()
              override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>
                cb.goto(childProducer.LproduceElement)
                cb.define(childProducer.LproduceElementDone)
                cb.goto(LproduceElementDone)
              }

              val element: EmitCode = bodyResult

              def close(cb: EmitCodeBuilder): Unit = childProducer.close(cb)
            }

            SStreamCode2(
              SStream(bodyResult.st, required = childStream.st.required),
              producer
            )
          }

      case x@StreamScan(childIR, zeroIR, accName, eltName, bodyIR) =>
        produce(cb, childIR).map(cb) { case (childStream: SStreamCode2) =>
          val childProducer = childStream.producer


          val accValueAccRegion = mb.newEmitField(x.accPType)
          val accValueEltRegion = mb.newEmitField(x.accPType)

          val accRegion = mb.genFieldThisRef[Region]("streamscan_acc_region")
          if (childProducer.separateRegions)
            cb.assign(accRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))

          // accRegion is unused if separateRegions is false


          val first = mb.genFieldThisRef[Boolean]("streamscan_has_pulled")
          cb.assign(first, true)

          val producer = new StreamProducer {
            override val length: Option[Code[Int]] = childProducer.length.map(_ + const(1))

            override val elementRegion: Settable[Region] = childProducer.elementRegion

            override val separateRegions: Boolean = childProducer.separateRegions

            override val LendOfStream: CodeLabel = childProducer.LendOfStream
            override val LproduceElementDone: CodeLabel = CodeLabel()
            override val LproduceElement: CodeLabel = mb.defineHangingLabel { cb =>

              val LcopyAndReturn = CodeLabel()

              cb.ifx(first, {

                cb.assign(first, false)
                cb.assign(accValueEltRegion, emitIR(zeroIR, cb, region = elementRegion))

                cb.goto(LcopyAndReturn)
              })


              cb.goto(childProducer.LproduceElement)
              cb.define(childProducer.LproduceElementDone)

              if (separateRegions) {
                // deep copy accumulator into element region, then clear accumulator region
                cb.assign(accValueEltRegion, accValueAccRegion.map(_.castTo(cb, childProducer.elementRegion, x.accPType, deepCopy = true)))
                cb += accRegion.clearRegion()
              }

              val childEltValue = cb.memoizeField(childProducer.element, "scan_child_elt")
              cb.assign(accValueEltRegion,
                emitIR(bodyIR, cb, env = env.bind((accName, accValueEltRegion), (eltName, childEltValue)), region = childProducer.elementRegion)
                  .map(cb)(pc => pc.castTo(cb, childProducer.elementRegion, x.accPType, deepCopy = false)))

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
          SStreamCode2(SStream(accValueEltRegion.st, childStream.st.required, producer.separateRegions), producer)
        }

      case RunAggScan(child, name, init, seqs, result, states) =>
        val (newContainer, aggSetup, aggCleanup) = AggContainer.fromMethodBuilder(states.toArray, mb, "run_agg_scan")

        produce(cb, child).map(cb) { case (childStream: SStreamCode2) =>
          val childProducer = childStream.producer

          val childEltField = mb.newEmitField("runaggscan_child_elt", childProducer.element.pt)
          val bodyEnv = env.bind(name -> childEltField)
          val bodyResult = EmitCode.fromI(mb)(cb => emitIR(result, cb = cb, region = childProducer.elementRegion,
            env = bodyEnv, container = Some(newContainer)))
          val bodyResultField = mb.newEmitField("runaggscan_result_elt", bodyResult.pt)

          cb += aggSetup
          emitVoid(init, cb = cb, region = outerRegion, container = Some(newContainer))

          val producer = new StreamProducer {
            override val length: Option[Code[Int]] = childProducer.length
            override val elementRegion: Settable[Region] = childProducer.elementRegion
            override val separateRegions: Boolean = childProducer.separateRegions
            override val LendOfStream: CodeLabel = childProducer.LendOfStream
            override val LproduceElementDone: CodeLabel = CodeLabel()
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

          SStreamCode2(SStream(producer.element.st, childStream.st.required, producer.separateRegions), producer)
        }

      case StreamFlatMap(a, name, body) =>
        produce(cb, a).map(cb) { case (outerStream: SStreamCode2) =>
          val outerProducer = outerStream.producer

          // variables used in control flow
          val first = mb.genFieldThisRef[Boolean]("flatmap_first")
          val innerUnclosed = mb.genFieldThisRef[Boolean]("flatmap_inner_unclosed")

          cb.assign(first, true)

          if (outerProducer.separateRegions)
            cb.assign(outerProducer.elementRegion, Region.stagedCreate(Region.REGULAR, cb.emb.ecb.pool()))
          else
            cb.assign(outerProducer.elementRegion, outerRegion)

          val innerStreamEmitCode = EmitCode.fromI(mb) { cb =>
            val outerProducerValue = cb.memoize(outerProducer.element, "flatmap_outer_element")
            emitIR(body,
              cb = cb,
              env = env.bind(name, outerProducerValue),
              region = outerProducer.elementRegion)
          }

          // grabbing emitcode.pv weird pattern but should be safe
          val SStreamCode2(_, innerProducer) = innerStreamEmitCode.pv

          val producer = new StreamProducer {
            override val length: Option[Code[Int]] = None

            override val elementRegion: Settable[Region] = innerProducer.elementRegion

            override val separateRegions: Boolean = innerProducer.separateRegions || outerProducer.separateRegions

            override val LendOfStream: CodeLabel = outerProducer.LendOfStream
            override val LproduceElementDone: CodeLabel = CodeLabel()
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

          SStreamCode2(
            SStream(innerProducer.element.st, required = outerStream.st.required),
            producer
          )
        }

      case StreamZip(as, names, body, behavior) =>
        IEmitCode.multiMapEmitCodes(cb, as.map(a => EmitCode.fromI(mb)(cb => produce(cb, a)))) { childStreams =>

          val producers = childStreams.map(_.asInstanceOf[SStreamCode2].producer)

          assert(names.length == producers.length)

          val producer: StreamProducer = behavior match {
            case behavior@(ArrayZipBehavior.TakeMinLength | ArrayZipBehavior.AssumeSameLength) =>
              val vars = names.zip(producers).map { case (name, p) => mb.newEmitField(name, p.element.pt) }

              val eltRegion = mb.genFieldThisRef[Region]("streamzip_eltregion")
              val bodyCode = EmitCode.fromI(mb)(cb => emitIR(body, cb, env.bind(names.zip(vars): _*), eltRegion))

              producers.foreach { p =>
                if (p.separateRegions)
                  cb.assign(p.elementRegion, eltRegion)
                else
                  cb.assign(p.elementRegion, outerRegion)
              }

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

                override val elementRegion: Settable[Region] = eltRegion

                override val separateRegions: Boolean = producers.exists(_.separateRegions)

                override val LendOfStream: CodeLabel = CodeLabel()
                override val LproduceElementDone: CodeLabel = CodeLabel()
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
              val bodyCode = EmitCode.fromI(mb)(cb => emitIR(body, cb, env.bind(names.zip(vars): _*), eltRegion))

              val anyEOS = mb.genFieldThisRef[Boolean]("zip_any_eos")
              val allEOS = mb.genFieldThisRef[Boolean]("zip_all_eos")

              cb.assign(anyEOS, false)
              cb.assign(allEOS, true)

              producers.foreach { p =>
                if (p.separateRegions)
                  cb.assign(p.elementRegion, eltRegion)
                else
                  cb.assign(p.elementRegion, outerRegion)
              }


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

                override val elementRegion: Settable[Region] = eltRegion

                override val separateRegions: Boolean = producers.exists(_.separateRegions)

                override val LendOfStream: CodeLabel = CodeLabel()
                override val LproduceElementDone: CodeLabel = CodeLabel()
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
              val bodyCode = EmitCode.fromI(mb)(cb => emitIR(body, cb, env.bind(names.zip(vars): _*), eltRegion))

              val eosPerStream = producers.indices.map(i => mb.genFieldThisRef[Boolean](s"zip_eos_$i"))
              val nEOS = mb.genFieldThisRef[Int]("zip_n_eos")

              eosPerStream.foreach { eos =>
                cb.assign(eos, false)
              }
              cb.assign(nEOS, 0)

              producers.foreach { p =>
                if (p.separateRegions)
                  cb.assign(p.elementRegion, eltRegion)
                else
                  cb.assign(p.elementRegion, outerRegion)
              }

              new StreamProducer {
                override val length: Option[Code[Int]] = producers.map(_.length).reduceLeft(_.liftedZip(_).map {
                  case (l1, l2) => l1.max(l2)
                })

                override val elementRegion: Settable[Region] = eltRegion

                override val separateRegions: Boolean = producers.exists(_.separateRegions)

                override val LendOfStream: CodeLabel = CodeLabel()
                override val LproduceElementDone: CodeLabel = CodeLabel()
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
    }
  }
}

