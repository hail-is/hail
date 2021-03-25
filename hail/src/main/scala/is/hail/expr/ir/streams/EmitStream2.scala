package is.hail.expr.ir.streams

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.types.physical.stypes.SType
import is.hail.types.physical.stypes.interfaces.{SIndexableCode, SStream}
import is.hail.types.physical.stypes.primitives.{SInt32, SInt32Code}
import is.hail.types.physical.{PCanonicalStream, PCode, PStream, PStreamCode, PValue}
import is.hail.utils._

abstract class StreamProducer {
  val length: Option[Code[Int]]

  // the stream's element region, and the consumer should manage memory appropriately according to `separateRegions`
  val elementRegion: Settable[Region]
  val separateRegions: Boolean

  // defined by producer, producer jumps to LproduceElementDone
  val LproduceElement: CodeLabel

  // defined by consumer, producer can jump to
  val LproduceElementDone: CodeLabel

  val element: EmitCode

  // defined by consumer, producer can jump to
  val LendOfStream: CodeLabel

  def close(cb: EmitCodeBuilder): Unit
}

abstract class StreamConsumer {
  def init(cb: EmitCodeBuilder, eltType: SType, length: Option[Code[Int]], eltRegion: Settable[Region], separateRegions: Boolean): Unit

  def consumeElement(cb: EmitCodeBuilder, elt: EmitCode): Unit

  def done: CodeLabel

  def finish(cb: EmitCodeBuilder): IEmitCode
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


    def emitIR(ir: IR, cb: EmitCodeBuilder = cb, env: Emit.E = env, region: Value[Region] = outerRegion, container: Option[AggContainer] = container): IEmitCode =
      emitter.emitI(ir, cb, StagedRegion(region), env, container, None)


    def produce(cb: EmitCodeBuilder, streamIR: IR, outerRegion: Value[Region] = outerRegion,
      env: Emit.E = env, container: Option[AggContainer] = container): IEmitCode =
      EmitStream2.produce(emitter, cb, streamIR, outerRegion, env, container)


    streamIR match {
      case ToStream(a, _separateRegions) =>
        val idx = mb.genFieldThisRef[Int]("tostream_idx")
        val regionVar = mb.genFieldThisRef[Region]("tostream_region")

        emitIR(a).map(cb) { case ind: SIndexableCode =>
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

        emitIR(startIR).flatMap(cb) { startc =>
          emitIR(stopIR).flatMap(cb) { stopc =>
            emitIR(stepIR).map(cb) { stepc =>
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

                def close(cb: EmitCodeBuilder): Unit = childProducer.close(cb)

              })
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


//      def writeToArray(
//        cb: EmitCodeBuilder,
//        //      stream: SStreamCode,
//        ab: StagedArrayBuilder,
//        destRegion: Value[Region]
//      ): StreamConsumer = {
//
//
//        new StreamConsumer {
//          private[this] var elementType: SType = _
//          private[this] var ab: StagedArrayBuilder = _
//
//          def initType(elementType: SType): Unit = this.elementType = elementType
//
//          def init(cb: EmitCodeBuilder, len: Option[Code[Int]]): Unit = {
//
//
//            cb += ab.clear
//            cb += ab.ensureCapacity(len.getOrElse(const(16)))
//          }
//
//          def consumeElement(cb: EmitCodeBuilder, element: EmitCode, eltRegion: Option[Settable[Region]]): Unit = {
//            element.toI(cb).consume(cb,
//              cb += ab.addMissing(),
//              { pc =>
//                val pcCopy = if (eltRegion.isDefined) pc.copyToRegion(cb, destRegion) else pc
//                cb += ab.add(SingleCodePCode.fromPCode(cb, pcCopy, destRegion).code)
//              }
//            )
//            eltRegion.foreach { region =>
//              cb += region.clearRegion()
//            }
//          }
//
//          val done: CodeLabel = CodeLabel()
//
//          def close(cb: EmitCodeBuilder): Unit = {}
//
//          def finish(cb: EmitCodeBuilder): IEmitCode = {
//
//          }
//        }
//
//        if (stream.separateRegions)
//          cb.assign(stream.region, Region.stagedCreate(Region.REGULAR, destRegion.getPool()))
//        else
//          cb.assign(stream.region, destRegion)
//        cb += stream.setup
//        cb += ab.clear
//        cb += ab.ensureCapacity(stream.length.getOrElse(const(16)))
//
//        stream.stream.forEachI(cb) { case (cb, elt) =>
//          elt.toI(cb)
//            .consume(cb,
//              cb += ab.addMissing(),
//              { pc =>
//                val pcCopy = if (stream.separateRegions) pc.copyToRegion(cb, destRegion) else pc
//                cb += ab.add(SingleCodePCode.fromPCode(cb, pcCopy, destRegion).code)
//              }
//            )
//          if (stream.separateRegions)
//            cb += stream.region.clear()
//        }
//      }
//
//      def toArray(
//        cb: EmitCodeBuilder,
//        aTyp: PCanonicalArray,
//        stream: SStreamCode,
//        destRegion: Value[Region]
//      ): PCode = {
//        val mb = cb.emb
//        val xLen = mb.newLocal[Int]("sta_len")
//        stream.length match {
//          case None =>
//            val vab = new StagedArrayBuilder(aTyp.elementType, mb, 0)
//            write(cb, stream, vab, destRegion)
//            cb.assign(xLen, vab.size)
//
//            aTyp.constructFromElements(cb, destRegion, xLen, deepCopy = false) { (cb, i) =>
//              IEmitCode(cb, vab.isMissing(i), PCode(aTyp.elementType, vab(i)))
//            }
//
//          case Some(len) =>
//            if (stream.separateRegions)
//              cb.assign(stream.region, Region.stagedCreate(Region.REGULAR, destRegion.getPool()))
//            else
//              cb.assign(stream.region, destRegion)
//            cb += stream.setup
//            cb.assign(xLen, len)
//
//            val (push, finish) = aTyp.constructFromFunctions(cb, destRegion, xLen, deepCopy = stream.separateRegions)
//            stream.stream.forEachI(cb) { case (cb, elt) =>
//              push(cb, elt.toI(cb))
//              if (stream.separateRegions)
//                cb += stream.region.clear()
//            }
//            finish(cb)
//        }
//      }

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

//
//object EmitStream2 {
//
//  def iota(mb: EmitMethodBuilder[_], start: Code[Int], step: Code[Int]): StreamProducer = {
//    val lstep = mb.genFieldThisRef[Int]("sr_lstep")
//    val cur = mb.genFieldThisRef[Int]("sr_cur")
//
//    StreamProducer(
//      Code(lstep := step, cur := start - lstep).start,
//      Code._empty.start,
//      Code._empty.start,
//      EmitCode.present(mb, new SInt32Code(true, cur)),
//      CodeLabel()
//    )
//    unfold[Code[Int]](
//      f = {
//        case (_ctx, k) =>
//          implicit val ctx = _ctx
//          Code(cur := cur + lstep, k(COption.present(cur)))
//      },
//      setup = Some())
//  }
//
//  private[ir] def emit(
//    emitter: Emit[_],
//    streamIR0: IR,
//    mb: EmitMethodBuilder[_],
//    outerRegion: Value[Region],
//    env0: Emit.E,
//    container: Option[AggContainer]
//  ): EmitCode = {
//
//    def _emitStream(cb: EmitCodeBuilder, streamIR: IR, outerRegion: Value[Region], env: Emit.E): IEmitCode = {
//      assert(cb.isOpenEnded)
//
//      def emitVoidIR(ir: IR, cb: EmitCodeBuilder = cb, env: Emit.E = env, region: Value[Region] = outerRegion, container: Option[AggContainer] = container): Unit =
//        emitter.emitVoid(cb, ir, mb, region, env, container, None)
//
//      def emitStream(streamIR: IR, cb: EmitCodeBuilder = cb, env: Emit.E = env, outerRegion: Value[Region] = outerRegion): IEmitCode =
//        _emitStream(cb, streamIR, outerRegion, env)
//
//      def emitIR(ir: IR, cb: EmitCodeBuilder = cb, env: Emit.E = env, region: Value[Region] = outerRegion, container: Option[AggContainer] = container): IEmitCode =
//        emitter.emitI(ir, cb, region, env, container, None)
//
//
//      val result: IEmitCode = streamIR match {
//        case x@NA(_) =>
//          IEmitCode.missing(cb, x.pType.defaultValue(mb))
//
//
//        case x@StreamRange(startIR, stopIR, stepIR, separateRegions) =>
//          val llen = mb.genFieldThisRef[Long]("sr_llen")
//          val len = mb.genFieldThisRef[Int]("sr_len")
//
//          emitIR(startIR).flatMap(cb) { startc =>
//            emitIR(stopIR).flatMap(cb) { stopc =>
//              emitIR(stepIR).map(cb) { stepc =>
//                val start = cb.memoizeField(startc, "sr_step")
//                val stop = cb.memoizeField(stopc, "sr_stop")
//                val step = cb.memoizeField(stepc, "sr_step")
//                cb.ifx(step.asInt.intCode(cb) ceq const(0), cb._fatal("Array range cannot have step size 0."))
//                cb.ifx(step.asInt.intCode(cb) < const(0), {
//                  cb.ifx(start.asInt.intCode(cb).toL <= stop.asInt.intCode(cb).toL, {
//                    cb.assign(llen, 0L)
//                  }, {
//                    cb.assign(llen, (start.asInt.intCode(cb).toL - stop.asInt.intCode(cb).toL - 1L) / (-step.asInt.intCode(cb).toL) + 1L)
//                  })
//                }, {
//                  cb.ifx(start.asInt.intCode(cb).toL >= stop.asInt.intCode(cb).toL, {
//                    cb.assign(llen, 0L)
//                  }, {
//                    cb.assign(llen, (stop.asInt.intCode(cb).toL - start.asInt.intCode(cb).toL - 1L) / step.asInt.intCode(cb).toL + 1L)
//                  })
//                })
//                cb.ifx(llen > const(Int.MaxValue.toLong), {
//                  cb._fatal("Array range cannot have more than MAXINT elements.")
//                })
//
//                val region = cb.emb.genFieldThisRef[Region]("streamrange_region")
//                new SStreamCode2(
//                  SStream(SInt32(true), required = true),
//                  region,
//                  range(mb, start.asInt.intCode(cb), step.asInt.intCode(cb), len)
//                    .map(i => EmitCode.present(mb, new SInt32Code(true, i))),
//                  separateRegions = separateRegions,
//                  Code._empty,
//                  Some(len)
//                )
//              }
//            }
//          }
//
//
//      }
//    }
//  }
