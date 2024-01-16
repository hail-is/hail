package is.hail.expr.ir.streams

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, EmitMethodBuilder, IEmitCode, IR, NDArrayMap, NDArrayMap2, Ref, RunAggScan, StagedArrayBuilder, StreamFilter, StreamFlatMap, StreamFold, StreamFold2, StreamFor, StreamJoinRightDistinct, StreamMap, StreamScan, StreamZip, StreamZipJoin}
import is.hail.expr.ir.orderings.StructOrdering
import is.hail.types.physical.{PCanonicalArray, PCanonicalStruct}
import is.hail.types.physical.stypes.SingleCodeType
import is.hail.types.physical.stypes.interfaces.{NoBoxLongIterator, SIndexableValue}
import is.hail.utils._

object StreamUtils {

  def storeNDArrayElementsAtAddress(
    cb: EmitCodeBuilder,
    stream: StreamProducer,
    destRegion: Value[Region],
    addr: Value[Long],
    errorId: Int,
  ): Unit = {
    val currentElementIndex = cb.newLocal[Long]("store_ndarray_elements_stream_current_index", 0)
    val currentElementAddress =
      cb.newLocal[Long]("store_ndarray_elements_stream_current_addr", addr)
    val elementType = stream.element.emitType.storageType
    val elementByteSize = elementType.byteSize

    var push: (EmitCodeBuilder, IEmitCode) => Unit = null
    stream.memoryManagedConsume(
      destRegion,
      cb,
      setup = { cb =>
        push = { case (cb, iec) =>
          iec.consume(
            cb,
            cb._throw(Code.newInstance[HailException, String, Int](
              "Cannot construct an ndarray with missing values.",
              errorId,
            )),
            sc =>
              elementType.storeAtAddress(cb, currentElementAddress, destRegion, sc, deepCopy = true),
          )
          cb.assign(currentElementIndex, currentElementIndex + 1)
          cb.assign(currentElementAddress, currentElementAddress + elementByteSize)
        }
      },
    )(cb => push(cb, stream.element.toI(cb)))
  }

  def toArray(
    cb: EmitCodeBuilder,
    stream: StreamProducer,
    destRegion: Value[Region],
  ): SIndexableValue = {
    val mb = cb.emb

    val xLen = mb.newLocal[Int]("sta_len")
    val aTyp = PCanonicalArray(stream.element.emitType.storageType, true)
    stream.length match {
      case None =>
        val vab = new StagedArrayBuilder(
          cb,
          SingleCodeType.fromSType(stream.element.st),
          stream.element.required,
          0,
        )
        writeToArrayBuilder(cb, stream, vab, destRegion)
        cb.assign(xLen, vab.size)

        aTyp.constructFromElements(cb, destRegion, xLen, deepCopy = false) { (cb, i) =>
          vab.loadFromIndex(cb, destRegion, i)
        }

      case Some(computeLen) =>
        var pushElem: (EmitCodeBuilder, IEmitCode) => Unit = null
        var finish: (EmitCodeBuilder) => SIndexableValue = null

        stream.memoryManagedConsume(
          destRegion,
          cb,
          setup = { cb =>
            cb.assign(xLen, computeLen(cb))
            val (_pushElem, _finish) = aTyp.constructFromFunctions(
              cb,
              destRegion,
              xLen,
              deepCopy = stream.requiresMemoryManagementPerElement,
            )
            pushElem = _pushElem
            finish = _finish
          },
        )(cb => pushElem(cb, stream.element.toI(cb)))

        finish(cb)
    }
  }

  def writeToArrayBuilder(
    cb: EmitCodeBuilder,
    stream: StreamProducer,
    ab: StagedArrayBuilder,
    destRegion: Value[Region],
  ): Unit = {
    stream.memoryManagedConsume(
      destRegion,
      cb,
      setup = { cb =>
        ab.clear(cb)
        stream.length match {
          case Some(computeLen) => ab.ensureCapacity(cb, computeLen(cb))
          case None => ab.ensureCapacity(cb, 16)
        }

      },
    ) { cb =>
      stream.element.toI(cb).consume(
        cb,
        ab.addMissing(cb),
        sc =>
          ab.add(
            cb,
            ab.elt.coerceSCode(
              cb,
              sc,
              destRegion,
              deepCopy = stream.requiresMemoryManagementPerElement,
            ).code,
          ),
      )
    }
  }

  private[ir] def multiplicity(root: IR, refName: String): Int = {
    var uses = 0

    // assumes no name collisions, a bit hacky...
    def traverse(ir: IR, mult: Int): Unit = ir match {
      case Ref(name, _) => if (refName == name) uses += mult
      case StreamMap(a, _, b) => traverse(a, mult); traverse(b, 2)
      case StreamFilter(a, _, b) => traverse(a, mult); traverse(b, 2)
      case StreamFlatMap(a, _, b) => traverse(a, mult); traverse(b, 2)
      case StreamJoinRightDistinct(l, r, _, _, _, _, j, _) =>
        traverse(l, mult); traverse(r, mult); traverse(j, 2)
      case StreamScan(a, z, _, _, b) =>
        traverse(a, mult); traverse(z, 2); traverse(b, 2)
      case RunAggScan(a, _, i, s, r, _) =>
        traverse(a, mult); traverse(i, 2); traverse(s, 2); traverse(r, 2)
      case StreamZipJoin(as, _, _, _, f) =>
        as.foreach(traverse(_, mult)); traverse(f, 2)
      case StreamZip(as, _, body, _, _) =>
        as.foreach(traverse(_, mult)); traverse(body, 2)
      case StreamFold(a, zero, _, _, body) =>
        traverse(a, mult); traverse(zero, mult); traverse(body, 2)
      case StreamFold2(a, accs, _, seqs, res) =>
        traverse(a, mult)
        accs.foreach { case (_, acc) => traverse(acc, mult) }
        seqs.foreach(traverse(_, 2))
        traverse(res, 2)
      case StreamFor(a, _, body) =>
        traverse(a, mult); traverse(body, 2)
      case NDArrayMap(a, _, body) =>
        traverse(a, mult); traverse(body, 2)
      case NDArrayMap2(l, r, _, _, body, _) =>
        traverse(l, mult); traverse(r, mult); traverse(body, 2)

      case _ => ir.children.foreach {
          case child: IR => traverse(child, mult)
          case _ =>
        }
    }

    traverse(root, 1)
    uses min 2
  }

  def isIterationLinear(ir: IR, refName: String): Boolean =
    multiplicity(ir, refName) <= 1

  abstract class StreamMultiMergeBase(
    key: IndexedSeq[String],
    unifiedType: PCanonicalStruct,
    val k: Value[Int],
    mb: EmitMethodBuilder[_],
  ) extends StreamProducer {
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

    val region = mb.genFieldThisRef[Region]("smm_region")

    /** The ordering function in StreamMultiMerge should use missingFieldsEqual=false to be
      * consistent with other nodes that deal with struct keys. When keys compare equal, the earlier
      * index (in the list of stream children) should win. These semantics extend to missing key
      * fields, which requires us to compile two orderings (l/r and r/l) to maintain the abilty to
      * take from the left when key fields are missing.
      */
    def comp(cb: EmitCodeBuilder, li: Code[Int], lv: Code[Long], ri: Code[Int], rv: Code[Long])
      : Code[Boolean] = {
      val l = unifiedType.loadCheapSCode(cb, lv).asBaseStruct.subset(key: _*)
      val r = unifiedType.loadCheapSCode(cb, rv).asBaseStruct.subset(key: _*)
      val ord1 = StructOrdering.make(
        l.asBaseStruct.st,
        r.asBaseStruct.st,
        cb.emb.ecb,
        missingFieldsEqual = false,
      )
      val ord2 = StructOrdering.make(
        r.asBaseStruct.st,
        l.asBaseStruct.st,
        cb.emb.ecb,
        missingFieldsEqual = false,
      )
      val b = cb.newLocal[Boolean]("stream_merge_comp_result")
      cb.if_(
        li < ri,
        cb.assign(b, ord1.compareNonnull(cb, l, r) <= 0),
        cb.assign(b, ord2.compareNonnull(cb, r, l) > 0),
      )
      b
    }

    def implInit(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit

    final def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
      implInit(cb, outerRegion)
      cb.assign(bracket, Code.newArray[Int](k))
      cb.assign(heads, Code.newArray[Long](k))
      cb.for_(cb.assign(i, 0), i < k, cb.assign(i, i + 1), cb += (bracket(i) = -1))
      cb.assign(i, 0)
      cb.assign(winner, 0)
    }

    def implClose(cb: EmitCodeBuilder): Unit

    final def close(cb: EmitCodeBuilder): Unit = {
      implClose(cb)
      cb.assign(bracket, Code._null)
      cb.assign(heads, Code._null)
    }

    final override val elementRegion: Settable[Region] = region

    final override val element: EmitCode =
      EmitCode.fromI(mb)(cb => IEmitCode.present(cb, unifiedType.loadCheapSCode(cb, heads(winner))))

  }

  def multiMergeIterators(
    cb: EmitCodeBuilder,
    reqMemManagementArray: Either[Array[Boolean], Boolean],
    iterators: Value[Array[NoBoxLongIterator]],
    key: IndexedSeq[String],
    unifiedType: PCanonicalStruct,
  ): StreamProducer = {

    val mb = cb.emb

    val regionArray = mb.genFieldThisRef[Array[Region]]("smm_region_array")

    val (memManagementArrayField, initMemManagement) = reqMemManagementArray match {
      case Left(arr) =>
        val fd = mb.genFieldThisRef[Array[Boolean]]("memManagement")
        fd -> ((cb: EmitCodeBuilder) => (cb.assign(fd, mb.getObject[Array[Boolean]](arr))))
      case Right(_) => (null, ((cb: EmitCodeBuilder) => ()))
    }

    def lookupMemoryManagementByIndex(cb: EmitCodeBuilder, idx: Code[Int]): Value[Boolean] =
      reqMemManagementArray match {
        case Left(_) => cb.memoize(memManagementArrayField(idx))
        case Right(b) => b
      }

    new StreamMultiMergeBase(key, unifiedType, cb.memoizeField(iterators.length()), mb) {

      def forEachIterator(
        cb: EmitCodeBuilder
      )(
        f: (EmitCodeBuilder, Value[Int], Value[NoBoxLongIterator]) => Unit
      ) = {
        val idx = cb.newLocal[Int]("idx", 0)
        cb.while_(
          idx < k, {
            val iter = cb.memoize(iterators(idx))
            f(cb, idx, iter)
            cb.assign(idx, idx + 1)
          },
        )
      }

      override def method: EmitMethodBuilder[_] = mb
      override val length: Option[EmitCodeBuilder => Code[Int]] = None

      override def implInit(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
        cb.assign(regionArray, Code.newArray[Region](k))

        forEachIterator(cb) { case (cb, idx, iter) =>
          val reqMM = lookupMemoryManagementByIndex(cb, idx)
          val eltRegion = cb.newLocal[Region]("eltRegion")
          cb.if_(
            reqMM,
            cb.assign(eltRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool())),
            cb.assign(eltRegion, outerRegion),
          )
          cb += iter.invoke[Region, Region, Unit]("init", outerRegion, eltRegion)
          cb += regionArray.update(idx, eltRegion)
        }
        initMemManagement(cb)
      }

      override val requiresMemoryManagementPerElement: Boolean = reqMemManagementArray match {
        case Left(a) => a.exists(b => b)
        case Right(b) => b
      }

      override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
        val LrunMatch = CodeLabel()
        val LpullChild = CodeLabel()
        val LloopEnd = CodeLabel()

        cb.define(LpullChild)

        cb.if_(winner >= k, cb.goto(LendOfStream))
        val winnerIter = cb.memoize(iterators(winner))
        val next = cb.memoize(winnerIter.invoke[Long]("next"))
        cb.if_(
          winnerIter.invoke[Boolean]("eos"), {
            cb.assign(matchIdx, (winner + k) >>> 1)
            cb.assign(winner, k)
            cb.goto(LrunMatch)
          },
        )

        cb.if_(next ceq 0L, cb._fatal("stream multi merge: elements cannot be missing"))
        cb += heads.update(winner, next)
        cb.assign(matchIdx, (winner + k) >>> 1)
        cb.goto(LrunMatch)

        cb.define(LrunMatch)
        cb.assign(challenger, bracket(matchIdx))
        cb.if_(matchIdx.ceq(0) || challenger.ceq(-1), cb.goto(LloopEnd))

        val LafterChallenge = CodeLabel()
        cb.if_(
          challenger.cne(k), {
            val Lwon = CodeLabel()
            cb.if_(winner.ceq(k), cb.goto(Lwon))
            cb.if_(
              comp(cb, challenger, heads(challenger), winner, heads(winner)),
              cb.goto(Lwon),
              cb.goto(LafterChallenge),
            )

            cb.define(Lwon)
            cb += (bracket(matchIdx) = winner)
            cb.assign(winner, challenger)
          },
        )
        cb.define(LafterChallenge)

        cb.assign(matchIdx, matchIdx >>> 1)
        cb.goto(LrunMatch)

        cb.define(LloopEnd)

        cb.if_(
          matchIdx.ceq(0), {
            // 'winner' is smallest of all k heads. If 'winner' = k, all heads
            // must be k, and all streams are exhausted.
            cb.if_(
              winner.ceq(k),
              cb.goto(LendOfStream), {
                // we have a winner
                cb.if_(
                  lookupMemoryManagementByIndex(cb, winner), {
                    val winnerRegion = cb.newLocal[Region]("smm_winner_region", regionArray(winner))
                    cb += elementRegion.trackAndIncrementReferenceCountOf(winnerRegion)
                    cb += winnerRegion.clearRegion()
                  },
                )
                cb.goto(LproduceElementDone)
              },
            )
          }, {
            cb += (bracket(matchIdx) = winner)
            cb.assign(i, i + 1)
            cb.assign(winner, i)
            cb.goto(LpullChild)
          },
        )
      }

      override def implClose(cb: EmitCodeBuilder): Unit =
        forEachIterator(cb) { case (cb, idx, iter) =>
          cb.if_(lookupMemoryManagementByIndex(cb, idx), cb += regionArray(idx).invalidate())
          cb += iter.invoke[Unit]("close")
        }
    }
  }
}
