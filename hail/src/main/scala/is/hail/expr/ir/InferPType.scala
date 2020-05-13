package is.hail.expr.ir

import is.hail.types.physical._
import is.hail.types.virtual.{TNDArray, TVoid}
import is.hail.utils._
import is.hail.HailContext
import is.hail.types.{BaseTypeWithRequiredness, RDict, RIterable, TypeWithRequiredness}

object InferPType {

  def clearPTypes(x: BaseIR): Unit = {
    x match {
      case x: IR =>
        x._pType = null
      case _ =>
    }
    x.children.foreach(clearPTypes)
  }

  // does not unify physical arg types if multiple nested seq/init ops appear; instead takes the first. The emitter checks equality.
  def computePhysicalAgg(virt: AggStateSignature, initsAB: ArrayBuilder[RecursiveArrayBuilderElement[InitOp]],
    seqAB: ArrayBuilder[RecursiveArrayBuilderElement[SeqOp]]): AggStatePhysicalSignature = {
    val inits = initsAB.result()
    val seqs = seqAB.result()
    assert(inits.nonEmpty)
    assert(seqs.nonEmpty)
    virt.default match {
      case AggElementsLengthCheck() =>
        assert(inits.length == 1)
        assert(seqs.length == 2)

        val iHead = inits.find(_.value.op == AggElementsLengthCheck()).get
        val iNested = iHead.nested.get
        val iHeadArgTypes = iHead.value.args.map(_.pType)

        val sLCHead = seqs.find(_.value.op == AggElementsLengthCheck()).get
        val sLCArgTypes = sLCHead.value.args.map(_.pType)
        val sAEHead = seqs.find(_.value.op == AggElements()).get
        val sNested = sAEHead.nested.get
        val sHeadArgTypes = sAEHead.value.args.map(_.pType)

        val vNested = virt.nested.get.toArray

        val nested = vNested.indices.map { i => computePhysicalAgg(vNested(i), iNested(i), sNested(i)) }
        AggStatePhysicalSignature(Map(
          AggElementsLengthCheck() -> PhysicalAggSignature(AggElementsLengthCheck(), iHeadArgTypes, sLCArgTypes),
          AggElements() -> PhysicalAggSignature(AggElements(), FastIndexedSeq(), sHeadArgTypes)
        ), AggElementsLengthCheck(), Some(nested))

      case Group() =>
        assert(inits.length == 1)
        assert(seqs.length == 1)
        val iHead = inits.head
        val iNested = iHead.nested.get
        val iHeadArgTypes = iHead.value.args.map(_.pType)

        val sHead = seqs.head
        val sNested = sHead.nested.get
        val sHeadArgTypes = sHead.value.args.map(_.pType)

        val vNested = virt.nested.get.toArray

        val nested = vNested.indices.map { i => computePhysicalAgg(vNested(i), iNested(i), sNested(i)) }
        val psig = PhysicalAggSignature(Group(), iHeadArgTypes, sHeadArgTypes)
        AggStatePhysicalSignature(Map(Group() -> psig), Group(), Some(nested))

      case _ =>
        assert(inits.forall(_.nested.isEmpty))
        assert(seqs.forall(_.nested.isEmpty))
        val initArgTypes = inits.map(i => i.value.args.map(_.pType).toArray).transpose
          .map(ts => unifyPTypes(ts))
        val seqArgTypes = seqs.map(i => i.value.args.map(_.pType).toArray).transpose
          .map(ts => unifyPTypes(ts))
        virt.defaultSignature.toPhysical(initArgTypes, seqArgTypes).singletonContainer
    }
  }

  def unifyPTypes(pTypes: Seq[PType]): PType = {
    val r = TypeWithRequiredness.apply(pTypes.head.virtualType)
    pTypes.foreach(r.fromPType)
    unifyPTypes(pTypes, r)
  }

  def unifyPTypes(pTypes: Seq[PType], result: TypeWithRequiredness): PType = {
    assert(pTypes.tail.forall(pt => pt.virtualType == pTypes.head.virtualType))
    if (pTypes.tail.forall(pt => pt == pTypes.head))
      pTypes.head
    else result.canonicalPType(pTypes.head.virtualType)
  }

  def apply(ir: IR): Unit = apply(ir, Env.empty, null, null, null)

  private type AAB[T] = Array[ArrayBuilder[RecursiveArrayBuilderElement[T]]]

  case class RecursiveArrayBuilderElement[T](value: T, nested: Option[AAB[T]])

  def newBuilder[T](n: Int): AAB[T] = Array.fill(n)(new ArrayBuilder[RecursiveArrayBuilderElement[T]])

  def apply(ir: IR, env: Env[PType], aggs: Array[AggStatePhysicalSignature], inits: AAB[InitOp], seqs: AAB[SeqOp]): Unit = {
    try {
      val usesAndDefs = ComputeUsesAndDefs(ir, errorIfFreeVariables = false)
      val requiredness = Requiredness.apply(ir, usesAndDefs, null, env) // Value IR inference doesn't need context
      requiredness.states.m.foreach { case (ir, types) =>
        ir.t match {
          case x: StreamFold => x.accPTypes = types.map(r => r.canonicalPType(x.zero.typ))
          case x: StreamScan => x.accPTypes = types.map(r => r.canonicalPType(x.zero.typ))
          case x: StreamFold2 =>
            x.accPTypes = x.accum.zip(types).map { case ((_, arg), r) => r.canonicalPType(arg.typ) }.toArray
          case x: TailLoop =>
            x.accPTypes = x.params.zip(types).map { case ((_, arg), r) => r.canonicalPType(arg.typ) }.toArray
        }
      }
      _inferWithRequiredness(ir, env, requiredness, usesAndDefs, aggs)
      if (inits != null || seqs != null)
        _inferAggs(ir, inits, seqs)
    } catch {
      case e: Exception =>
        throw new RuntimeException(s"error while inferring IR:\n${Pretty(ir)}", e)
    }
    VisitIR(ir) { case (node: IR) =>
      if (node._pType == null)
        throw new RuntimeException(s"ptype inference failure: node not inferred:\n${Pretty(node)}\n ** Full IR: **\n${Pretty(ir)}")
    }
  }

  private def lookup(name: String, r: TypeWithRequiredness, defNode: IR): PType = defNode match {
    case Let(`name`, value, _) => value.pType
    case TailLoop(`name`, _, body) => r.canonicalPType(body.typ)
    case x: TailLoop => x.accPTypes(x.paramIdx(name))
    case ArraySort(a, l, r, c) => coerce[PStream](a.pType).elementType
    case StreamMap(a, `name`, _) => coerce[PStream](a.pType).elementType
    case x@StreamZip(as, _, _, _) =>
      coerce[PStream](as(x.nameIdx(name)).pType).elementType.setRequired(r.required)
    case StreamFilter(a, `name`, _) => coerce[PStream](a.pType).elementType
    case StreamFlatMap(a, `name`, _) => coerce[PStream](a.pType).elementType
    case StreamFor(a, `name`, _) => coerce[PStream](a.pType).elementType
    case StreamFold(a, _, _, `name`, _) => coerce[PStream](a.pType).elementType
    case x: StreamFold => x.accPType
    case StreamScan(a, _, _, `name`, _) => coerce[PStream](a.pType).elementType
    case x: StreamScan => x.accPType
    case StreamFold2(a, _, `name`, _, _) => coerce[PStream](a.pType).elementType
    case x: StreamFold2 => x.accPTypes(x.nameIdx(name))
    case StreamLeftJoinDistinct(left, _, `name`, _, _, _) => coerce[PStream](left.pType).elementType
    case StreamLeftJoinDistinct(_, right, _, `name`, _, _) => coerce[PStream](right.pType).elementType.setRequired(false)
    case RunAggScan(a, `name`, _, _, _, _) => coerce[PStream](a.pType).elementType
    case NDArrayMap(nd, `name`, _) => coerce[PNDArray](nd.pType).elementType
    case NDArrayMap2(left, _, `name`, _, _) => coerce[PNDArray](left.pType  ).elementType
    case NDArrayMap2(_, right, _, `name`, _) => coerce[PNDArray](right.pType).elementType
    case x@CollectDistributedArray(_, _, `name`, _, _) => x.decodedContextPType
    case x@CollectDistributedArray(_, _, _, `name`, _) => x.decodedGlobalPType
    case _ => throw new RuntimeException(s"$name not found in definition \n${ Pretty(defNode) }")
  }

  private def _inferAggs(node: IR, inits: AAB[InitOp] = null, seqs: AAB[SeqOp] = null): Unit =
    node match {
      case x@InitOp(i, args, sig, op) =>
        op match {
          case Group() =>
            val newInits = newBuilder[InitOp](sig.nested.get.length)
            val IndexedSeq(initArg) = args
            _inferAggs(initArg, inits = newInits, seqs = null)
            if (inits != null)
              inits(i) += RecursiveArrayBuilderElement(x, Some(newInits))
          case AggElementsLengthCheck() =>
            val newInits = newBuilder[InitOp](sig.nested.get.length)
            val initArg = args.last
            _inferAggs(initArg, inits = newInits, seqs = null)
            if (inits != null)
              inits(i) += RecursiveArrayBuilderElement(x, Some(newInits))
          case _ =>
            assert(sig.nested.isEmpty)
            if (inits != null)
              inits(i) += RecursiveArrayBuilderElement(x, None)
        }
      case x@SeqOp(i, args, sig, op) =>
        op match {
          case Group() =>
            val newSeqs = newBuilder[SeqOp](sig.nested.get.length)
            val IndexedSeq(_, seqArg) = args
            _inferAggs(seqArg, inits = null, seqs = newSeqs)
            if (seqs != null)
              seqs(i) += RecursiveArrayBuilderElement(x, Some(newSeqs))
          case AggElements() =>
            val newSeqs = newBuilder[SeqOp](sig.nested.get.length)
            val IndexedSeq(_, seqArg) = args
            _inferAggs(seqArg, inits = null, seqs = newSeqs)
            if (seqs != null)
              seqs(i) += RecursiveArrayBuilderElement(x, Some(newSeqs))
          case AggElementsLengthCheck() =>
            if (seqs != null)
              seqs(i) += RecursiveArrayBuilderElement(x, None)
          case _ =>
            assert(sig.nested.isEmpty)
            if (seqs != null)
              seqs(i) += RecursiveArrayBuilderElement(x, None)
        }
      case _ => _inferAggs(node, inits, seqs)
    }

  private def _inferWithRequiredness(node: IR, env: Env[PType], requiredness: RequirednessAnalysis, usesAndDefs: UsesAndDefs, aggs: Array[AggStatePhysicalSignature] = null): Unit = {
    if (node._pType != null)
      throw new RuntimeException(node.toString)
    node match {
      case x@RunAgg(body, result, signature) =>
        _inferWithRequiredness(body, env, requiredness, usesAndDefs)
        val inits = newBuilder[InitOp](signature.length)
        val seqs = newBuilder[SeqOp](signature.length)
        _inferAggs(body, inits = inits, seqs = seqs)
        val sigs = signature.indices.map { i => computePhysicalAgg(signature(i), inits(i), seqs(i)) }.toArray
        x.physicalSignatures = sigs
        _inferWithRequiredness(result, env, requiredness, usesAndDefs, x.physicalSignatures)
      case x@RunAggScan(array, name, init, seq, result, signature) =>
        _inferWithRequiredness(array, env, requiredness, usesAndDefs)
        _inferWithRequiredness(init, env, requiredness, usesAndDefs)
        _inferWithRequiredness(seq, env, requiredness, usesAndDefs)
        val inits = newBuilder[InitOp](signature.length)
        val seqs = newBuilder[SeqOp](signature.length)
        _inferAggs(init, inits = inits, seqs = null)
        _inferAggs(seq, inits = null, seqs = seqs)
        val sigs = signature.indices.map { i => computePhysicalAgg(signature(i), inits(i), seqs(i)) }.toArray
        x.physicalSignatures = sigs
        _inferWithRequiredness(result, env, requiredness, usesAndDefs, x.physicalSignatures)
      case _ =>
        node.children.foreach {
          case x: IR => _inferWithRequiredness(x, env, requiredness, usesAndDefs, aggs)
          case c => throw new RuntimeException(s"unsupported node:\n${Pretty(c)}")
        }
    }
    node._pType = node match {
      case _: I32 | _: I64 | _: F32 | _: F64 | _: Str | _: Literal | _: True | _: False
           | _: Cast | _: NA | _: Die | _: IsNA | _: ArrayZeros | _: ArrayLen | _: StreamLen
           | _: LowerBoundOnOrderedCollection | _: ApplyBinaryPrimOp
           | _: ApplyUnaryPrimOp | _: ApplyComparisonOp | _: WriteValue
           | _: NDArrayAgg | _: ShuffleWrite | _: ShuffleStart | _: AggStateValue =>
        requiredness(node).canonicalPType(node.typ)
      case x: BaseRef if usesAndDefs.free.contains(RefEquality(x)) =>
        env.lookup(x.name)
      case x: BaseRef =>
        lookup(x.name, requiredness(node), usesAndDefs.defs.lookup(node).asInstanceOf[IR])
      case MakeNDArray(data, shape, rowMajor) =>
        val nElem = shape.pType.asInstanceOf[PTuple].size
        PCanonicalNDArray(coerce[PArray](data.pType).elementType.setRequired(true), nElem, data.pType.required && shape.pType.required)
      case StreamRange(start: IR, stop: IR, step: IR) =>
        assert(start.pType isOfType stop.pType)
        assert(start.pType isOfType step.pType)
        PCanonicalStream(start.pType.setRequired(true), requiredness(node).required)
      case Let(_, _, body) => body.pType
      case TailLoop(_, _, body) => body.pType
      case a: AbstractApplyNode[_] => a.implementation.returnPType(a.returnType, a.args.map(_.pType))
      case ArrayRef(a, i, s) =>
        assert(i.pType isOfType PInt32())
        coerce[PArray](a.pType).elementType.setRequired(requiredness(node).required)
      case ArraySort(a, leftName, rightName, lessThan) =>
        assert(lessThan.pType.isOfType(PBoolean()))
        PCanonicalArray(coerce[PIterable](a.pType).elementType, requiredness(node).required)
      case ToSet(a) =>
        PCanonicalSet(coerce[PIterable](a.pType).elementType, requiredness(node).required)
      case ToDict(a) =>
        val elt = coerce[PBaseStruct](coerce[PIterable](a.pType).elementType)
        PCanonicalDict(elt.types(0), elt.types(1), requiredness(node).required)
      case ToArray(a) =>
        val elt = coerce[PIterable](a.pType).elementType
        PCanonicalArray(elt, requiredness(node).required)
      case CastToArray(a) =>
        val elt = coerce[PIterable](a.pType).elementType
        PCanonicalArray(elt, requiredness(node).required)
      case ToStream(a) =>
        val elt = coerce[PIterable](a.pType).elementType
        PCanonicalStream(elt, requiredness(node).required)
      case GroupByKey(collection) =>
        val r = coerce[RDict](requiredness(node))
        val elt = coerce[PBaseStruct](coerce[PStream](collection.pType).elementType)
        PCanonicalDict(elt.types(0), PCanonicalArray(elt.types(1), r.valueType.required), r.required)
      case StreamTake(a, len) =>
        a.pType.setRequired(requiredness(node).required)
      case StreamDrop(a, len) =>
        a.pType.setRequired(requiredness(node).required)
      case StreamGrouped(a, size) =>
        val r = coerce[RIterable](requiredness(node))
        assert(size.pType isOfType PInt32())
        assert(a.pType.isInstanceOf[PStream])
        PCanonicalStream(a.pType.setRequired(r.elementType.required), r.required)
      case StreamGroupByKey(a, key) =>
        val r = coerce[RIterable](requiredness(node))
        val structType = a.pType.asInstanceOf[PStream].elementType.asInstanceOf[PStruct]
        assert(structType.required)
        assert(key.forall { k => structType.fieldType(k).required })
        PCanonicalStream(a.pType.setRequired(r.elementType.required), r.required)
      case StreamMap(a, name, body) =>
        PCanonicalStream(body.pType, requiredness(node).required)
      case StreamZip(as, names, body, behavior) =>
        PCanonicalStream(body.pType, requiredness(node).required)
      case StreamFilter(a, name, cond) => a.pType
      case StreamFlatMap(a, name, body) =>
        PCanonicalStream(coerce[PIterable](body.pType).elementType, requiredness(node).required)
      case x: StreamFold =>
        x.accPType.setRequired(requiredness(node).required)
      case x: StreamFold2 =>
        x.result.pType.setRequired(requiredness(node).required)
      case x: StreamScan =>
        val r = coerce[RIterable](requiredness(node))
        PCanonicalStream(x.accPType.setRequired(r.elementType.required), r.required)
      case StreamLeftJoinDistinct(_, _, _, _, _, join) =>
        PCanonicalStream(join.pType, requiredness(node).required)
      case NDArrayShape(nd) =>
        val r = nd.pType.asInstanceOf[PNDArray].shape.pType
        r.setRequired(requiredness(node).required)
      case NDArrayReshape(nd, shape) =>
        val shapeT = shape.pType.asInstanceOf[PTuple]
        PCanonicalNDArray(coerce[PNDArray](nd.pType).elementType, shapeT.size,
          requiredness(node).required)
      case NDArrayConcat(nds, _) =>
        val ndtyp = coerce[PNDArray](coerce[PArray](nds.pType).elementType)
        ndtyp.setRequired(requiredness(node).required)
      case NDArrayMap(nd, name, body) =>
        val ndPType = nd.pType.asInstanceOf[PNDArray]
        PCanonicalNDArray(body.pType.setRequired(true), ndPType.nDims, requiredness(node).required)
      case NDArrayMap2(l, r, lName, rName, body) =>
        val lPType = l.pType.asInstanceOf[PNDArray]
        PCanonicalNDArray(body.pType.setRequired(true), lPType.nDims, requiredness(node).required)
      case NDArrayReindex(nd, indexExpr) =>
        PCanonicalNDArray(coerce[PNDArray](nd.pType).elementType, indexExpr.length, requiredness(node).required)
      case NDArrayRef(nd, idxs) =>
        coerce[PNDArray](nd.pType).elementType.setRequired(requiredness(node).required)
      case NDArraySlice(nd, slices) =>
        val remainingDims = coerce[PTuple](slices.pType).types.filter(_.isInstanceOf[PTuple])
        PCanonicalNDArray(coerce[PNDArray](nd.pType).elementType, remainingDims.length, requiredness(node).required)
      case NDArrayFilter(nd, filters) => coerce[PNDArray](nd.pType)
      case NDArrayMatMul(l, r) =>
        val lTyp = coerce[PNDArray](l.pType)
        val rTyp = coerce[PNDArray](r.pType)
        PCanonicalNDArray(lTyp.elementType, TNDArray.matMulNDims(lTyp.nDims, rTyp.nDims), requiredness(node).required)
      case NDArrayQR(nd, mode) => NDArrayQR.pTypes(mode)
      case MakeStruct(fields) =>
        PCanonicalStruct(requiredness(node).required,
          fields.map { case (name, a) => (name, a.pType) }: _ *)
      case SelectFields(old, fields) =>
        if(HailContext.getFlag("use_spicy_ptypes") != null) {
          PSubsetStruct(coerce[PStruct](old.pType), fields:_*)
        } else {
          val tbs = coerce[PStruct](old.pType)
          tbs.selectFields(fields.toFastIndexedSeq)
        }
      case InsertFields(old, fields, fieldOrder) =>
        val tbs = coerce[PStruct](old.pType)
        val s = tbs.insertFields(fields.map(f => { (f._1, f._2.pType) }))
        fieldOrder.map { fds =>
          assert(fds.length == s.size)
          PCanonicalStruct(tbs.required, fds.map(f => f -> s.fieldType(f)): _*)
        }.getOrElse(s)
      case GetField(o, name) =>
        val t = coerce[PStruct](o.pType)
        if (t.index(name).isEmpty)
          throw new RuntimeException(s"$name not in $t")
        t.field(name).typ.setRequired(requiredness(node).required)
      case MakeTuple(values) =>
        PCanonicalTuple(values.map { case (idx, v) =>
          PTupleField(idx, v.pType)
        }.toFastIndexedSeq, requiredness(node).required)
      case MakeArray(irs, t) =>
        val r = coerce[RIterable](requiredness(node))
        if (irs.isEmpty) r.canonicalPType(t) else
        PCanonicalArray(unifyPTypes(irs.map(_.pType), r.elementType), r.required)
      case GetTupleElement(o, idx) =>
        val t = coerce[PTuple](o.pType)
        t.fields(t.fieldIndex(idx)).typ.setRequired(requiredness(node).required)
      case If(cond, cnsq, altr) =>
        assert(cond.pType isOfType PBoolean())
        val r = requiredness(node)
        unifyPTypes(FastIndexedSeq(cnsq.pType, altr.pType), r).setRequired(r.required)
      case Coalesce(values) =>
        val r = requiredness(node)
        unifyPTypes(values.map(_.pType), r).setRequired(r.required)
      case In(_, pType: PType) => pType
      case x: CollectDistributedArray =>
        PCanonicalArray(x.decodedBodyPType, requiredness(node).required)
      case ReadPartition(context, rowType, reader) =>
        val child = reader.rowPType(rowType)
        PCanonicalStream(child, requiredness(node).required)
      case ReadValue(path, spec, requestedType) =>
        spec.decodedPType(requestedType).setRequired(requiredness(node).required)
      case MakeStream(irs, t) =>
        val r = coerce[RIterable](requiredness(node))
        if (irs.isEmpty) r.canonicalPType(t) else
          PCanonicalStream(unifyPTypes(irs.map(_.pType), r.elementType), r.required)
      case x@ResultOp(resultIdx, sigs) =>
        PCanonicalTuple(true, (resultIdx until resultIdx + sigs.length).map(i => aggs(i).resultType): _*)
      case x@RunAgg(body, result, signature) => result.pType
      case x@RunAggScan(array, name, init, seq, result, signature) =>
        PCanonicalStream(result.pType, array.pType.required)
      case ShuffleGetPartitionBounds(_, _, keyFields, rowType, keyEType) =>
        val keyPType = keyEType.decodedPType(rowType.typeAfterSelectNames(keyFields.map(_.field)))
        PCanonicalArray(keyPType, requiredness(node).required)
      case ShuffleRead(_, _, rowType, rowEType) =>
        val rowPType = rowEType.decodedPType(rowType)
        PCanonicalStream(rowPType, requiredness(node).required)
      case x if x.typ == TVoid => PVoid
      case _ => throw new RuntimeException(s"unsupported node:\n${Pretty(node)}")
    }
    if (node.pType.virtualType != node.typ)
      throw new RuntimeException(s"pType.virtualType: ${node.pType.virtualType}, vType = ${node.typ}\n  ir=$node")
  }
}
