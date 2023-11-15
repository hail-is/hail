package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.Pretty.prettyBooleanLiteral
import is.hail.expr.ir.agg._
import is.hail.expr.ir.functions.RelationalFunctions
import is.hail.types.TableType
import is.hail.types.virtual.{TArray, TInterval, TStream, Type}
import is.hail.utils.prettyPrint._
import is.hail.utils.richUtils.RichIterable
import is.hail.utils.{space => _, _}
import org.json4s.DefaultFormats
import org.json4s.jackson.{JsonMethods, Serialization}

import scala.collection.mutable

object Pretty {
  def apply(ctx: ExecuteContext, ir: BaseIR, width: Int = 100, ribbonWidth: Int = 50, elideLiterals: Boolean = true, maxLen: Int = -1, allowUnboundRefs: Boolean = false): String = {
    val useSSA = ctx != null && ctx.getFlag("use_ssa_logs") != null
    val pretty = new Pretty(width, ribbonWidth, elideLiterals, maxLen, allowUnboundRefs, useSSA)
    pretty(ir)
  }

  def sexprStyle(ir: BaseIR, width: Int = 100, ribbonWidth: Int = 50, elideLiterals: Boolean = true, maxLen: Int = -1, allowUnboundRefs: Boolean = false): String = {
    val pretty = new Pretty(width, ribbonWidth, elideLiterals, maxLen, allowUnboundRefs, useSSA = false)
    pretty(ir)
  }

  def ssaStyle(ir: BaseIR, width: Int = 100, ribbonWidth: Int = 50, elideLiterals: Boolean = true, maxLen: Int = -1, allowUnboundRefs: Boolean = false): String = {
    val pretty = new Pretty(width, ribbonWidth, elideLiterals, maxLen, allowUnboundRefs, useSSA = true)
    pretty(ir)
  }

  def prettyBooleanLiteral(b: Boolean): String =
    if (b) "True" else "False"

  def prettyClass(x: AnyRef): String = {
    val name = x.getClass.getName.split("\\.").last
    if (name.endsWith("$")) name.substring(0, name.length - 1)
    else name
  }
}

class Pretty(width: Int, ribbonWidth: Int, elideLiterals: Boolean, maxLen: Int, allowUnboundRefs: Boolean, useSSA: Boolean) {
  def short(ir: BaseIR): String = {
    val s = apply(ir)
    if (s.length < maxLen) s else s.substring(0, maxLen) + "..."
  }

  def prettyStringLiteral(s: String, elide: Boolean = false, maxLen: Int = 1000): String = {
    val esc = StringEscapeUtils.escapeString(s)
    if (elide && esc.length > maxLen) {
      s""""${esc.substring(0, maxLen)}...""""
    } else {
      s""""$esc""""
    }
  }

  def fillList(docs: Iterable[Doc], indent: Int = 2): Doc =
    group(lineAlt, nest(indent, prettyPrint.fillList(docs)))

  def fillList(docs: Doc*): Doc = fillList(docs)

  def list(docs: Iterable[Doc], indent: Int = 2): Doc =
    nest(indent, prettyPrint.list(docs))

  def list(docs: Doc*): Doc = list(docs)

  def prettyStrings(xs: IndexedSeq[String]): Doc =
    fillList(xs.view.map(x => text(prettyStringLiteral(x))))

  def prettyStringsOpt(x: Option[IndexedSeq[String]]): Doc =
    x.map(prettyStrings).getOrElse("None")

  def prettyTypes(x: Seq[Type]): Doc =
    fillList(x.view.map(typ => text(typ.parsableString())))

  def prettySortFields(x: Seq[SortField]): Doc =
    fillList(x.view.map(typ => text(typ.parsableString())))

  def prettySortFieldsString(x: Seq[SortField]): String =
    x.view.map(_.parsableString()).mkString("(", " ", ")")

  val MAX_VALUES_TO_LOG: Int = 25

  def prettyIntOpt(x: Option[Int]): String =
    x.map(_.toString).getOrElse("None")

  def prettyLongs(x: IndexedSeq[Long], elideLiterals: Boolean): Doc = {
    val truncate = elideLiterals && x.length > MAX_VALUES_TO_LOG
    val view = if (truncate) x.view else x.view(0, MAX_VALUES_TO_LOG)
    val docs = view.map(i => text(i.toString))
    concat(docs.intersperse[Doc](
      "(",
      softline,
      if (truncate) s"... ${ x.length - MAX_VALUES_TO_LOG } more values... )" else ")"))
  }

  def prettyInts(x: IndexedSeq[Int], elideLiterals: Boolean): Doc = {
    val truncate = elideLiterals && x.length > MAX_VALUES_TO_LOG
    val view = if (truncate) x.view else x.view(0, MAX_VALUES_TO_LOG)
    val docs = view.map(i => text(i.toString))
    concat(docs.intersperse[Doc](
      "(",
      softline,
      if (truncate) s"... ${ x.length - MAX_VALUES_TO_LOG } more values... )" else ")"))
  }

  def prettyIdentifiers(x: IndexedSeq[String]): Doc =
    fillList(x.view.map(text))

  def prettyAggStateSignatures(states: Seq[AggStateSig]): Doc =
    list(states.view.map(prettyAggStateSignature))

  def prettyAggStateSignature(state: AggStateSig): Doc = {
    state match {
      case FoldStateSig(resultEmitType, accumName, otherAccumName, combOpIR) =>
        fillList(IndexedSeq(text(Pretty.prettyClass(state)), text(resultEmitType.typeWithRequiredness.canonicalPType.toString),
          text(accumName), text(otherAccumName), text(apply(combOpIR))))
      case _ =>
        fillList(state.n match {
          case None => text(Pretty.prettyClass(state)) +: state.t.view.map(typ => text(typ.canonicalPType.toString))
          case Some(nested) => text(Pretty.prettyClass(state)) +: state.t.view.map(typ => text(typ.canonicalPType.toString)) :+ prettyAggStateSignatures(nested)
        })
    }
  }

  def prettyPhysicalAggSigs(aggSigs: Seq[PhysicalAggSig]): Doc =
    list(aggSigs.view.map(prettyPhysicalAggSig))

  def prettyPhysicalAggSig(aggSig: PhysicalAggSig): Doc = {
    aggSig match {
      case GroupedAggSig(t, nested) =>
        fillList("Grouped", t.canonicalPType.toString, prettyPhysicalAggSigs(nested))
      case ArrayLenAggSig(kl, nested) =>
        fillList("ArrayLen", Pretty.prettyBooleanLiteral(kl), prettyPhysicalAggSigs(nested))
      case AggElementsAggSig(nested) =>
        fillList("AggElements", prettyPhysicalAggSigs(nested))
      case PhysicalAggSig(op, state) =>
        fillList(Pretty.prettyClass(op), prettyAggStateSignature(state))
    }
  }

  def single(d: Doc): Iterable[Doc] = RichIterable.single(d)

  def header(ir: BaseIR, elideBindings: Boolean = false): Iterable[Doc] = ir match {
    case ApplyAggOp(initOpArgs, seqOpArgs, aggSig) => single(Pretty.prettyClass(aggSig.op))
    case ApplyScanOp(initOpArgs, seqOpArgs, aggSig) => single(Pretty.prettyClass(aggSig.op))
    case InitOp(i, args, aggSig) => FastSeq(i.toString, prettyPhysicalAggSig(aggSig))
    case SeqOp(i, args, aggSig) => FastSeq(i.toString, prettyPhysicalAggSig(aggSig))
    case CombOp(i1, i2, aggSig) => FastSeq(i1.toString, i2.toString, prettyPhysicalAggSig(aggSig))
    case ResultOp(i, aggSig) => FastSeq(i.toString, prettyPhysicalAggSig(aggSig))
    case AggStateValue(i, sig) => FastSeq(i.toString, prettyAggStateSignature(sig))
    case InitFromSerializedValue(i, value, aggSig) =>
      FastSeq(i.toString, prettyAggStateSignature(aggSig))
    case CombOpValue(i, value, sig) => FastSeq(i.toString, prettyPhysicalAggSig(sig))
    case SerializeAggs(i, i2, spec, aggSigs) =>
      FastSeq(i.toString, i2.toString, prettyStringLiteral(spec.toString), prettyAggStateSignatures(aggSigs))
    case DeserializeAggs(i, i2, spec, aggSigs) =>
      FastSeq(i.toString, i2.toString, prettyStringLiteral(spec.toString), prettyAggStateSignatures(aggSigs))
    case RunAgg(body, result, signature) => single(prettyAggStateSignatures(signature))
    case RunAggScan(a, name, init, seq, res, signature) =>
      FastSeq(prettyIdentifier(name), prettyAggStateSignatures(signature))
    case I32(x) => single(x.toString)
    case I64(x) => single(x.toString)
    case F32(x) => single(x.toString)
    case F64(x) => single(x.toString)
    case Str(x) => single(prettyStringLiteral(if (elideLiterals && x.length > 13) x.take(10) + "..." else x))
    case UUID4(id) => single(prettyIdentifier(id))
    case Cast(_, typ) => single(typ.parsableString())
    case CastRename(_, typ) => single(typ.parsableString())
    case NA(typ) => single(typ.parsableString())
    case Literal(typ, value) =>
      FastSeq(typ.parsableString(),
        if (!elideLiterals)
          prettyStringLiteral(JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(value, typ)))
        else
          "<literal value>")
    case EncodedLiteral(codec, _) => single(codec.encodedVirtualType.parsableString())
    case Let(name, _, _) if !elideBindings => single(prettyIdentifier(name))
    case AggLet(name, _, _, isScan) => if (elideBindings)
      single(Pretty.prettyBooleanLiteral(isScan))
    else
      FastSeq(prettyIdentifier(name), Pretty.prettyBooleanLiteral(isScan))
    case TailLoop(name, args, returnType, _) if !elideBindings =>
      FastSeq(prettyIdentifier(name), prettyIdentifiers(args.map(_._1).toFastSeq), returnType.parsableString())
    case Recur(name, _, t) if !elideBindings =>
      FastSeq(prettyIdentifier(name))
//    case Ref(name, t) if t != null => FastSeq(prettyIdentifier(name), t.parsableString())  // For debug purposes
    case Ref(name, _) => single(prettyIdentifier(name))
    case RelationalRef(name, t) => if (elideBindings)
      single(t.parsableString())
    else
      FastSeq(prettyIdentifier(name), t.parsableString())
    case RelationalLet(name, _, _) if !elideBindings => single(prettyIdentifier(name))
    case ApplyBinaryPrimOp(op, _, _) => single(Pretty.prettyClass(op))
    case ApplyUnaryPrimOp(op, _) => single(Pretty.prettyClass(op))
    case ApplyComparisonOp(op, _, _) => single(op.render())
    case GetField(_, name) => single(prettyIdentifier(name))
    case GetTupleElement(_, idx) => single(idx.toString)
    case MakeTuple(fields) => FastSeq(prettyInts(fields.map(_._1).toFastSeq, elideLiterals))
    case MakeArray(_, typ) => single(typ.parsableString())
    case MakeStream(_, typ, requiresMemoryManagementPerElement) =>
      FastSeq(typ.parsableString(), Pretty.prettyBooleanLiteral(requiresMemoryManagementPerElement))
    case StreamIota(_, _, requiresMemoryManagementPerElement) => FastSeq(Pretty.prettyBooleanLiteral(requiresMemoryManagementPerElement))
    case StreamRange(_, _, _, requiresMemoryManagementPerElement, errorID) => FastSeq(errorID.toString, Pretty.prettyBooleanLiteral(requiresMemoryManagementPerElement))
    case ToStream(_, requiresMemoryManagementPerElement) => single(Pretty.prettyBooleanLiteral(requiresMemoryManagementPerElement))
    case StreamMap(_, name, _) if !elideBindings => single(prettyIdentifier(name))
    case StreamZip(_, names, _, behavior, errorID) => if (elideBindings)
      FastSeq(errorID.toString, behavior match {
        case ArrayZipBehavior.AssertSameLength => "AssertSameLength"
        case ArrayZipBehavior.TakeMinLength => "TakeMinLength"
        case ArrayZipBehavior.ExtendNA => "ExtendNA"
        case ArrayZipBehavior.AssumeSameLength => "AssumeSameLength"
      })
    else
      FastSeq(errorID.toString, behavior match {
      case ArrayZipBehavior.AssertSameLength => "AssertSameLength"
      case ArrayZipBehavior.TakeMinLength => "TakeMinLength"
      case ArrayZipBehavior.ExtendNA => "ExtendNA"
      case ArrayZipBehavior.AssumeSameLength => "AssumeSameLength"
    }, prettyIdentifiers(names))
    case StreamZipJoin(streams, key, curKey, curVals, _) if !elideBindings =>
      FastSeq(streams.length.toString, prettyIdentifiers(key), prettyIdentifier(curKey), prettyIdentifier(curVals))
    case StreamZipJoinProducers(_, ctxName, _, key, curKey, curVals, _) if !elideBindings =>
      FastSeq(prettyIdentifiers(key), prettyIdentifier(ctxName), prettyIdentifier(curKey), prettyIdentifier(curVals))
    case StreamMultiMerge(_, key) => single(prettyIdentifiers(key))
    case StreamFilter(_, name, _) if !elideBindings => single(prettyIdentifier(name))
    case StreamTakeWhile(_, name, _) if !elideBindings => single(prettyIdentifier(name))
    case StreamDropWhile(_, name, _) if !elideBindings => single(prettyIdentifier(name))
    case StreamFlatMap(_, name, _) if !elideBindings => single(prettyIdentifier(name))
    case StreamFold(_, _, accumName, valueName, _) if !elideBindings => FastSeq(prettyIdentifier(accumName), prettyIdentifier(valueName))
    case StreamFold2(_, acc, valueName, _, _) if !elideBindings => FastSeq(prettyIdentifiers(acc.map(_._1)), prettyIdentifier(valueName))
    case StreamScan(_, _, accumName, valueName, _) if !elideBindings => FastSeq(prettyIdentifier(accumName), prettyIdentifier(valueName))
    case StreamWhiten(_, newChunk, prevWindow, vecSize, windowSize, chunkSize, blockSize, normalizeAfterWhiten) =>
      FastSeq(prettyIdentifier(newChunk), prettyIdentifier(prevWindow), vecSize.toString, windowSize.toString, chunkSize.toString, blockSize.toString, Pretty.prettyBooleanLiteral(normalizeAfterWhiten))
    case StreamJoinRightDistinct(_, _, lKey, rKey, l, r, _, joinType) => if (elideBindings)
      FastSeq(prettyIdentifiers(lKey), prettyIdentifiers(rKey), joinType)
    else
      FastSeq(prettyIdentifiers(lKey), prettyIdentifiers(rKey), prettyIdentifier(l), prettyIdentifier(r), joinType)
    case StreamFor(_, valueName, _) if !elideBindings => single(prettyIdentifier(valueName))
    case StreamAgg(a, name, query) if !elideBindings => single(prettyIdentifier(name))
    case StreamAggScan(a, name, query) if !elideBindings => single(prettyIdentifier(name))
    case StreamGroupByKey(a, key, missingEqual) => FastSeq(prettyIdentifiers(key), prettyBooleanLiteral(missingEqual))
    case AggFold(_, _, _, accumName, otherAccumName, isScan) => if (elideBindings)
      single(Pretty.prettyBooleanLiteral(isScan))
    else
      FastSeq(prettyIdentifier(accumName), prettyIdentifier(otherAccumName), Pretty.prettyBooleanLiteral(isScan))
    case AggExplode(_, name, _, isScan) => if (elideBindings)
      single(Pretty.prettyBooleanLiteral(isScan))
    else
      FastSeq(prettyIdentifier(name), Pretty.prettyBooleanLiteral(isScan))
    case AggFilter(_, _, isScan) => single(Pretty.prettyBooleanLiteral(isScan))
    case AggGroupBy(_, _, isScan) => single(Pretty.prettyBooleanLiteral(isScan))
    case AggArrayPerElement(_, elementName, indexName, _, knownLength, isScan) => if (elideBindings)
      FastSeq(Pretty.prettyBooleanLiteral(isScan), Pretty.prettyBooleanLiteral(knownLength.isDefined))
    else
      FastSeq(prettyIdentifier(elementName), prettyIdentifier(indexName), Pretty.prettyBooleanLiteral(isScan), Pretty.prettyBooleanLiteral(knownLength.isDefined))
    case NDArrayMap(_, name, _) if !elideBindings => single(prettyIdentifier(name))
    case NDArrayMap2(_, _, lName, rName, _, errorID) => if (elideBindings)
      single(s"$errorID")
    else
      FastSeq(s"$errorID", prettyIdentifier(lName), prettyIdentifier(rName))
    case NDArrayReindex(_, indexExpr) => single(prettyInts(indexExpr, elideLiterals))
    case NDArrayConcat(_, axis) => single(axis.toString)
    case NDArrayAgg(_, axes) => single(prettyInts(axes, elideLiterals))
    case NDArrayRef(_, _, errorID) => single(s"$errorID")
    case NDArrayReshape(_, _, errorID) => single(s"$errorID")
    case NDArrayMatMul(_, _, errorID) => single(s"$errorID")
    case NDArrayQR(_, mode, errorID) => FastSeq(errorID.toString, mode)
    case NDArraySVD(_, fullMatrices, computeUV, errorID) => FastSeq(errorID.toString, fullMatrices.toString, computeUV.toString)
    case NDArrayEigh(_, eigvalsOnly, errorID) => FastSeq(errorID.toString, eigvalsOnly.toString)
    case NDArrayInv(_, errorID) => single(s"$errorID")
    case ArraySort(_, l, r, _) if !elideBindings => FastSeq(prettyIdentifier(l), prettyIdentifier(r))
    case ArrayRef(_,_, errorID) => single(s"$errorID")
    case ApplyIR(function, typeArgs, _, _, errorID) => FastSeq(s"$errorID", prettyIdentifier(function), prettyTypes(typeArgs), ir.typ.parsableString())
    case Apply(function, typeArgs, _, t, errorID) => FastSeq(s"$errorID", prettyIdentifier(function), prettyTypes(typeArgs), t.parsableString())
    case ApplySeeded(function, _, rngState, staticUID, t) => FastSeq(prettyIdentifier(function), staticUID.toString, t.parsableString())
    case ApplySpecial(function, typeArgs, _, t, errorID) => FastSeq(s"$errorID", prettyIdentifier(function), prettyTypes(typeArgs), t.parsableString())
    case SelectFields(_, fields) => single(fillList(fields.view.map(f => text(prettyIdentifier(f)))))
    case LowerBoundOnOrderedCollection(_, _, onKey) => single(Pretty.prettyBooleanLiteral(onKey))
    case In(i, typ) => FastSeq(typ.toString, i.toString)
    case Die(message, typ, errorID) => FastSeq(typ.parsableString(), errorID.toString)
    case CollectDistributedArray(_, _, cname, gname, _, _, staticID, _) if !elideBindings =>
      FastSeq(staticID, prettyIdentifier(cname), prettyIdentifier(gname))
    case MatrixRead(typ, dropCols, dropRows, reader) =>
      FastSeq(if (typ == reader.fullMatrixType) "None" else typ.parsableString(),
        Pretty.prettyBooleanLiteral(dropCols),
        Pretty.prettyBooleanLiteral(dropRows),
        if (elideLiterals) reader.renderShort() else '"' + StringEscapeUtils.escapeString(JsonMethods.compact(reader.toJValue)) + '"')
    case MatrixWrite(_, writer) =>
      single('"' + StringEscapeUtils.escapeString(Serialization.write(writer)(MatrixWriter.formats)) + '"')
    case MatrixMultiWrite(_, writer) =>
      single('"' + StringEscapeUtils.escapeString(Serialization.write(writer)(MatrixNativeMultiWriter.formats)) + '"')
    case BlockMatrixRead(reader) =>
      single('"' + StringEscapeUtils.escapeString(JsonMethods.compact(reader.toJValue)) + '"')
    case BlockMatrixWrite(_, writer) =>
      single('"' + StringEscapeUtils.escapeString(Serialization.write(writer)(BlockMatrixWriter.formats)) + '"')
    case BlockMatrixMultiWrite(_, writer) =>
      single('"' + StringEscapeUtils.escapeString(Serialization.write(writer)(BlockMatrixWriter.formats)) + '"')
    case BlockMatrixBroadcast(_, inIndexExpr, shape, blockSize) =>
      FastSeq(prettyInts(inIndexExpr, elideLiterals),
        prettyLongs(shape, elideLiterals),
        blockSize.toString)
    case BlockMatrixAgg(_, outIndexExpr) => single(prettyInts(outIndexExpr, elideLiterals))
    case BlockMatrixSlice(_, slices) =>
      single(fillList(slices.view.map(slice => prettyLongs(slice, elideLiterals))))
    case ValueToBlockMatrix(_, shape, blockSize) =>
      FastSeq(prettyLongs(shape, elideLiterals), blockSize.toString)
    case BlockMatrixFilter(_, indicesToKeepPerDim) =>
      single(fillList(indicesToKeepPerDim.toSeq.view.map(indices => prettyLongs(indices, elideLiterals))))
    case BlockMatrixSparsify(_, sparsifier) =>
      single(sparsifier.pretty())
    case BlockMatrixRandom(staticUID, gaussian, shape, blockSize) =>
      FastSeq(
        staticUID.toString,
        Pretty.prettyBooleanLiteral(gaussian),
        prettyLongs(shape, elideLiterals),
        blockSize.toString)
    case BlockMatrixMap(_, name, _, needsDense) => if (elideBindings)
      single(Pretty.prettyBooleanLiteral(needsDense))
    else
      FastSeq(prettyIdentifier(name), Pretty.prettyBooleanLiteral(needsDense))
    case BlockMatrixMap2(_, _, lName, rName, _, sparsityStrategy) => if (elideBindings)
      single(Pretty.prettyClass(sparsityStrategy))
    else
      FastSeq(prettyIdentifier(lName), prettyIdentifier(rName), Pretty.prettyClass(sparsityStrategy))
    case MatrixRowsHead(_, n) => single(n.toString)
    case MatrixColsHead(_, n) => single(n.toString)
    case MatrixRowsTail(_, n) => single(n.toString)
    case MatrixColsTail(_, n) => single(n.toString)
    case MatrixAnnotateRowsTable(_, _, uid, product) =>
      FastSeq(prettyStringLiteral(uid), Pretty.prettyBooleanLiteral(product))
    case MatrixAnnotateColsTable(_, _, uid) => single(prettyStringLiteral(uid))
    case MatrixExplodeRows(_, path) => single(prettyIdentifiers(path))
    case MatrixExplodeCols(_, path) => single(prettyIdentifiers(path))
    case MatrixRepartition(_, n, strategy) => single(s"$n $strategy")
    case MatrixChooseCols(_, oldIndices) => single(prettyInts(oldIndices, elideLiterals))
    case MatrixMapCols(_, _, newKey) => single(prettyStringsOpt(newKey))
    case MatrixUnionCols(l, r, joinType) => single(joinType)
    case MatrixKeyRowsBy(_, keys, isSorted) =>
      FastSeq(prettyIdentifiers(keys), Pretty.prettyBooleanLiteral(isSorted))
    case TableRead(typ, dropRows, tr) =>
      FastSeq(if (typ == tr.fullType) "None" else typ.parsableString(),
        Pretty.prettyBooleanLiteral(dropRows),
        if (elideLiterals) tr.renderShort() else '"' + StringEscapeUtils.escapeString(JsonMethods.compact(tr.toJValue)) + '"')
    case TableWrite(_, writer) =>
      single('"' + StringEscapeUtils.escapeString(Serialization.write(writer)(TableWriter.formats)) + '"')
    case TableMultiWrite(_, writer) =>
      single('"' + StringEscapeUtils.escapeString(Serialization.write(writer)(WrappedMatrixNativeMultiWriter.formats)) + '"')
    case TableKeyBy(_, keys, isSorted) =>
      FastSeq(prettyIdentifiers(keys), Pretty.prettyBooleanLiteral(isSorted))
    case TableRange(n, nPartitions) => FastSeq(n.toString, nPartitions.toString)
    case TableRepartition(_, n, strategy) => FastSeq(n.toString, strategy.toString)
    case TableHead(_, n) => single(n.toString)
    case TableTail(_, n) => single(n.toString)
    case TableJoin(_, _, joinType, joinKey) => FastSeq(joinType, joinKey.toString)
    case TableLeftJoinRightDistinct(_, _, root) => single(prettyIdentifier(root))
    case TableIntervalJoin(_, _, root, product) =>
      FastSeq(prettyIdentifier(root), Pretty.prettyBooleanLiteral(product))
    case TableMultiWayZipJoin(_, dataName, globalName) =>
      FastSeq(prettyStringLiteral(dataName), prettyStringLiteral(globalName))
    case TableKeyByAndAggregate(_, _, _, nPartitions, bufferSize) =>
      FastSeq(prettyIntOpt(nPartitions), bufferSize.toString)
    case TableExplode(_, path) => single(prettyStrings(path))
    case TableMapPartitions(_, g, p, _, requestedKey, allowedOverlap) => FastSeq(prettyIdentifier(g), prettyIdentifier(p), requestedKey.toString, allowedOverlap.toString)
    case TableParallelize(_, nPartitions) => single(prettyIntOpt(nPartitions))
    case TableOrderBy(_, sortFields) => single(prettySortFields(sortFields))
    case CastMatrixToTable(_, entriesFieldName, colsFieldName) =>
      FastSeq(prettyStringLiteral(entriesFieldName), prettyStringLiteral(colsFieldName))
    case CastTableToMatrix(_, entriesFieldName, colsFieldName, colKey) =>
      FastSeq(prettyIdentifier(entriesFieldName), prettyIdentifier(colsFieldName), prettyIdentifiers(colKey))
    case MatrixToMatrixApply(_, function) =>
      single(prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats)))
    case MatrixToTableApply(_, function) =>
      single(prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats)))
    case TableToTableApply(_, function) =>
      single(prettyStringLiteral(JsonMethods.compact(function.toJValue)))
    case TableToValueApply(_, function) =>
      single(prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats)))
    case MatrixToValueApply(_, function) =>
      single(prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats)))
    case BlockMatrixToValueApply(_, function) =>
      single(prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats)))
    case BlockMatrixToTableApply(_, _, function) =>
      single(prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats)))
    case TableGen(_, _, cname, gname, _, partitioner, errorId) =>
      implicit val jsonFormats = DefaultFormats
      FastSeq(
        prettyIdentifier(cname),
        prettyIdentifier(gname),
        {
          val boundsJson = Serialization.write(partitioner.rangeBounds.map(_.toJSON(partitioner.kType.toJSON)))
          list("Partitioner " + partitioner.kType.parsableString() + prettyStringLiteral(boundsJson))
        },
        text(errorId.toString)
      )
    case TableRename(_, rowMap, globalMap) =>
      val rowKV = rowMap.toArray
      val globalKV = globalMap.toArray
      FastSeq(prettyStrings(rowKV.map(_._1)), prettyStrings(rowKV.map(_._2)),
        prettyStrings(globalKV.map(_._1)), prettyStrings(globalKV.map(_._2)))
    case MatrixRename(_, globalMap, colMap, rowMap, entryMap) =>
      val globalKV = globalMap.toArray
      val colKV = colMap.toArray
      val rowKV = rowMap.toArray
      val entryKV = entryMap.toArray
      FastSeq(prettyStrings(globalKV.map(_._1)), prettyStrings(globalKV.map(_._2)),
        prettyStrings(colKV.map(_._1)), prettyStrings(colKV.map(_._2)),
        prettyStrings(rowKV.map(_._1)), prettyStrings(rowKV.map(_._2)),
        prettyStrings(entryKV.map(_._1)), prettyStrings(entryKV.map(_._2)))
    case TableFilterIntervals(child, intervals, keep) =>
      FastSeq(
        child.typ.keyType.parsableString(),
        prettyStringLiteral(Serialization.write(
          JSONAnnotationImpex.exportAnnotation(intervals, TArray(TInterval(child.typ.keyType)))
        )(RelationalSpec.formats)),
        Pretty.prettyBooleanLiteral(keep))
    case MatrixFilterIntervals(child, intervals, keep) =>
      FastSeq(
        child.typ.rowType.parsableString(),
        prettyStringLiteral(Serialization.write(
          JSONAnnotationImpex.exportAnnotation(intervals, TArray(TInterval(child.typ.rowKeyStruct)))
        )(RelationalSpec.formats)),
        Pretty.prettyBooleanLiteral(keep))
    case RelationalLetTable(name, _, _) => single(prettyIdentifier(name))
    case RelationalLetMatrixTable(name, _, _) => single(prettyIdentifier(name))
    case RelationalLetBlockMatrix(name, _, _) => single(prettyIdentifier(name))
    case ReadPartition(_, rowType, reader) =>
      FastSeq(rowType.parsableString(),
           prettyStringLiteral(JsonMethods.compact(reader.toJValue)))
    case WritePartition(value, writeCtx, writer) =>
      single(prettyStringLiteral(JsonMethods.compact(writer.toJValue)))
    case WriteMetadata(writeAnnotations, writer) =>
      single(prettyStringLiteral(JsonMethods.compact(writer.toJValue), elide = elideLiterals))
    case ReadValue(_, reader, reqType) =>
      FastSeq(prettyStringLiteral(JsonMethods.compact(reader.toJValue)), reqType.parsableString())
    case WriteValue(_, _, writer, _) =>
      single(prettyStringLiteral(JsonMethods.compact(writer.toJValue)))
    case MakeNDArray(_, _, _, errorId) => FastSeq(errorId.toString)

    case _ => Iterable.empty
  }

  def apply(ir: BaseIR): String = if (useSSA)
    ssaStyle(ir)
  else
    sexprStyle(ir)

  def sexprStyle(ir: BaseIR): String = {
    def prettySeq(xs: Seq[BaseIR]): Doc =
      list(xs.view.map(pretty))

    def pretty(ir: BaseIR): Doc = {

      val body: Iterable[Doc] = ir match {
        case MakeStruct(fields) =>
          fields.view.map { case (n, a) =>
            list(n, pretty(a))
          }
        case ApplyAggOp(initOpArgs, seqOpArgs, aggSig) =>
          FastSeq(prettySeq(initOpArgs), prettySeq(seqOpArgs))
        case ApplyScanOp(initOpArgs, seqOpArgs, aggSig) =>
          FastSeq(prettySeq(initOpArgs), prettySeq(seqOpArgs))
        case InitOp(i, args, aggSig) => single(prettySeq(args))
        case SeqOp(i, args, aggSig) => single(prettySeq(args))
        case InsertFields(old, fields, fieldOrder) =>
          val fieldDocs = fields.view.map { case (n, a) =>
            list(prettyIdentifier(n), pretty(a))
          }
          pretty(old) +: prettyStringsOpt(fieldOrder) +: fieldDocs
        case _ => ir.children.map(pretty).toFastSeq
      }

      /*
      val pt = ir match{
        case ir: IR => if (ir._pType != null) single(ir.pType.toString)
        case _ => Iterable.empty
      }
      list(fillSep(text(prettyClass(ir)) +: pt ++ header(ir, elideLiterals)) +: body)
      */
      list(fillSep(text(Pretty.prettyClass(ir)) +: header(ir)) +: body)
    }

    pretty(ir).render(width, ribbonWidth, maxLen)
  }

  def ssaStyle(ir: BaseIR): String = {
    def childIsStrict(ir: BaseIR, i: Int): Boolean = blockArgs(ir, i).isEmpty

    def blockArgs(ir: BaseIR, i: Int): Option[IndexedSeq[(String, String)]] = ir match {
      case If(_, _, _) =>
        if (i > 0) Some(FastSeq()) else None
      case _: Switch =>
        if (i > 0) Some(FastSeq()) else None
      case TailLoop(name, args, _, body) => if (i == args.length)
        Some(args.map { case (name, ir) => name -> "loopvar" } :+
          name -> "loop") else None
      case StreamMap(a, name, _) =>
        if (i == 1) Some(Array(name -> "elt")) else None
      case StreamZip(as, names, _, _, _) =>
        if (i == as.length) Some(names.map(_ -> "elt")) else None
      case StreamZipJoin(as, key, curKey, curVals, _) =>
        if (i == as.length)
          Some(Array(curKey -> "key", curVals -> "elts"))
        else
          None
      case StreamFor(a, name, _) =>
        if (i == 1) Some(Array(name -> "elt")) else None
      case StreamFlatMap(a, name, _) =>
        if (i == 1) Some(Array(name -> "elt")) else None
      case StreamFilter(a, name, _) =>
        if (i == 1) Some(Array(name -> "elt")) else None
      case StreamTakeWhile(a, name, _) =>
        if (i == 1) Some(Array(name -> "elt")) else None
      case StreamDropWhile(a, name, _) =>
        if (i == 1) Some(Array(name -> "elt")) else None
      case StreamFold(a, zero, accumName, valueName, _) =>
        if (i == 2) Some(Array(accumName -> "accum", valueName -> "elt")) else None
      case StreamFold2(a, accum, valueName, seq, result) =>
        if (i <= accum.length)
          None
        else if (i < 2 * accum.length + 1)
          Some(Array(valueName -> "elt") ++ accum.map { case (name, value) => name -> "accum" })
        else
          Some(accum.map { case (name, value) => name -> "accum" })
      case RunAggScan(a, name, _, _, _, _) =>
        if (i == 2 || i == 3) Some(Array(name -> "elt")) else None
      case StreamScan(a, zero, accumName, valueName, _) =>
        if (i == 2) Some(Array(accumName -> "accum", valueName -> "elt")) else None
      case StreamAggScan(a, name, _) =>
        if (i == 1) Some(FastSeq(name -> "elt")) else None
      case StreamJoinRightDistinct(ll, rr, _, _, l, r, _, _) =>
        if (i == 2) Some(Array(l -> "l_elt", r -> "r_elt")) else None
      case ArraySort(a, left, right, _) =>
        if (i == 1) Some(Array(left -> "l", right -> "r")) else None
      case AggArrayPerElement(_, elementName, indexName, _, _, _) =>
        if (i == 1) Some(Array(elementName -> "elt", indexName -> "idx")) else None
      case AggFold(zero, seqOp, combOp, accumName, otherAccumName, _) => {
        if (i == 1) Some(Array(accumName -> "accum"))
        else if (i == 2) Some(Array(accumName -> "l", otherAccumName -> "r"))
        else None
      }
      case NDArrayMap(nd, name, _) =>
        if (i == 1) Some(Array(name -> "elt")) else None
      case NDArrayMap2(l, r, lName, rName, _, _) => if (i == 2)
        Some(Array(lName -> "l_elt", rName -> "r_elt"))
      else
        None
      case CollectDistributedArray(contexts, globals, cname, gname, _, _, _, _) =>
        if (i == 2) Some(Array(cname -> "ctx", gname -> "g")) else None
      case TableAggregate(child, _) =>
        if (i == 1) Some(Array("global" -> "g", "row" -> "row")) else None
      case MatrixAggregate(child, _) =>
        if (i == 1) Some(Array("global" -> "g", "sa" -> "col", "va" -> "row", "g" -> "entry")) else None
      case TableFilter(child, _) =>
        if (i == 1) Some(Array("global" -> "g", "row" -> "row")) else None
      case TableMapGlobals(child, _) =>
        if (i == 1) Some(Array("global" -> "g")) else None
      case TableMapRows(child, _) =>
        if (i == 1) Some(Array("global" -> "g", "row" -> "row")) else None
      case TableAggregateByKey(child, _) =>
        if (i == 1) Some(Array("global" -> "g", "row" -> "row")) else None
      case TableKeyByAndAggregate(child, _, _, _, _) =>
        if (i == 1 || i == 2)
          Some(Array("global" -> "g", "row" -> "row"))
        else None
      case TableMapPartitions(child, g, p, _, _, _) =>
        if (i == 1) Some(Array(g -> "g", p -> "part")) else None
      case MatrixMapRows(child, _) =>
        if (i == 1) Some(Array("global" -> "g", "va" -> "row", "sa" -> "col", "g" -> "entry", "n_cols" -> "n_cols")) else None
      case MatrixFilterRows(child, _) =>
        if (i == 1) Some(Array("global" -> "g", "va" -> "row")) else None
      case MatrixMapCols(child, _, _) =>
        if (i == 1) Some(Array("global" -> "g", "va" -> "row", "sa" -> "col", "g" -> "entry", "n_rows" -> "n_rows")) else None
      case MatrixFilterCols(child, _) =>
        if (i == 1) Some(Array("global" -> "g", "sa" -> "col")) else None
      case MatrixMapEntries(child, _) =>
        if (i == 1) Some(Array("global" -> "g", "sa" -> "col", "va" -> "row", "g" -> "entry")) else None
      case MatrixFilterEntries(child, _) =>
        if (i == 1) Some(Array("global" -> "g", "sa" -> "col", "va" -> "row", "g" -> "entry")) else None
      case MatrixMapGlobals(child, _) =>
        if (i == 1) Some(Array("global" -> "g")) else None
      case MatrixAggregateColsByKey(child, _, _) =>
        if (i == 1)
          Some(Array("global" -> "g", "va" -> "row", "sa" -> "col", "g" -> "entry"))
        else if (i == 2)
          Some(Array("global" -> "g", "sa" -> "col"))
        else
          None
      case MatrixAggregateRowsByKey(child, _, _) =>
        if (i == 1)
          Some(Array("global" -> "g", "va" -> "row", "sa" -> "col", "g" -> "entry"))
        else if (i == 2)
          Some(Array("global" -> "g", "va" -> "row"))
        else
          None
      case BlockMatrixMap(_, eltName, _, _) =>
        if (i == 1) Some(Array(eltName -> "elt")) else None
      case BlockMatrixMap2(_, _, lName, rName, _, _) =>
        if (i == 2) Some(Array(lName -> "l", rName -> "r")) else None
      case AggLet(name, _, _, _) =>
        if (i == 1) Some(Array(name -> "")) else None
      case AggExplode(_, name, _, _) =>
        if (i == 1) Some(Array(name -> "elt")) else None
      case StreamAgg(_, name, _) =>
        if (i == 1) Some(Array(name -> "elt")) else None
      case _ =>
        None
    }

    var identCounter: Int = 0
    val idents = collection.mutable.Map.empty[String, Int]

    def getIdentBase(ir: BaseIR): String = ir match {
      case True() => "true"
      case False() => "false"
      case I32(i) => s"c$i"
      case stream if stream.typ.isInstanceOf[TStream] => "s"
      case table if table.typ.isInstanceOf[TableType] => "ht"
      case mt if mt.typ.isInstanceOf[TableType] => "mt"
      case _ => ""
    }

    def uniqueify(base: String): String = {
      if (base.isEmpty) {
        identCounter += 1
        identCounter.toString
      } else if (idents.contains(base)) {
        idents(base) += 1
        if (base.last.isDigit)
          s"${base}_${idents(base)}"
        else
          s"${base}${idents(base)}"
      } else {
        idents(base) = 1
        base
      }
    }

    def prettyWithIdent(ir: BaseIR, bindings: Env[String], prefix: String): (Doc, String) = {
      val (pre, body) = pretty(ir, bindings)
      val ident = prefix + uniqueify(getIdentBase(ir))
      val doc = vsep(pre, hsep(text(ident), "=", body))
      (doc, ident)
    }

    def prettyBlock(ir: BaseIR, newBindings: IndexedSeq[(String, String)], bindings: Env[String]): Doc = {
      val args = newBindings.map { case (name, base) => name -> s"%${uniqueify(base)}" }
      val blockBindings = bindings.bindIterable(args)
      val openBlock = if (args.isEmpty)
        text("{")
      else
        concat("{", softline, args.map(_._2).mkString("(", ", ", ") =>"))
      ir match {
        case Ref(name, _) =>
          val body = blockBindings.lookupOption(name).getOrElse(uniqueify("%undefined_ref"))
          concat(openBlock, group(nest(2, concat(line, body, line)), "}"))
        case RelationalRef(name, _) =>
          val body = blockBindings.lookupOption(name).getOrElse(uniqueify("%undefined_relational_ref"))
          concat(openBlock, group(nest(2, concat(line, body, line)), "}"))
        case _ =>
          val (pre, body) = pretty(ir, blockBindings)
          concat(openBlock, nest(2, vsep(pre, body)), line, "}")
      }
    }

    def pretty(ir: BaseIR, bindings: Env[String]): (Doc, Doc) = ir match {
      case Let(name, value, body) =>
        val (valueDoc, valueIdent) = prettyWithIdent(value, bindings, "%")
        val (bodyPre, bodyHead) = pretty(body, bindings.bind(name, valueIdent))
        (concat(valueDoc, bodyPre), bodyHead)
      case RelationalLet(name, value, body) =>
        val (valueDoc, valueIdent) = prettyWithIdent(value, bindings, "%")
        val (bodyPre, bodyHead) = pretty(body, bindings.bind(name, valueIdent))
        (concat(valueDoc, bodyPre), bodyHead)
      case RelationalLetTable(name, value, body) =>
        val (valueDoc, valueIdent) = prettyWithIdent(value, bindings, "%")
        val (bodyPre, bodyHead) = pretty(body, bindings.bind(name, valueIdent))
        (concat(valueDoc, bodyPre), bodyHead)
      case RelationalLetMatrixTable(name, value, body) =>
        val (valueDoc, valueIdent) = prettyWithIdent(value, bindings, "%")
        val (bodyPre, bodyHead) = pretty(body, bindings.bind(name, valueIdent))
        (concat(valueDoc, bodyPre), bodyHead)
      case RelationalLetBlockMatrix(name, value, body) =>
        val (valueDoc, valueIdent) = prettyWithIdent(value, bindings, "%")
        val (bodyPre, bodyHead) = pretty(body, bindings.bind(name, valueIdent))
        (concat(valueDoc, bodyPre), bodyHead)
      case _ =>
        val strictChildBodies = mutable.ArrayBuilder.make[Doc]()
        val strictChildIdents = for {
          (child, i) <- ir.children.zipWithIndex
          if childIsStrict(ir, i)
        } yield {
          child match {
            case Ref(name, _) =>
              bindings.lookupOption(name).getOrElse(uniqueify("%undefined_ref"))
            case RelationalRef(name, _) =>
              bindings.lookupOption(name).getOrElse(uniqueify("%undefined_relational_ref"))
            case _ =>
              val (body, ident) = prettyWithIdent(child, bindings, "!")
              strictChildBodies += body
              ident
          }
        }

        val nestedBlocks = (for {
          (child, i) <- ir.children.zipWithIndex
          if !childIsStrict(ir, i)
        } yield prettyBlock(child, blockArgs(ir, i).get, bindings)).toFastSeq

        val attsIterable = header(ir, elideBindings = true)
        val attributes = if (attsIterable.isEmpty) Iterable.empty else
          RichIterable.single(concat(attsIterable.intersperse[Doc]("[", ", ", "]")))

        def standardArgs = if (strictChildIdents.isEmpty)
          ""
        else
          strictChildIdents.mkString("(", ", ", ")")

        val head = ir match {
          case MakeStruct(fields) =>
            val args = (fields.map(_._1), strictChildIdents).zipped.map { (field, value) =>
              s"$field: $value"
            }.mkString("(", ", ", ")")
            hsep(text(Pretty.prettyClass(ir) + args) +: (attributes ++ nestedBlocks))
          case InsertFields(_, fields, _) =>
            val newFields = (fields.map(_._1), strictChildIdents.tail).zipped.map { (field, value) =>
              s"$field: $value"
            }.mkString("(", ", ", ")")
            val args = s" ${strictChildIdents.head} $newFields"
            hsep(text(Pretty.prettyClass(ir) + args) +: (attributes ++ nestedBlocks))
          case ir: If =>
            hsep(
              text(s"${Pretty.prettyClass(ir)} ${strictChildIdents.head}"),
              text("then"),
              nestedBlocks(0),
              text("else"),
              nestedBlocks(1))
          case _ =>
            hsep(text(Pretty.prettyClass(ir) + standardArgs) +: (attributes ++ nestedBlocks))
        }

        (hsep(strictChildBodies.result()), head)
    }

    val (pre, head) = pretty(ir, Env.empty)
    vsep(pre, head, empty).render(width, ribbonWidth, maxLen)
  }
}
