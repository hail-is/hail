package is.hail.expr.ir

import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.agg.{AggElementsAggSig, AggStateSig, ArrayLenAggSig, GroupedAggSig, PhysicalAggSig}
import is.hail.expr.ir.functions.RelationalFunctions
import is.hail.types.virtual.{TArray, TInterval, Type}
import is.hail.utils._
import org.json4s.jackson.{JsonMethods, Serialization}

object Pretty {

  def short(ir: BaseIR, elideLiterals: Boolean = false, maxLen: Int = 100): String = {
    val s = Pretty(ir, elideLiterals = elideLiterals, maxLen = maxLen)
    if (s.length < maxLen) s else s.substring(0, maxLen) + "..."
  }

  def prettyStringLiteral(s: String): String =
    "\"" + StringEscapeUtils.escapeString(s) + "\""

  def prettyStrings(xs: IndexedSeq[String]): String = xs.map(prettyStringLiteral).mkString("(", " ", ")")

  def prettyStringsOpt(x: Option[IndexedSeq[String]]): String = x.map(prettyStrings).getOrElse("None")

  def prettyTypes(x: Seq[Type]): String = x.map(typ => typ.parsableString()).mkString("(", " ", ")")

  def prettySortFields(x: Seq[SortField]): String = x.map(typ => typ.parsableString()).mkString("(", " ", ")")

  def prettyBooleanLiteral(b: Boolean): String =
    if (b) "True" else "False"

  def prettyClass(x: AnyRef): String =
    x.getClass.getName.split("\\.").last

  val MAX_VALUES_TO_LOG: Int = 25

  def apply(ir: BaseIR, elideLiterals: Boolean = true, maxLen: Int = -1): String = {
    val sb = new StringBuilder

    def prettyIntOpt(x: Option[Int]): String = x.map(_.toString).getOrElse("None")

    def prettyLongs(x: IndexedSeq[Long]): String = if (elideLiterals && x.length > MAX_VALUES_TO_LOG)
      x.mkString("(", " ", s"... ${ x.length - MAX_VALUES_TO_LOG } more values... )")
    else
      x.mkString("(", " ", ")")

    def prettyInts(x: IndexedSeq[Int]): String = if (elideLiterals && x.length > MAX_VALUES_TO_LOG)
      x.mkString("(", " ", s"... ${ x.length - MAX_VALUES_TO_LOG } more values... )")
    else
      x.mkString("(", " ", ")")

    def prettyIdentifiers(x: IndexedSeq[String]): String = x.map(prettyIdentifier).mkString("(", " ", ")")

    def basePrettySeq[T](xs: Seq[T], depth: Int, f: (T, Int) => Unit): Unit = {
      sb.append(" " * depth)
      sb += '('
      xs.foreach { x =>
        sb += '\n'
        f(x, depth + 2)
      }
      sb += ')'
    }
    def prettySeq(xs: Seq[BaseIR], depth: Int): Unit = basePrettySeq(xs, depth, pretty)

    def prettyAggStateSignatures(states: Seq[AggStateSig], depth: Int): Unit =
      basePrettySeq(states, depth, prettyAggStateSignature)

    def prettyAggStateSignature(state: AggStateSig, depth: Int): Unit = {
      sb.append(" " * depth)
      sb += '('
      sb.append(prettyClass(state))
      sb += ' '
      state.t.foreachBetween(typ => sb.append(typ.toString))(sb += ' ')
      if (state.n.isDefined) {
        sb += '\n'
        prettyAggStateSignatures(state.n.get, depth + 2)
      }
      sb += ')'
    }

    def prettyPhysicalAggSigs(aggSigs: Seq[PhysicalAggSig], depth: Int): Unit =
      basePrettySeq(aggSigs, depth, prettyPhysicalAggSig)

    def prettyPhysicalAggSig(aggSig: PhysicalAggSig, depth: Int): Unit = {
      sb.append(" " * depth)
      sb += '('
      aggSig match {
        case GroupedAggSig(t, nested) =>
          sb.append("Grouped")
          sb += ' '
          sb.append(t.toString)
          sb += '\n'
          prettyPhysicalAggSigs(nested, depth + 2)
        case ArrayLenAggSig(kl, nested) =>
          sb.append("ArrayLen")
          sb += ' '
          sb.append(prettyBooleanLiteral(kl))
          sb += '\n'
          prettyPhysicalAggSigs(nested, depth + 2)
        case AggElementsAggSig(nested) =>
          sb.append("AggElements")
          sb += '\n'
          prettyPhysicalAggSigs(nested, depth + 2)
        case PhysicalAggSig(op, state) =>
          sb.append(prettyClass(op))
          sb += '\n'
          prettyAggStateSignature(state, depth + 2)
      }
      sb += ')'
    }

    def pretty(ir: BaseIR, depth: Int) {
      if (maxLen > 0 && sb.size > maxLen)
        return

      sb.append(" " * depth)
      sb += '('

      sb.append(prettyClass(ir) )

      ir match {
        case MakeStruct(fields) =>
          if (fields.nonEmpty) {
            sb += '\n'
            fields.foreachBetween { case (n, a) =>
              sb.append(" " * (depth + 2))
              sb += '('
              sb.append(n)
              sb += '\n'
              pretty(a, depth + 4)
              sb += ')'
            }(sb += '\n')
          }
        case ApplyAggOp(initOpArgs, seqOpArgs, aggSig) =>
          sb += ' '
          sb.append(prettyClass(aggSig.op))
          sb += '\n'
          prettySeq(initOpArgs, depth + 2)
          sb += '\n'
          prettySeq(seqOpArgs, depth + 2)
        case ApplyScanOp(initOpArgs, seqOpArgs, aggSig) =>
          sb += ' '
          sb.append(prettyClass(aggSig.op))
          sb += '\n'
          prettySeq(initOpArgs, depth + 2)
          sb += '\n'
          prettySeq(seqOpArgs, depth + 2)
        case InitOp(i, args, aggSig) =>
          sb += ' '
          sb.append(i)
          sb += ' '
          sb += '\n'
          prettyPhysicalAggSig(aggSig, depth + 2)
          sb += '\n'
          prettySeq(args, depth + 2)
        case SeqOp(i, args, aggSig) =>
          sb += ' '
          sb.append(i)
          sb += '\n'
          prettyPhysicalAggSig(aggSig, depth + 2)
          sb += '\n'
          prettySeq(args, depth + 2)
        case CombOp(i1, i2, aggSig) =>
          sb += ' '
          sb.append(i1)
          sb += ' '
          sb.append(i2)
          sb += ' '
          prettyPhysicalAggSig(aggSig, depth + 2)
        case ResultOp(i, aggSigs) =>
          sb += ' '
          sb.append(i)
          sb += '\n'
          prettyPhysicalAggSigs(aggSigs, depth + 2)
        case AggStateValue(i, sig) =>
          sb += ' '
          sb.append(i)
          sb += ' '
          prettyAggStateSignature(sig, depth + 2)
        case InitFromSerializedValue(i, value, aggSig) =>
          sb += ' '
          sb.append(i)
          sb += ' '
          prettyAggStateSignature(aggSig, depth + 2)
          sb += ' '
          pretty(value, depth + 2)
        case CombOpValue(i, value, sig) =>
          sb += ' '
          sb.append(i)
          sb += ' '
          prettyPhysicalAggSig(sig, depth + 2)
          sb += ' '
          pretty(value, depth + 2)
        case SerializeAggs(i, i2, spec, aggSigs) =>
          sb += ' '
          sb.append(i)
          sb += ' '
          sb.append(i2)
          sb += ' '
          sb.append(prettyStringLiteral(spec.toString))
          sb += '\n'
          prettyAggStateSignatures(aggSigs, depth + 2)
        case DeserializeAggs(i, i2, spec, aggSigs) =>
          sb += ' '
          sb.append(i)
          sb += ' '
          sb.append(i2)
          sb += ' '
          sb.append(prettyStringLiteral(spec.toString))
          sb += '\n'
          prettyAggStateSignatures(aggSigs, depth + 2)
        case RunAgg(body, result, signature) =>
          prettyAggStateSignatures(signature, depth + 2)
          sb += '\n'
          pretty(body, depth + 2)
          sb += '\n'
          pretty(result, depth + 2)
        case RunAggScan(a, name, init, seq, res, signature) =>
          sb += ' '
          sb.append(prettyIdentifier(name))
          sb += ' '
          prettyAggStateSignatures(signature, depth + 2)
          sb += '\n'
          pretty(a, depth + 2)
          sb += '\n'
          pretty(init, depth + 2)
          sb += '\n'
          pretty(seq, depth + 2)
          sb += '\n'
          pretty(res, depth + 2)
        case InsertFields(old, fields, fieldOrder) =>
          sb += '\n'
          pretty(old, depth + 2)
          sb.append('\n')
          sb.append(" " * (depth + 2))
          sb.append(prettyStringsOpt(fieldOrder))
          if (fields.nonEmpty) {
            sb += '\n'
            fields.foreachBetween { case (n, a) =>
              sb.append(" " * (depth + 2))
              sb += '('
              sb.append(prettyIdentifier(n))
              sb += '\n'
              pretty(a, depth + 4)
              sb += ')'
            }(sb += '\n')
          }
        case _ =>
          val header = ir match {
            case I32(x) => x.toString
            case I64(x) => x.toString
            case F32(x) => x.toString
            case F64(x) => x.toString
            case Str(x) => prettyStringLiteral(if (elideLiterals && x.length > 13) x.take(10) + "..." else x)
            case UUID4(id) => prettyIdentifier(id)
            case Cast(_, typ) => typ.parsableString()
            case CastRename(_, typ) => typ.parsableString()
            case NA(typ) => typ.parsableString()
            case Literal(typ, value) =>
              s"${ typ.parsableString() } " + (
                  if (!elideLiterals)
                    s"${ prettyStringLiteral(JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(value, typ))) }"
                  else
                    "<literal value>"
                )
            case Let(name, _, _) => prettyIdentifier(name)
            case AggLet(name, _, _, isScan) => prettyIdentifier(name) + " " + prettyBooleanLiteral(isScan)
            case TailLoop(name, args, _) => prettyIdentifier(name) + " " + prettyIdentifiers(args.map(_._1).toFastIndexedSeq)
            case Recur(name, _, t) => prettyIdentifier(name) + " " + t.parsableString()
            case Ref(name, _) => prettyIdentifier(name)
            case RelationalRef(name, t) => prettyIdentifier(name) + " " + t.parsableString()
            case RelationalLet(name, _, _) => prettyIdentifier(name)
            case ApplyBinaryPrimOp(op, _, _) => prettyClass(op)
            case ApplyUnaryPrimOp(op, _) => prettyClass(op)
            case ApplyComparisonOp(op, _, _) => prettyClass(op)
            case GetField(_, name) => prettyIdentifier(name)
            case GetTupleElement(_, idx) => idx.toString
            case MakeTuple(fields) => prettyInts(fields.map(_._1).toFastIndexedSeq)
            case MakeArray(_, typ) => typ.parsableString()
            case MakeStream(_, typ) => typ.parsableString()
            case StreamMap(_, name, _) => prettyIdentifier(name)
            case StreamMerge(_, _, key) => prettyIdentifiers(key)
            case StreamZip(_, names, _, behavior) => prettyIdentifier(behavior match {
              case ArrayZipBehavior.AssertSameLength => "AssertSameLength"
              case ArrayZipBehavior.TakeMinLength => "TakeMinLength"
              case ArrayZipBehavior.ExtendNA => "ExtendNA"
              case ArrayZipBehavior.AssumeSameLength => "AssumeSameLength"
            }) + " " + prettyIdentifiers(names)
            case StreamZipJoin(_, key, curKey, curVals, _) =>
              s"${prettyIdentifiers(key)} ${prettyIdentifier(curKey)} ${prettyIdentifier(curVals)}"
            case StreamMultiMerge(_, key) => prettyIdentifiers(key)
            case StreamFilter(_, name, _) => prettyIdentifier(name)
            case StreamFlatMap(_, name, _) => prettyIdentifier(name)
            case StreamFold(_, _, accumName, valueName, _) => prettyIdentifier(accumName) + " " + prettyIdentifier(valueName)
            case StreamFold2(_, acc, valueName, _, _) => prettyIdentifiers(acc.map(_._1)) + " " + prettyIdentifier(valueName)
            case StreamScan(_, _, accumName, valueName, _) => prettyIdentifier(accumName) + " " + prettyIdentifier(valueName)
            case StreamJoinRightDistinct(_, _, lKey, rKey, l, r, _, joinType) =>
              s"${prettyIdentifiers(lKey)} ${prettyIdentifiers(rKey)} ${prettyIdentifier(l)} ${prettyIdentifier(r)} $joinType"
            case StreamFor(_, valueName, _) => prettyIdentifier(valueName)
            case StreamAgg(a, name, query) => prettyIdentifier(name)
            case StreamAggScan(a, name, query) => prettyIdentifier(name)
            case AggExplode(_, name, _, isScan) => prettyIdentifier(name) + " " + prettyBooleanLiteral(isScan)
            case AggFilter(_, _, isScan) => prettyBooleanLiteral(isScan)
            case AggGroupBy(_, _, isScan) => prettyBooleanLiteral(isScan)
            case AggArrayPerElement(_, elementName, indexName, _, knownLength, isScan) =>
              prettyIdentifier(elementName) + " " + prettyIdentifier(indexName) + " " + prettyBooleanLiteral(isScan) + " " + prettyBooleanLiteral(knownLength.isDefined)
            case NDArrayMap(_, name, _) => prettyIdentifier(name)
            case NDArrayMap2(_, _, lName, rName, _) => prettyIdentifier(lName) + " " + prettyIdentifier(rName)
            case NDArrayReindex(_, indexExpr) => prettyInts(indexExpr)
            case NDArrayConcat(_, axis) => axis.toString
            case NDArrayAgg(_, axes) => prettyInts(axes)
            case ArraySort(_, l, r, _) => prettyIdentifier(l) + " " + prettyIdentifier(r)
            case ApplyIR(function, typeArgs, _) => prettyIdentifier(function) + " " + prettyTypes(typeArgs) + " " + ir.typ.parsableString()
            case Apply(function, typeArgs, _, t) => prettyIdentifier(function) + " " + prettyTypes(typeArgs) + " " + t.parsableString()
            case ApplySeeded(function, _, seed, t) => prettyIdentifier(function) + " " + seed.toString + " " + t.parsableString()
            case ApplySpecial(function, typeArgs, _, t) => prettyIdentifier(function) + " " + prettyTypes(typeArgs) + " " + t.parsableString()
            case SelectFields(_, fields) => fields.map(prettyIdentifier).mkString("(", " ", ")")
            case LowerBoundOnOrderedCollection(_, _, onKey) => prettyBooleanLiteral(onKey)
            case In(i, typ) => s"$typ $i"
            case Die(message, typ) => typ.parsableString()
            case CollectDistributedArray(_, _, cname, gname, _) =>
              s"${ prettyIdentifier(cname) } ${ prettyIdentifier(gname) }"
            case MatrixRead(typ, dropCols, dropRows, reader) =>
              (if (typ == reader.fullMatrixType) "None" else typ.parsableString()) + " " +
              prettyBooleanLiteral(dropCols) + " " +
              prettyBooleanLiteral(dropRows) + " " +
                '"' + StringEscapeUtils.escapeString(JsonMethods.compact(reader.toJValue)) + '"'
            case MatrixWrite(_, writer) =>
              '"' + StringEscapeUtils.escapeString(Serialization.write(writer)(MatrixWriter.formats)) + '"'
            case MatrixMultiWrite(_, writer) =>
              '"' + StringEscapeUtils.escapeString(Serialization.write(writer)(MatrixNativeMultiWriter.formats)) + '"'
            case BlockMatrixRead(reader) =>
              '"' + StringEscapeUtils.escapeString(JsonMethods.compact(reader.toJValue)) + '"'
            case BlockMatrixWrite(_, writer) =>
              '"' + StringEscapeUtils.escapeString(Serialization.write(writer)(BlockMatrixWriter.formats)) + '"'
            case BlockMatrixMultiWrite(_, writer) =>
              '"' + StringEscapeUtils.escapeString(Serialization.write(writer)(BlockMatrixWriter.formats)) + '"'
            case BlockMatrixBroadcast(_, inIndexExpr, shape, blockSize) =>
              prettyInts(inIndexExpr) + " " +
              prettyLongs(shape) + " " +
              blockSize.toString + " "
            case BlockMatrixAgg(_, outIndexExpr) => prettyInts(outIndexExpr)
            case BlockMatrixSlice(_, slices) => slices.map(slice => prettyLongs(slice)).mkString("(", " ", ")")
            case ValueToBlockMatrix(_, shape, blockSize) =>
              prettyLongs(shape) + " " +
              blockSize.toString + " "
            case BlockMatrixFilter(_, indicesToKeepPerDim) =>
              indicesToKeepPerDim.map(indices => prettyLongs(indices)).mkString("(", " ", ")")
            case BlockMatrixSparsify(_, sparsifier) =>
              sparsifier.pretty()
            case BlockMatrixRandom(seed, gaussian, shape, blockSize) =>
              seed.toString + " " +
              prettyBooleanLiteral(gaussian) + " " +
              prettyLongs(shape) + " " +
              blockSize.toString + " "
            case BlockMatrixMap(_, name, _, needsDense) =>
              prettyIdentifier(name) + " " + prettyBooleanLiteral(needsDense)
            case BlockMatrixMap2(_, _, lName, rName, _, sparsityStrategy) =>
              prettyIdentifier(lName) + " " + prettyIdentifier(rName) + prettyClass(sparsityStrategy)
            case MatrixRowsHead(_, n) => n.toString
            case MatrixColsHead(_, n) => n.toString
            case MatrixRowsTail(_, n) => n.toString
            case MatrixColsTail(_, n) => n.toString
            case MatrixAnnotateRowsTable(_, _, uid, product) =>
              prettyStringLiteral(uid) + " " + prettyBooleanLiteral(product)
            case MatrixAnnotateColsTable(_, _, uid) =>
              prettyStringLiteral(uid)
            case MatrixExplodeRows(_, path) => prettyIdentifiers(path)
            case MatrixExplodeCols(_, path) => prettyIdentifiers(path)
            case MatrixRepartition(_, n, strategy) => s"$n $strategy"
            case MatrixChooseCols(_, oldIndices) => prettyInts(oldIndices)
            case MatrixMapCols(_, _, newKey) => prettyStringsOpt(newKey)
            case MatrixKeyRowsBy(_, keys, isSorted) =>
              prettyIdentifiers(keys) + " " +
                prettyBooleanLiteral(isSorted)
            case TableRead(typ, dropRows, tr) =>
              (if (typ == tr.fullType) "None" else typ.parsableString()) + " " +
                prettyBooleanLiteral(dropRows) + " " +
                '"' + StringEscapeUtils.escapeString(JsonMethods.compact(tr.toJValue)) + '"'
            case TableWrite(_, writer) =>
              '"' + StringEscapeUtils.escapeString(Serialization.write(writer)(TableWriter.formats)) + '"'
            case TableMultiWrite(_, writer) =>
              '"' + StringEscapeUtils.escapeString(Serialization.write(writer)(WrappedMatrixNativeMultiWriter.formats)) + '"'
            case TableKeyBy(_, keys, isSorted) =>
              prettyIdentifiers(keys) + " " +
                prettyBooleanLiteral(isSorted)
            case TableRange(n, nPartitions) => s"$n $nPartitions"
            case TableRepartition(_, n, strategy) => s"$n $strategy"
            case TableGroupWithinPartitions(_, name, n) => s"${ prettyIdentifier(name) } $n"
            case TableHead(_, n) => n.toString
            case TableTail(_, n) => n.toString
            case TableJoin(_, _, joinType, joinKey) => s"$joinType $joinKey"
            case TableLeftJoinRightDistinct(_, _, root) => prettyIdentifier(root)
            case TableIntervalJoin(_, _, root, product) =>
              prettyIdentifier(root) + " " + prettyBooleanLiteral(product)
            case TableMultiWayZipJoin(_, dataName, globalName) =>
              s"${ prettyStringLiteral(dataName) } ${ prettyStringLiteral(globalName) }"
            case TableKeyByAndAggregate(_, _, _, nPartitions, bufferSize) =>
              prettyIntOpt(nPartitions) + " " + bufferSize.toString
            case TableExplode(_, path) => prettyStrings(path)
            case TableParallelize(_, nPartitions) =>
                prettyIntOpt(nPartitions)
            case TableOrderBy(_, sortFields) => prettySortFields(sortFields)
            case CastMatrixToTable(_, entriesFieldName, colsFieldName) =>
              s"${ prettyStringLiteral(entriesFieldName) } ${ prettyStringLiteral(colsFieldName) }"
            case CastTableToMatrix(_, entriesFieldName, colsFieldName, colKey) =>
              s"${ prettyIdentifier(entriesFieldName) } ${ prettyIdentifier(colsFieldName) } " +
                prettyIdentifiers(colKey)
            case MatrixToMatrixApply(_, function) => prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats))
            case MatrixToTableApply(_, function) => prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats))
            case TableToTableApply(_, function) => prettyStringLiteral(JsonMethods.compact(function.toJValue))
            case TableToValueApply(_, function) => prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats))
            case MatrixToValueApply(_, function) => prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats))
            case BlockMatrixToTableApply(_, _, function) => prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats))
            case TableRename(_, rowMap, globalMap) =>
              val rowKV = rowMap.toArray
              val globalKV = globalMap.toArray
              s"${ prettyStrings(rowKV.map(_._1)) } ${ prettyStrings(rowKV.map(_._2)) } " +
                s"${ prettyStrings(globalKV.map(_._1)) } ${ prettyStrings(globalKV.map(_._2)) }"
            case MatrixRename(_, globalMap, colMap, rowMap, entryMap) =>
              val globalKV = globalMap.toArray
              val colKV = colMap.toArray
              val rowKV = rowMap.toArray
              val entryKV = entryMap.toArray
              s"${ prettyStrings(globalKV.map(_._1)) } ${ prettyStrings(globalKV.map(_._2)) } " +
                s"${ prettyStrings(colKV.map(_._1)) } ${ prettyStrings(colKV.map(_._2)) } " +
                s"${ prettyStrings(rowKV.map(_._1)) } ${ prettyStrings(rowKV.map(_._2)) } " +
                s"${ prettyStrings(entryKV.map(_._1)) } ${ prettyStrings(entryKV.map(_._2)) }"
            case TableFilterIntervals(child, intervals, keep) =>
              prettyStringLiteral(Serialization.write(
                JSONAnnotationImpex.exportAnnotation(intervals, TArray(TInterval(child.typ.keyType)))
              )(RelationalSpec.formats)) + " " + prettyBooleanLiteral(keep)
            case MatrixFilterIntervals(child, intervals, keep) =>
              prettyStringLiteral(Serialization.write(
                JSONAnnotationImpex.exportAnnotation(intervals, TArray(TInterval(child.typ.rowKeyStruct)))
              )(RelationalSpec.formats)) + " " + prettyBooleanLiteral(keep)
            case RelationalLetTable(name, _, _) => prettyIdentifier(name)
            case RelationalLetMatrixTable(name, _, _) => prettyIdentifier(name)
            case RelationalLetBlockMatrix(name, _, _) => prettyIdentifier(name)
            case ReadPartition(_, rowType, reader) =>
              s"${ rowType.parsableString() } ${ prettyStringLiteral(JsonMethods.compact(reader.toJValue)) }"
            case WritePartition(value, writeCtx, writer) =>
              prettyStringLiteral(JsonMethods.compact(writer.toJValue))
            case WriteMetadata(writeAnnotations, writer) =>
              prettyStringLiteral(JsonMethods.compact(writer.toJValue))
            case ReadValue(_, spec, reqType) =>
              s"${ prettyStringLiteral(spec.toString) } ${ reqType.parsableString() }"
            case WriteValue(_, _, spec) => prettyStringLiteral(spec.toString)
            case x@ShuffleWith(_, _, _, _, name, _, _) =>
              s"${ x.shuffleType.parsableString() } ${ prettyIdentifier(name) }"

            case _ => ""
          }

          if (header.nonEmpty) {
            sb += ' '
            sb.append(header)
          }

          val children = ir.children
          if (children.nonEmpty) {
            sb += '\n'
            children.foreachBetween(c => pretty(c, depth + 2))(sb += '\n')
          }
      }

      sb += ')'
    }

    pretty(ir, 0)

    sb.result()
  }
}
