package is.hail.expr.ir

import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.functions.RelationalFunctions
import is.hail.table.Ascending
import is.hail.utils._
import org.json4s.jackson.{JsonMethods, Serialization}

object Pretty {
  def prettyStringLiteral(s: String): String =
    "\"" + StringEscapeUtils.escapeString(s) + "\""

  def prettyStrings(xs: IndexedSeq[String]): String = xs.map(prettyStringLiteral).mkString("(", " ", ")")

  def prettyStringsOpt(x: Option[IndexedSeq[String]]): String = x.map(prettyStrings).getOrElse("None")

  def prettyBooleanLiteral(b: Boolean): String =
    if (b) "True" else "False"

  def prettyClass(x: AnyRef): String =
    x.getClass.getName.split("\\.").last

  def prettyAggSignature(aggSig: AggSignature): String = {
    val sb = new StringBuilder
    sb += '('
    sb.append(prettyClass(aggSig.op))
    sb += ' '
    sb.append(aggSig.constructorArgs.map(_.parsableString()).mkString(" (", " ", ")"))
    sb.append(aggSig.initOpArgs.map(_.map(_.parsableString()).mkString(" (", " ", ")")).getOrElse(" None"))
    sb.append(aggSig.seqOpArgs.map(_.parsableString()).mkString(" (", " ", ")"))
    sb += ')'
    sb.result()
  }

  def prettyIntOpt(x: Option[Int]): String = x.map(_.toString).getOrElse("None")

  def prettyLongs(x: IndexedSeq[Long]): String = x.mkString("(", " ", ")")

  def prettyInts(x: IndexedSeq[Int]): String = x.mkString("(", " ", ")")

  def prettyBooleans(bools: IndexedSeq[Boolean]): String = prettyIdentifiers(bools.map(prettyBooleanLiteral))

  def prettyLongsOpt(x: Option[IndexedSeq[Long]]): String =
    x.map(prettyLongs).getOrElse("None")

  def prettyIdentifiers(x: IndexedSeq[String]): String = x.map(prettyIdentifier).mkString("(", " ", ")")

  def prettyIdentifiersOpt(x: Option[IndexedSeq[String]]): String = x.map(prettyIdentifiers).getOrElse("None")

  def apply(ir: BaseIR, elideLiterals: Boolean = false): String = {
    val sb = new StringBuilder

    def prettySeq(xs: Seq[BaseIR], depth: Int) {
      sb.append(" " * depth)
      sb += '('
      xs.foreachBetween(x => pretty(x, depth + 2))(sb += '\n')
      sb += ')'
    }

    def pretty(ir: BaseIR, depth: Int) {
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
        case ApplyAggOp(ctorArgs, initOpArgs, seqOpArgs, aggSig) =>
          sb += ' '
          sb.append(prettyClass(aggSig.op))
          sb += '\n'
          prettySeq(ctorArgs, depth + 2)
          sb += '\n'
          initOpArgs match {
            case Some(initOpArgs) => prettySeq(initOpArgs, depth + 2)
            case None =>
              sb.append(" " * (depth + 2))
              sb.append("None")
          }
          sb += '\n'
          prettySeq(seqOpArgs, depth + 2)
        case ApplyScanOp(ctorArgs, initOpArgs, seqOpArgs, aggSig) =>
          sb += ' '
          sb.append(prettyClass(aggSig.op))
          sb += '\n'
          prettySeq(ctorArgs, depth + 2)
          sb += '\n'
          initOpArgs match {
            case Some(initOpArgs) => prettySeq(initOpArgs, depth + 2)
            case None =>
              sb.append(" " * (depth + 2))
              sb.append("None")
          }
          sb += '\n'
          prettySeq(seqOpArgs, depth + 2)
        case InitOp(i, args, aggSig) =>
          sb += ' '
          sb.append(prettyAggSignature(aggSig))
          sb += '\n'
          pretty(i, depth + 2)
          sb += '\n'
          prettySeq(args, depth + 2)
        case SeqOp(i, args, aggSig) =>
          sb += ' '
          sb.append(prettyAggSignature(aggSig))
          sb += '\n'
          pretty(i, depth + 2)
          sb += '\n'
          prettySeq(args, depth + 2)
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
            case Str(x) => prettyStringLiteral(x)
            case Cast(_, typ) => typ.parsableString()
            case NA(typ) => typ.parsableString()
            case Literal(typ, value) =>
              s"${ typ.parsableString() } " + (
                  if (!elideLiterals)
                    s"${ prettyStringLiteral(JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(value, typ))) }"
                  else
                    "<literal value>"
                )
            case Let(name, _, _) => prettyIdentifier(name)
            case AggLet(name, _, _) => prettyIdentifier(name)
            case Ref(name, _) => prettyIdentifier(name)
            case ApplyBinaryPrimOp(op, _, _) => prettyClass(op)
            case ApplyUnaryPrimOp(op, _) => prettyClass(op)
            case ApplyComparisonOp(op, _, _) => prettyClass(op)
            case GetField(_, name) => prettyIdentifier(name)
            case GetTupleElement(_, idx) => idx.toString
            case MakeArray(_, typ) => typ.parsableString()
            case ArrayMap(_, name, _) => prettyIdentifier(name)
            case ArrayFilter(_, name, _) => prettyIdentifier(name)
            case ArrayFlatMap(_, name, _) => prettyIdentifier(name)
            case ArrayFold(_, _, accumName, valueName, _) => prettyIdentifier(accumName) + " " + prettyIdentifier(valueName)
            case ArrayScan(_, _, accumName, valueName, _) => prettyIdentifier(accumName) + " " + prettyIdentifier(valueName)
            case ArrayLeftJoinDistinct(_, _, l, r, _, _) => prettyIdentifier(l) + " " + prettyIdentifier(r)
            case ArrayFor(_, valueName, _) => prettyIdentifier(valueName)
            case ArrayAgg(a, name, query) => prettyIdentifier(name)
            case AggExplode(_, name, _) => prettyIdentifier(name)
            case AggArrayPerElement(_, name, _) => prettyIdentifier(name)
            case ArraySort(_, l, r, _) => prettyIdentifier(l) + " " + prettyIdentifier(r)
            case ApplyIR(function, _, _) => prettyIdentifier(function)
            case Apply(function, _) => prettyIdentifier(function)
            case ApplySeeded(function, _, seed) => prettyIdentifier(function) + " " + seed.toString
            case ApplySpecial(function, _) => prettyIdentifier(function)
            case SelectFields(_, fields) => fields.map(prettyIdentifier).mkString("(", " ", ")")
            case LowerBoundOnOrderedCollection(_, _, onKey) => prettyBooleanLiteral(onKey)
            case In(i, typ) => s"${ typ.parsableString() } $i"
            case Die(message, typ) => typ.parsableString()
            case Uniroot(name, _, _, _) => prettyIdentifier(name)
            case MatrixRead(typ, dropCols, dropRows, reader) =>
              (if (typ == reader.fullType) "None" else typ.parsableString()) + " " +
              prettyBooleanLiteral(dropCols) + " " +
              prettyBooleanLiteral(dropRows) + " " +
              '"' + StringEscapeUtils.escapeString(Serialization.write(reader)(MatrixReader.formats)) + '"'
            case MatrixWrite(_, writer) =>
              '"' + StringEscapeUtils.escapeString(Serialization.write(writer)(MatrixWriter.formats)) + '"'
            case MatrixMultiWrite(_, writer) =>
              '"' + StringEscapeUtils.escapeString(Serialization.write(writer)(MatrixNativeMultiWriter.formats)) + '"'
            case BlockMatrixRead(path) =>
              prettyStringLiteral(path)
            case BlockMatrixWrite(_, path, overwrite, forceRowMajor, stageLocally) =>
              prettyStringLiteral(path) + " " +
              prettyBooleanLiteral(overwrite) + " " +
              prettyBooleanLiteral(forceRowMajor) + " " +
              prettyBooleanLiteral(stageLocally)
            case BlockMatrixBroadcast(_, inIndexExpr, shape, blockSize, dimsPartitioned) =>
              prettyInts(inIndexExpr) + " " +
              prettyLongs(shape) + " " +
              blockSize.toString + " " +
              prettyBooleans(dimsPartitioned)
            case ValueToBlockMatrix(_, shape, blockSize, dimsPartitioned) =>
              prettyLongs(shape) + " " +
              blockSize.toString + " " +
              prettyBooleans(dimsPartitioned)
            case MatrixRowsHead(_, n) => n.toString
            case MatrixAnnotateRowsTable(_, _, uid) =>
              prettyStringLiteral(uid) + " "
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
                '"' + StringEscapeUtils.escapeString(Serialization.write(tr)(TableReader.formats)) + '"'
            case TableWrite(_, path, overwrite, stageLocally, codecSpecJSONStr) =>
              prettyStringLiteral(path) + " " +
                prettyBooleanLiteral(overwrite) + " " +
                prettyBooleanLiteral(stageLocally) + " " +
                (if (codecSpecJSONStr == null)
                  "None"
                else
                  prettyStringLiteral(codecSpecJSONStr))
            case TableExport(_, path, typesFile, header, exportType, delimiter) =>
              prettyStringLiteral(path) + " " +
                (if (typesFile == null) "None" else prettyStringLiteral(typesFile)) + " " +
                prettyBooleanLiteral(header) + " " +
                exportType.toString + " " +
                prettyStringLiteral(delimiter)
            case TableKeyBy(_, keys, isSorted) =>
              prettyIdentifiers(keys) + " " +
                prettyBooleanLiteral(isSorted)
            case TableRange(n, nPartitions) => s"$n $nPartitions"
            case TableRepartition(_, n, strategy) => s"$n $strategy"
            case TableHead(_, n) => n.toString
            case TableJoin(_, _, joinType, joinKey) => s"$joinType $joinKey"
            case TableLeftJoinRightDistinct(_, _, root) => prettyIdentifier(root)
            case TableIntervalJoin(_, _, root) => prettyIdentifier(root)
            case TableMultiWayZipJoin(_, dataName, globalName) =>
              s"${ prettyStringLiteral(dataName) } ${ prettyStringLiteral(globalName) }"
            case TableKeyByAndAggregate(_, _, _, nPartitions, bufferSize) =>
              prettyIntOpt(nPartitions) + " " + bufferSize.toString
            case TableExplode(_, path) => prettyStrings(path)
            case TableParallelize(_, nPartitions) =>
                prettyIntOpt(nPartitions)
            case TableOrderBy(_, sortFields) => prettyIdentifiers(sortFields.map(sf =>
              (if (sf.sortOrder == Ascending) "A" else "D") + sf.field))
            case CastMatrixToTable(_, entriesFieldName, colsFieldName) =>
              s"${ prettyStringLiteral(entriesFieldName) } ${ prettyStringLiteral(colsFieldName) }"
            case CastTableToMatrix(_, entriesFieldName, colsFieldName, colKey) =>
              s"${ prettyIdentifier(entriesFieldName) } ${ prettyIdentifier(colsFieldName) } " +
                prettyIdentifiers(colKey)
            case MatrixToMatrixApply(_, function) => prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats))
            case MatrixToTableApply(_, function) => prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats))
            case TableToTableApply(_, function) => prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats))
            case TableToValueApply(_, function) => prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats))
            case MatrixToValueApply(_, function) => prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats))
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
