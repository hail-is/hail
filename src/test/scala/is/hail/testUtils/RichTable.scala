package is.hail.testUtils

import is.hail.annotations.Inserter
import is.hail.expr._
import is.hail.expr.ir.{InsertFields, MakeStruct, Ref}
import is.hail.expr.types.TStruct
import is.hail.table.Table
import is.hail.utils._
import org.apache.spark.sql.Row

class RichTable(ht: Table) {
  def forall(code: String): Boolean = {
    val ec = ht.rowEvalContext()
    ec.set(0, ht.globals.value)

    val f: () => java.lang.Boolean = Parser.parseTypedExpr[java.lang.Boolean](code, ec)(boxedboolHr)

    ht.rdd.forall { a =>
      ec.set(1, a)
      val b = f()
      if (b == null)
        false
      else
        b
    }
  }

  def exists(code: String): Boolean = {
    val ec = ht.rowEvalContext()
    ec.set(0, ht.globals.value)
    val f: () => java.lang.Boolean = Parser.parseTypedExpr[java.lang.Boolean](code, ec)(boxedboolHr)

    ht.rdd.exists { a =>
      ec.set(1, a)
      val b = f()
      if (b == null)
        false
      else
        b
    }
  }

  def rename(rowUpdateMap: Map[String, String], globalUpdateMap: Map[String, String]): Table = {
    select(ht.fieldNames.map(n => s"${ rowUpdateMap.getOrElse(n, n) } = row.$n"))
      .keyBy(ht.key.map(_.map(k => rowUpdateMap.getOrElse(k, k)).toArray))
      .selectGlobal(ht.globalSignature.fieldNames.map(n => s"${ globalUpdateMap.getOrElse(n, n) } = global.$n"))
  }

  def select(exprs: Array[String]): Table = {
    val ec = ht.rowEvalContext()

    val (paths, types, f) = Parser.parseSelectExprs(exprs, ec)

    val insertionPaths = paths.map {
      case Left(name) => name
      case Right(path) => path.last
    }

    val overlappingPaths = paths.counter().filter { case (n, i) => i != 1 }.keys

    if (overlappingPaths.nonEmpty)
      fatal(s"Found ${ overlappingPaths.size } ${ plural(overlappingPaths.size, "selected field name") } that are duplicated.\n" +
        "Overlapping fields:\n  " +
        s"@1", overlappingPaths.truncatable("\n  "))

    val inserterBuilder = new ArrayBuilder[Inserter]()

    val finalSignature = (insertionPaths, types).zipped.foldLeft(TStruct()) { case (vs, (p, sig)) =>
      val (s: TStruct, i) = vs.insert(sig, p)
      inserterBuilder += i
      s
    }

    val inserters = inserterBuilder.result()
    val globalsBc = ht.globals.broadcast

    val annotF: Row => Row = { r =>
      ec.setAll(globalsBc.value, r)

      f().zip(inserters)
        .foldLeft(Row()) { case (a1, (v, inserter)) =>
          inserter(a1, v).asInstanceOf[Row]
        }
    }

    var newKey = ht.key.map(_.filter(insertionPaths.toSet))
    if (newKey.exists(_.isEmpty)) newKey = None

    ht.copy(rdd = ht.rdd.map(annotF), signature = finalSignature, key = newKey)
  }

  def keyByExpr(exprs: (String, String)*): Table =
    ht.select(s"annotate(row, {${ exprs.map { case (n, e) => s"${ prettyIdentifier(n) }: $e" }.mkString(",") }})", Some(exprs.map(_._1).toIndexedSeq), None)

  def annotate(exprs: (String, String)*): Table =
    ht.select(s"annotate(row, {${ exprs.map { case (n, e) => s"${ prettyIdentifier(n) }: $e" }.mkString(",") }})", ht.key, ht.key.map(_.length))

  def selectGlobal(fields: Array[String]): Table = {
    val ec = EvalContext("global" -> ht.globalSignature)
    ec.set(0, ht.globals.value)

    val (paths, types, f) = Parser.parseSelectExprs(fields, ec)

    val names = paths.map {
      case Left(n) => n
      case Right(l) => l.last
    }

    val overlappingPaths = names.counter().filter { case (n, i) => i != 1 }.keys

    if (overlappingPaths.nonEmpty)
      fatal(s"Found ${ overlappingPaths.size } ${ plural(overlappingPaths.size, "selected field name") } that are duplicated.\n" +
        "Overlapping fields:\n  " +
        s"@1", overlappingPaths.truncatable("\n  "))

    val inserterBuilder = new ArrayBuilder[Inserter]()

    val finalSignature = (names, types).zipped.foldLeft(TStruct()) { case (vs, (p, sig)) =>
      val (s: TStruct, i) = vs.insert(sig, p)
      inserterBuilder += i
      s
    }

    val inserters = inserterBuilder.result()

    val newGlobal = f().zip(inserters)
      .foldLeft(Row()) { case (a1, (v, inserter)) =>
        inserter(a1, v).asInstanceOf[Row]
      }

    ht.copy2(globalSignature = finalSignature, globals = ht.globals.copy(value = newGlobal, t = finalSignature))
  }

  def annotateGlobalExpr(expr: String): Table = {
    val ec = EvalContext("global" -> ht.globalSignature)
    ec.set(0, ht.globals.value)

    val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, None)

    val inserterBuilder = new ArrayBuilder[Inserter]()

    val finalType = (paths, types).zipped.foldLeft(ht.globalSignature) { case (v, (ids, signature)) =>
      val (s, i) = v.insert(signature, ids)
      inserterBuilder += i
      s.asInstanceOf[TStruct]
    }

    val inserters = inserterBuilder.result()

    val ga = inserters
      .zip(f())
      .foldLeft(ht.globals.value) { case (a, (ins, res)) =>
        ins(a, res).asInstanceOf[Row]
      }

    ht.copy2(globals = ht.globals.copy(value = ga, t = finalType),
      globalSignature = finalType)
  }
}
