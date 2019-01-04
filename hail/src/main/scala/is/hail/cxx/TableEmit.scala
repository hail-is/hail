package is.hail.cxx
import java.io.InputStream

import is.hail.HailContext
import is.hail.annotations.BroadcastRow
import is.hail.expr.ir
import is.hail.expr.ir.RowSym
import is.hail.expr.types._
import is.hail.io.CodecSpec
import is.hail.rvd.AbstractRVDSpec
import is.hail.table.TableSpec
import is.hail.utils._
import is.hail.variant.{FileFormat, PartitionCountsComponentSpec, RVDComponentSpec, ReferenceGenome}
import org.json4s.jackson.JsonMethods

case class TableEmitTriplet(typ: TableType, rvdEmitTriplet: RVDEmitTriplet, globals: BroadcastRow) {

  def write(tub: TranslationUnitBuilder, path: String, overwrite: Boolean, stageLocally: Boolean, codecSpecJSONStr: String) {
    val hc = HailContext.get

    val codecSpec =
      if (codecSpecJSONStr != null) {
        implicit val formats = AbstractRVDSpec.formats
        val codecSpecJSON = JsonMethods.parse(codecSpecJSONStr)
        codecSpecJSON.extract[CodecSpec]
      } else
        CodecSpec.default

    if (overwrite)
      hc.hadoopConf.delete(path, recursive = true)
    else if (hc.hadoopConf.exists(path))
      fatal(s"file already exists: $path")

    hc.hadoopConf.mkDir(path)

    val globalsPath = path + "/globals"
    hc.hadoopConf.mkDir(globalsPath)
    AbstractRVDSpec.writeLocal(hc, globalsPath, typ.globalType.physicalType, codecSpec, Array(globals.value))

    val partitionCounts = RVDEmitTriplet.write(rvdEmitTriplet, tub, path + "/rows", stageLocally, codecSpec)

    val referencesPath = path + "/references"
    hc.hadoopConf.mkDir(referencesPath)
    ReferenceGenome.exportReferences(hc, referencesPath, typ.rowType)
    ReferenceGenome.exportReferences(hc, referencesPath, typ.globalType)

    val spec = TableSpec(
      FileFormat.version.rep,
      hc.version,
      "references",
      typ,
      Map("globals" -> RVDComponentSpec("globals"),
        "rows" -> RVDComponentSpec("rows"),
        "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
    spec.write(hc, path)

    hc.hadoopConf.writeTextFile(path + "/_SUCCESS")(out => ())
  }
}

object TableEmit {
  def apply(tub: TranslationUnitBuilder, x: ir.TableIR): TableEmitTriplet = {
    val emitter = new TableEmitter(tub)
    emitter.emit(x)
  }
}

class TableEmitter(tub: TranslationUnitBuilder) { outer =>
  type E = ir.Env[TableEmitTriplet]

  def emit(x: ir.TableIR): TableEmitTriplet = emit(x, ir.Env.empty[TableEmitTriplet])

  def emit(x: ir.TableIR, env: E): TableEmitTriplet = {

    def emit(x: ir.TableIR, env: E = env): TableEmitTriplet = this.emit(x, env)

    val typ = x.typ
    x match {
      case ir.TableRead(path, spec, _, dropRows) =>
        val hc = HailContext.get
        val globals = spec.globalsComponent.readLocal(hc, path, typ.globalType.physicalType)(0)
        val rvd = if (dropRows)
          RVDEmitTriplet.empty[InputStream](tub, typ.canonicalRVDType)
        else {
          val rvd = spec.rowsComponent.cxxEmitRead(hc, path, typ.rowType, tub)
//          if (rvd.typ.key startsWith typ.key)
//            rvd
//          else {
//            log.info("Sorting a table after read. Rewrite the table to prevent this in the future.")
//            rvd.changeKey(typ.key)
//          }
          rvd
        }
        new TableEmitTriplet(typ, rvd, BroadcastRow(globals, typ.globalType, hc.sc))

      case ir.TableMapRows(child, newRow) =>
        val prev = emit(child)
        val rvd = prev.rvdEmitTriplet
        val oldRowIt = rvd.iterator

        val mapName = tub.genSym("MapRowIterator")
        val mapper = tub.buildClass(mapName)

        val region = tub.variable("region", "ScalaRegion *")
        val prevIt = tub.variable("it", oldRowIt.typ)
        mapper += region
        mapper += prevIt

        mapper +=
          s"""
             |$mapName(ScalaRegion * region, ${ oldRowIt.typ } it) :
             |$region(region), $prevIt(it) { }
           """.stripMargin

        val mapF = tub.buildFunction("map_row", Array("ScalaRegion*" -> "region", "const char *" -> "row"), "char *")
        val substEnv = ir.Env.empty[ir.IR].bind(RowSym, ir.In(0, child.typ.rowType))
        val et = Emit(mapF, 1, ir.Subst(newRow, substEnv))
        mapF +=
          s"""
             |${ et.setup }
             |if (${ et.m }) {
             |  ${ mapF.nativeError("mapped row can't be missing!") }
             |} else {
             |  return ${ et.v };
             |}
           """.stripMargin
        mapF.end()

        mapper += new Function(s"$mapName&", "operator++", Array(), s"++$prevIt; return *this;")
        mapper += new Function(s"char const*", "operator*", Array(), s"return map_row($region, *$prevIt);")
        val lhs = tub.variable("lhs", s"$mapName&")
        val rhs = tub.variable("rhs", s"$mapName&")
        mapper += new Function(s"friend bool", "operator==", Array(lhs, rhs), s"return $lhs.$prevIt == $rhs.$prevIt;")
        mapper += new Function(s"friend bool", "operator!=", Array(lhs, rhs), s"return !($lhs == $rhs);")

        mapper.end()

        val newIt = tub.variable("mapIt", mapName, s"{${ rvd.region }, $oldRowIt}")
        val newEnd = tub.variable("end", mapName, s"{${ rvd.region }, ${ rvd.end }}")
        val newSetup =
          s"""
             |${ rvd.setup }
             |${ newIt.define }
             |${ newEnd.define }
           """.stripMargin

        prev.copy(
          typ = typ,
          rvdEmitTriplet = rvd.copy(
            typ = typ.canonicalRVDType,
            setup = newSetup,
            iterator = newIt,
            end = newEnd))

      case ir.TableFilter(child, cond) =>
        val prev = emit(child)
        val rvd = prev.rvdEmitTriplet
        val oldRowIt = rvd.iterator

        val filterName = tub.genSym("FilterRowIterator")
        val filter = tub.buildClass(filterName)

        val region = tub.variable("region", "ScalaRegion *")
        val prevIt = tub.variable("it", oldRowIt.typ)
        val endIt = tub.variable("end", oldRowIt.typ)
        filter += region
        filter += prevIt
        filter += endIt

        filter +=
          s"""
             |$filterName(ScalaRegion * region, ${ oldRowIt.typ } it, ${ oldRowIt.typ } end) :
             |$region(region), $prevIt(it), $endIt(end) {
             |  while(($prevIt != $endIt) && !keep_row($region, *$prevIt)) {
             |    ++$prevIt;
             |  }
             |}
           """.stripMargin

        val filterF = tub.buildFunction("keep_row", Array("ScalaRegion*" -> "region", "const char *" -> "row"), "bool")
        val substEnv = ir.Env.empty[ir.IR].bind(RowSym, ir.In(0, child.typ.rowType))
        val et = Emit(filterF, 1, ir.Subst(cond, substEnv))
        filterF +=
          s"""
             |${ et.setup }
             |return !(${ et.m }) && (${ et.v });
           """.stripMargin
        filterF.end()

        filter += new Function(s"$filterName&", "operator++", Array(),
          s"""
             |do {
             |  ++$prevIt;
             |} while(($prevIt != $endIt) && !keep_row($region, *$prevIt));
             |return *this;
           """.stripMargin)
        filter += new Function(s"char const*", "operator*", Array(), s"return *$prevIt;")
        val lhs = tub.variable("lhs", s"$filterName&")
        val rhs = tub.variable("rhs", s"$filterName&")
        filter += new Function(s"friend bool", "operator==", Array(lhs, rhs), s"return $lhs.$prevIt == $rhs.$prevIt;")
        filter += new Function(s"friend bool", "operator!=", Array(lhs, rhs), s"return !($lhs == $rhs);")

        filter.end()

        val newIt = tub.variable("filterIt", filterName, s"{${ rvd.region }, $oldRowIt, ${ rvd.end }}")
        val newEnd = tub.variable("end", filterName, s"{${ rvd.region }, ${ rvd.end }, ${ rvd.end }}")
        val newSetup =
          s"""
             |${ rvd.setup }
             |${ newIt.define }
             |${ newEnd.define }
           """.stripMargin

        prev.copy(
          rvdEmitTriplet = rvd.copy(
            setup = newSetup,
            iterator = newIt,
            end = newEnd))
      case _ =>
        throw new CXXUnsupportedOperation(ir.Pretty(x))
    }
  }
}
