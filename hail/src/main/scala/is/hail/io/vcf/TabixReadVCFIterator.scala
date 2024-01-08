package is.hail.io.vcf

import is.hail.annotations.{Region, RegionValueBuilder, SafeRow}
import is.hail.backend.HailStateManager
import is.hail.expr.ir.{CloseableIterator, GenericLine}
import is.hail.io.fs.{FS, Positioned}
import is.hail.io.tabix.{TabixLineIterator, TabixReader}
import is.hail.types.physical.PStruct
import is.hail.utils.{fatal, makeJavaSet, MissingArrayBuilder, TextInputFilterAndReplace}
import is.hail.variant.ReferenceGenome

class TabixReadVCFIterator(
  fs: FS,
  file: String,
  contigMapping: Map[String, String],
  fileNum: Int,
  chrom: String,
  start: Int,
  end: Int,
  sm: HailStateManager,
  partitionRegion: Region,
  elementRegion: Region,
  requestedPType: PStruct,
  filterAndReplace: TextInputFilterAndReplace,
  infoFlagFieldNames: Set[String],
  nSamples: Int,
  _rg: ReferenceGenome,
  arrayElementsRequired: Boolean,
  skipInvalidLoci: Boolean,
  entriesFieldName: String,
  uidFieldName: String,
) {
  val chromToQuery = contigMapping.iterator.find(_._2 == chrom).map(_._1).getOrElse(chrom)

  val rg = Option(_rg)

  val reg = {
    val r = new TabixReader(file, fs)
    val tid = r.chr2tid(chromToQuery)
    r.queryPairs(tid, start - 1, end)
  }

  val linesIter = if (reg.isEmpty) {
    CloseableIterator.empty
  } else {
    new CloseableIterator[GenericLine] {
      private[this] val lines = new TabixLineIterator(fs, file, reg)
      private[this] var l = lines.next()
      private[this] var curIdx: Long = lines.getCurIdx()
      private[this] val inner = new Iterator[GenericLine] {
        def hasNext: Boolean = l != null

        def next(): GenericLine = {
          assert(l != null)
          try {
            val n = l
            val idx = curIdx
            l = lines.next()
            curIdx = lines.getCurIdx()
            if (l == null)
              lines.close()
            val bytes = n.getBytes
            new GenericLine(file, 0, idx, bytes, bytes.length)
          } catch {
            case e: Exception => fatal(s"error reading file: $file at ${lines.getCurIdx()}", e)
          }
        }
      }.filter { gl =>
        val s = gl.toString
        val t1 = s.indexOf('\t')
        val t2 = s.indexOf('\t', t1 + 1)

        if (t1 == -1 || t2 == -1) {
          fatal(
            s"invalid line in file ${gl.file} no CHROM or POS column at offset ${gl.offset}.\n$s"
          )
        }

        val chr = s.substring(0, t1)
        val pos = s.substring(t1 + 1, t2).toInt

        if (chr != chrom) {
          fatal(s"in file ${gl.file} at offset ${gl.offset}, bad chromosome! $chrom, $s")
        }
        start <= pos && pos <= end
      }

      def hasNext: Boolean = inner.hasNext

      def next(): GenericLine = inner.next()

      def close(): Unit = lines.close()
    }
  }

  val transformer = filterAndReplace.transformer()

  val parseLineContext = new ParseLineContext(
    requestedPType.virtualType,
    makeJavaSet(infoFlagFieldNames),
    nSamples,
    fileNum,
    entriesFieldName,
  )

  val rvb = new RegionValueBuilder(sm)

  val abs = new MissingArrayBuilder[String]
  val abi = new MissingArrayBuilder[Int]
  val abf = new MissingArrayBuilder[Float]
  val abd = new MissingArrayBuilder[Double]

  // 0 if EOS
  def next(elementRegion: Region): Long = {

    var done = false
    while (!done && linesIter.hasNext) {
      val line = linesIter.next()
      val text = line.toString
      val newText = transformer(text)
      done = (newText != null) && {
        rvb.clear()
        rvb.set(elementRegion)
        try {
          val vcfLine = new VCFLine(
            newText,
            line.fileNum,
            line.offset,
            arrayElementsRequired,
            abs,
            abi,
            abf,
            abd,
          )
          val pl = LoadVCF.parseLine(rg, contigMapping, skipInvalidLoci,
            requestedPType, rvb, parseLineContext, vcfLine, entriesFieldName, uidFieldName)
          pl
        } catch {
          case e: Exception =>
            fatal(
              s"${line.file}:offset ${line.offset}: error while parsing line\n" +
                s"$newText\n",
              e,
            )
        }
      }
    }

    if (done) {
      rvb.result().offset
    } else
      0L
  }

  def close(): Unit =
    linesIter.close()
}
