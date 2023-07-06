package is.hail.backend

import is.hail.HailFeatureFlags
import is.hail.expr.ir.analyses.SemanticHash
import is.hail.io.fs.FS
import is.hail.utils.{Logging, using}

import java.io.{FileNotFoundException, OutputStream}
import java.util.Base64
import java.util.concurrent.ConcurrentHashMap
import scala.io.Source
import scala.util.control.NonFatal


trait ExecutionCache extends Serializable {
  def lookup(s: SemanticHash.Type): IndexedSeq[(Array[Byte], Int)]
  def put(s: SemanticHash.Type, r: IndexedSeq[(Array[Byte], Int)]): Unit
}

case object ExecutionCache {
  case object Flags {
    val UseFastRestarts = "use_fast_restarts"
    val Cachedir = "cachedir"
  }

  def fromFlags(flags: HailFeatureFlags, fs: FS, tmpdir: String): ExecutionCache =
    if (Option(flags.get(Flags.UseFastRestarts)).isEmpty) noCache
    else fsCache(fs, Option(flags.get(Flags.Cachedir)).getOrElse(s"$tmpdir/hail"))

  def fsCache(fs: FS, cachedir: String): ExecutionCache = {
    assert(fs.validUrl(cachedir), s"""Invalid execution cache location (${fs.getClass.getSimpleName}): "$cachedir".""")
    FSExecutionCache(fs, s"$cachedir/${is.hail.HAIL_PIP_VERSION}")
  }

  def noCache: ExecutionCache = new ExecutionCache {
    override def lookup(s: SemanticHash.Type): IndexedSeq[(Array[Byte], Int)] =
      IndexedSeq.empty

    override def put(s: SemanticHash.Type, r: IndexedSeq[(Array[Byte], Int)]): Unit =
      ()
  }

  def forTesting: ExecutionCache = new ExecutionCache {
    val storage = new ConcurrentHashMap[SemanticHash.Type, IndexedSeq[(Array[Byte], Int)]]

    override def lookup(s: SemanticHash.Type): IndexedSeq[(Array[Byte], Int)] =
      storage.getOrDefault(s, IndexedSeq.empty)

    override def put(s: SemanticHash.Type, r: IndexedSeq[(Array[Byte], Int)]): Unit =
      storage.put(s, r)
  }
}

private case class FSExecutionCache(fs: FS, cacheDir: String)
  extends ExecutionCache
    with Logging {

  private val base64Encode: Array[Byte] => Array[Byte] =
    Base64.getUrlEncoder.encode

  private val base64Decode: String => Array[Byte] =
    Base64.getUrlDecoder.decode

  override def lookup(s: SemanticHash.Type): IndexedSeq[(Array[Byte], Int)] =
    try {
      using(fs.open(at(s))) {
        Source.fromInputStream(_).getLines().map(Line.read).toIndexedSeq
      }
    } catch {
      case _: FileNotFoundException =>
        IndexedSeq.empty

      case NonFatal(t) =>
        log.warn(s"Failed to read cache entry for $s", t)
        IndexedSeq.empty
    }

  override def put(s: SemanticHash.Type, r: IndexedSeq[(Array[Byte], Int)]): Unit =
    fs.write(at(s)) { ostream => r.foreach(Line.write(_, ostream)) }

  private def at(s: SemanticHash.Type): String =
    s"$cacheDir/${base64Encode(s.toString.getBytes).mkString}"

  private case object Line {
    private type Type = (Array[Byte], Int)
    def write(entry: Type, ostream: OutputStream): Unit = {
      ostream.write(entry._2.toString.getBytes)
      ostream.write(','.toInt)
      ostream.write(base64Encode(entry._1))
      ostream.write('\n'.toInt)
    }

    def read(string: String): Type = {
      val Array(index, bytes) = string.split(",")
      (base64Decode(bytes), index.toInt)
    }
  }
}

