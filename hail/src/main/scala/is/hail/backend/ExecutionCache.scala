package is.hail.backend

import is.hail.HailFeatureFlags
import is.hail.expr.ir.analyses.SemanticHash.Hash
import is.hail.expr.ir.analyses.SemanticHash.Hash.Type
import is.hail.io.fs.FS
import is.hail.utils.using

import java.io.OutputStream
import java.util.Base64
import java.util.concurrent.ConcurrentHashMap
import scala.io.Source
import scala.util.Try

case object ExecutionCache {
  case object Flags {
    val UseFastRestarts = "use_fast_restarts"
    val Cachedir = "cachedir"
  }

  def fromFlags(flags: HailFeatureFlags, fs: FS, tmpdir: String): ExecutionCache =
    if (Option(flags.get(Flags.UseFastRestarts)).isEmpty) noCache
    else fsCache(fs, Option(flags.get(Flags.Cachedir)).getOrElse(s"$tmpdir/hail"))

  def fsCache(fs: FS, cachedir: String): ExecutionCache = {
    assert(fs.validUrl(cachedir))
    FSExecutionCache(fs, s"$cachedir/${is.hail.HAIL_PIP_VERSION}")
  }

  def noCache: ExecutionCache = new ExecutionCache {
    override def lookup(s: Type): IndexedSeq[(Int, Array[Byte])] =
      IndexedSeq.empty

    override def put(s: Type, r: IndexedSeq[(Int, Array[Byte])]): Unit =
      ()
  }

  def forTesting: ExecutionCache = new ExecutionCache {
    val storage = new ConcurrentHashMap[Hash.Type, IndexedSeq[(Int, Array[Byte])]]

    override def lookup(s: Hash.Type): IndexedSeq[(Int, Array[Byte])] =
      storage.getOrDefault(s, IndexedSeq.empty)

    override def put(s: Hash.Type, r: IndexedSeq[(Int, Array[Byte])]): Unit =
      storage.put(s, r)
  }
}

trait ExecutionCache extends Serializable {
  def lookup(s: Hash.Type): IndexedSeq[(Int, Array[Byte])]
  def put(s: Hash.Type, r: IndexedSeq[(Int, Array[Byte])]): Unit
}

private case class FSExecutionCache(fs: FS, cacheDir: String) extends ExecutionCache {

  private val base64Encode: Array[Byte] => Array[Byte] =
    Base64.getUrlEncoder.encode

  private val base64Decode: String => Array[Byte] =
    Base64.getUrlDecoder.decode

  override def lookup(s: Hash.Type): IndexedSeq[(Int, Array[Byte])] =
    Try {
      using(fs.open(at(s))) {
        Source.fromInputStream(_).getLines().map(Line.read).toIndexedSeq
      }
    }.getOrElse(IndexedSeq.empty)

  override def put(s: Hash.Type, r: IndexedSeq[(Int, Array[Byte])]): Unit =
    fs.write(at(s)) { ostream => r.foreach(Line.write(_, ostream)) }

  private def at(s: Hash.Type): String =
    s"$cacheDir/${base64Encode(s.toString.getBytes).mkString}"

  private case object Line {
    private type Type = (Hash.Type, Array[Byte])
    def write(line: Type, ostream: OutputStream): Unit = {
      ostream.write(line._1.toString.getBytes)
      ostream.write(','.toInt)
      ostream.write(base64Encode(line._2))
      ostream.write('\n'.toInt)
    }

    def read(string: String): Type = {
      val Array(index, bytes) = string.split(", ")
      (index.toInt, base64Decode(bytes))
    }
  }
}

